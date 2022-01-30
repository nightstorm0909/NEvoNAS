import os
import sys
import logging
import random
import torch.nn as nn
import genotypes
import argparse
import numpy as np
import pandas as pd
import pickle
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import torch.utils
import torch.nn.functional as F
import time
import utils

from cell_operations                       import NAS_BENCH_201
from config_utils                          import load_config
from csv                                   import DictWriter
from datasets                              import get_datasets, get_nas_search_loaders
from nas_201_api                           import NASBench201API as API
from novelty                               import Novelty
from procedures                            import get_optim_scheduler
from search_model_nas                      import TinyNetwork_NAS
from torch.utils.tensorboard               import SummaryWriter
from pymoo.core.problem                    import ElementwiseProblem
from pymoo.algorithms.moo.nsga2            import NSGA2
from pymoo.core.evaluator                  import set_cv
from pymoo.util.termination.no_termination import NoTermination
from pymoo.operators.crossover.sbx         import SimulatedBinaryCrossover
from pymoo.operators.mutation.pm           import PolynomialMutation

parser = argparse.ArgumentParser("NAS201")
parser.add_argument('--batch_size', type = int, default = 256, help = 'batch size')
parser.add_argument('--cutout', action = 'store_true', default = False, help = 'use cutout')
parser.add_argument('--cutout_length', type = int, default = 16, help = 'cutout length')
parser.add_argument('--data', type = str, default = '../data', help = 'location of the data corpus')
parser.add_argument('--epochs', type = int, default = None, help = 'num of generations')
parser.add_argument('--gpu', type = int, default = 0, help = 'gpu device id')
parser.add_argument('--grad_clip', type = float, default = 5, help = 'gradient clipping')
parser.add_argument('--init_channels', type = int, default = 16, help = 'num of init channels')
parser.add_argument('--knn', type = int, default = 5, help = 'k-nearest neighbors')
parser.add_argument('--learning_rate', type = float, default = 0.025, help = 'init learning rate')
parser.add_argument('--learning_rate_min', type = float, default = 0.001, help = 'min learning rate')
parser.add_argument('--momentum', type = float, default = 0.9, help = 'momentum')
parser.add_argument('--mutate_rate', type = float, default = 0.1, help = 'mutation rate')
parser.add_argument('--novelty_threshold', type = float, default = 0.7, help = 'Novelty Threshold')
parser.add_argument('--output_dir', type = str, default = None, help = 'location of trials')
parser.add_argument('--pop_size', type = int, default = 20, help = 'population size')
parser.add_argument('--report_freq', type = float, default = 50, help = 'report frequency')
parser.add_argument('--record_filename', type = str, default = None, help = 'filename of the csv file for recording the final results')
parser.add_argument('--seed', type = int, default = None, help = 'random seed')
parser.add_argument('--valid_batch_size', type = int, default = 1024, help = 'validation batch size')
parser.add_argument('--weight_decay', type = float, default = 3e-4, help = 'weight decay')

# Added for NAS201
#parser.add_argument('--channel', type = int, default = 16, help = 'initial channel for NAS201 network')
parser.add_argument('--local_fitness_flag', default=False, action='store_true')
parser.add_argument('--num_cells', type = int, default = 5, help = 'number of cells for NAS201 network')
parser.add_argument('--max_nodes', type = int, default = 4, help = 'maximim nodes in the cell for NAS201 network')
parser.add_argument('--track_running_stats', action = 'store_true', default = False, help = 'use track_running_stats in BN layer')
parser.add_argument('--dataset', type = str, default = 'cifar10', help = '["cifar10", "cifar100", "ImageNet16-120"]')
parser.add_argument('--api_path', type = str, default = None, help = '["cifar10", "cifar10-valid","cifar100", "imagenet16-120"]')
parser.add_argument('--train_discrete', default=False, action='store_true')
parser.add_argument('--train_epochs', type = int, default = 0, help = 'num of training epochs')
parser.add_argument('--workers', type=int, default=2, help='number of data loading workers (default: 2)')
parser.add_argument('--config_path', type=str, help='The config path.')
parser.add_argument('--config_root', type=str, help='The root path of the config directory')
args = parser.parse_args()

def get_arch_score(api, arch_str, dataset, acc_type=None, use_012_epoch_training=False):
  arch_index = api.query_index_by_arch( arch_str )
  assert arch_index >= 0, 'can not find this arch : {:}'.format(arch_str)
  if use_012_epoch_training:
    info = api.get_more_info(arch_index, dataset, iepoch=None, hp='12', is_random=True)
    valid_acc, time_cost = info['valid-accuracy'], info['train-all-time'] + info['valid-per-time']
    return valid_acc, time_cost
  else:
    return api.query_by_index(arch_index=arch_index, hp = '200').get_metrics(dataset, acc_type)['accuracy']

def train(model, train_queue, criterion, optimizer, gen, device, pop=None):
  model.train()
  losses, top1, top5 = utils.AvgrageMeter(), utils.AvgrageMeter(), utils.AvgrageMeter()
  if pop is None:
    logging.info(f'In warm-up training')
    for step, (inputs, targets) in enumerate(train_queue):
      model.random_alphas(discrete=args.train_discrete)
      n = inputs.size(0)
      inputs = inputs.to(device)
      targets = targets.to(device)
      optimizer.zero_grad()
      _, logits = model(inputs)
      loss = criterion(logits, targets)
      loss.backward()
      nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
      optimizer.step()

      prec1, prec5 = utils.obtain_accuracy(logits.data, targets.data, topk = (1, 5))
      losses.update(loss.item(),  n)
      top1.update  (prec1.item(), n)
      top5.update  (prec5.item(), n)
      
      #print(step)
      if (step) % args.report_freq == 0:
        logging.info(f"[Epoch #{gen}]: train_discrete: {args.train_discrete}")
        logging.info(f"Using Training batch #{step} with loss: {losses.avg:.5f}, prec1: {top1.avg:.5f}, prec5: {top5.avg:.5f}")
      #break
  else:
    for step, (inputs, targets) in enumerate(train_queue):
      #Copying and checking the discretized alphas
      tx = torch.tensor(pop[step % len(pop)].X).type(torch.float).reshape_as(model.arch_parameters).to(device)
      #logging.info(f'step % len(pop): {step % len(pop)}')
      model.update_alphas(tx)
      discrete_alphas = model.discretize()
      _, df_max, _ = model.show_alphas_dataframe()
      assert np.all(np.equal(df_max.to_numpy(), discrete_alphas.cpu().numpy()))
      assert model.check_alphas(discrete_alphas)
      
      n = inputs.size(0)
      inputs = inputs.to(device)
      targets = targets.to(device)
      
      optimizer.zero_grad()
      _, logits = model(inputs)
      loss = criterion(logits, targets)
      loss.backward()
      nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
      optimizer.step()

      prec1, prec5 = utils.obtain_accuracy(logits, targets, topk = (1, 5))
      losses.update(loss.item(),  n)
      top1.update  (prec1.item(), n)
      top5.update  (prec5.item(), n)
      
      if (step) % args.report_freq == 0:
        logging.info(f"[Epoch #{gen}]: training using the population")
        logging.info(f"Using Training batch #{step} with loss: {losses.avg:.5f}, prec1: {top1.avg:.5f}, prec5: {top5.avg:.5f}")
      #break
  logging.info(f"Training finished with loss: {losses.avg:.5f}, prec1: {top1.avg:.5f}, prec5: {top5.avg:.5f}")

class NAS(ElementwiseProblem):
  def __init__(self, n_var, n_obj, xl, xu):
    super().__init__(n_var=n_var, n_obj=n_obj, n_constr=0, xl=xl, xu=xu)

  def validation(self, ind, model, valid_queue, criterion, gen, ind_idx, pop_size, device):
    tx = torch.tensor(ind.X).type(torch.float).reshape_as(model.arch_parameters).to(device)
    model.eval()
    valid_start = time.time()
    #Copying and checking the discretized alphas
    model.update_alphas(tx)
    arch_str_tmp = model.genotype().tostr()
    discrete_alphas = model.discretize()
    assert model.genotype().tostr() == arch_str_tmp, 'Something wrong with discretization'

    losses, top1, top5 = utils.AvgrageMeter(), utils.AvgrageMeter(), utils.AvgrageMeter()
    with torch.no_grad():
      for step, (inputs, targets) in enumerate(valid_queue):
        n = inputs.size(0)
        inputs = inputs.to(device)
        targets = targets.to(device)
        _, logits = model(inputs)
        loss = criterion(logits, targets)
        prec1, prec5 = utils.obtain_accuracy(logits, targets, topk = (1, 5))
        losses.update(loss.item(),  n)
        top1.update  (prec1.item(), n)
        top5.update  (prec5.item(), n)
        
        #break
    #print("Finished in {} seconds".format((time.time() - valid_start) ))
    arch_losses, arch_top1, arch_top5 = losses.avg, top1.avg, top5.avg
    logging.info(f"[{gen} Generation] {ind_idx}/{pop_size} finished with validation loss: {arch_losses:.5f}, prec1: {arch_top1:.5f}, prec5: {arch_top5:.5f}")
    return arch_losses, arch_top1, arch_top5, arch_str_tmp

if args.seed is None or args.seed < 0: args.seed = random.randint(1, 100000)
DIR = "search-{}-{}".format(time.strftime("%Y%m%d-%H%M%S"), args.dataset)
if args.output_dir is not None:
  if not os.path.exists(args.output_dir):
    utils.create_exp_dir(args.output_dir)
  DIR = os.path.join(args.output_dir, DIR)
else:
  DIR = os.path.join(os.getcwd(), DIR)
utils.create_exp_dir(DIR)
utils.create_exp_dir(os.path.join(DIR, "weights"))
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(DIR, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

# Initializing the summary writer
writer = SummaryWriter(os.path.join(DIR, 'runs'))

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

device = torch.device("cuda:{}".format(args.gpu))
cpu_device = torch.device("cpu")

torch.cuda.set_device(args.gpu)
cudnn.deterministic = True
cudnn.enabled = True
cudnn.benchmark = False

assert args.api_path is not None, 'NAS201 data path has not been provided'
api = API(args.api_path, verbose = False)
logging.info(f'length of api: {len(api)}')

logging.info(f'python {" ".join([ar for ar in sys.argv])}')
logging.info(f'torch version: {torch.__version__}, torchvision version: {torch.__version__}')
logging.info("gpu device = {}".format(args.gpu))
logging.info("args =  %s", args)
logging.info("[INFO] First Train and then evolve and repeat the cycle")
logging.info('[INFO] No novelty threshold used')

# Configuring dataset and dataloader
if args.dataset == 'cifar10':
  acc_type     = 'ori-test'
  val_acc_type = 'x-valid'
else:
  acc_type     = 'x-test'
  val_acc_type = 'x-valid'

datasets = ['cifar10', 'cifar100', 'ImageNet16-120']
assert args.dataset in datasets, 'Incorrect dataset'
if args.cutout:
  train_data, valid_data, xshape, num_classes = get_datasets(name = args.dataset, root = args.data, cutout=args.cutout)
else:
  train_data, valid_data, xshape, num_classes = get_datasets(name = args.dataset, root = args.data, cutout=-1)
logging.info("train data len: {}, valid data len: {}, xshape: {}, #classes: {}".format(len(train_data), len(valid_data), xshape, num_classes))

config = load_config(path=args.config_path, extra={'class_num': num_classes, 'xshape': xshape}, logger=None)
logging.info(f'config: {config}')
print(args.config_path, args.config_root)
_, train_loader, valid_loader = get_nas_search_loaders(train_data=train_data, valid_data=valid_data, dataset=args.dataset,
                                                        config_root=args.config_root, batch_size=(args.batch_size, args.valid_batch_size),
                                                        workers=args.workers)
train_queue, valid_queue = train_loader, valid_loader
logging.info('search_loader: {}, valid_loader: {}'.format(len(train_queue), len(valid_queue)))

# Model Initialization
#model_config = {'C': 16, 'N': 5, 'num_classes': num_classes, 'max_nodes': 4, 'search_space': NAS_BENCH_201, 'affine': False}
model = TinyNetwork_NAS(C = args.init_channels, N = args.num_cells, max_nodes = args.max_nodes,
                        num_classes = num_classes, search_space = NAS_BENCH_201, affine = False,
                        track_running_stats = args.track_running_stats)
model = model.to(device)
#logging.info(model)

optimizer, _, criterion = get_optim_scheduler(parameters=model.get_weights(), config=config)
criterion = criterion.cuda()
logging.info(f'optimizer: {optimizer}\nCriterion: {criterion}')

#arch_index = api.query_index_by_arch(model.genotype())
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.train_epochs+args.epochs), eta_min = args.learning_rate_min)
logging.info(f'Scheduler: {scheduler}')

# Intializing the Novelty Metric
novelty_metric = Novelty(knn=args.knn)

# Initializing the problem
n_params=model.arch_parameters.view(-1).shape[0]
nas = NAS(n_var=n_params,
          n_obj=2,
          xl=np.zeros(n_params),
          xu=np.ones(n_params)  
         )
# create the algorithm object
algorithm = NSGA2(pop_size=args.pop_size,
                  crossover=SimulatedBinaryCrossover(eta=15, prob=0.7),
                  mutation=PolynomialMutation(prob=args.mutate_rate, eta=20)
                  )

# let the algorithm object never terminate and let the loop control it
termination = NoTermination()

# create an algorithm object that never terminates
algorithm.setup(problem=nas, termination=termination, seed=args.seed, save_history=True)

#scheduler.step()
lr = scheduler.get_lr()[0]

df = pd.DataFrame(columns=['generation',  'arch', 'genotype', 'arch_loss', 'arch_top1', 'arch_top5', 'test_acc', 'valid_acc', 'novelty_score', 'knneighbors', 'fitness'])
acc_df = pd.DataFrame(columns=['genotype', 'top1'])
# STAGE 1
start = time.time()
#def train(model, train_queue, criterion, optimizer, gen, device, pop=None):
for train_epoch in range(args.train_epochs):
  logging.info('[INFO] Training the Supernet (Warmup)')
  train_time = time.time()
  logging.info("[INFO] Epoch {} with learning rate {}".format(train_epoch + 1, scheduler.get_lr()[0]))
  train(model=model, train_queue=train_queue, criterion=criterion, optimizer=optimizer, gen=train_epoch+1, device=device)
  logging.info("[INFO] Training finished in {} minutes".format((time.time() - train_time) / 60))
  scheduler.step()
  #torch.save(model.state_dict(), "model.pt")
  utils.model_save(model, os.path.join(DIR, "weights","weights.pt"))

for n_gen in range(args.epochs):
  start_time = time.time()
  # ask the algorithm for the next solution to be evaluated
  pop = algorithm.ask()

  ## Training the whole population
  logging.info("[INFO] Generation {} training with learning rate {}".format(n_gen + 1, scheduler.get_lr()[0]))
  #def train(model, train_queue, criterion, optimizer, gen, device, pop=None):
  train(model=model, train_queue=train_queue, criterion=criterion, optimizer=optimizer, gen=n_gen+1, device=device, pop=pop)
  logging.info("[INFO] Training finished in {} minutes".format((time.time() - start_time) / 60))
  utils.model_save(model, os.path.join(DIR, "weights","weights.pt"))
  #lr = scheduler.get_lr()[0]
  scheduler.step()
  
  # Evaluating the individuals in the population
  logging.info("[INFO] Evaluating Generation {} ".format(n_gen + 1))
  current_pop, arch_acc = [], []
  for ind_idx, ind in enumerate(pop):
    #losses.avg,top1.avg,top5.avg,arch_str=validation(self, ind, model, valid_queue, criterion, gen, ind_idx, pop_size, df, device)
    losses, top1, top5, arch_str = nas.validation(ind=ind,
                                                  model=model,
                                                  valid_queue=valid_queue,
                                                  criterion=criterion,
                                                  gen=n_gen+1,
                                                  ind_idx=ind_idx+1,
                                                  pop_size=len(pop),
                                                  device=device)
    arch_acc.append(-top1/100)
    current_pop.append(arch_str)
    ind.set('losses', losses)
    ind.set('top1', top1)
    ind.set('top5', top5)
    ind.set('genotype', arch_str)
    d_tmp = {'genotype': arch_str, 'top1': top1}
    acc_df = acc_df.append(d_tmp, ignore_index=True)
  assert len(current_pop)==len(pop)
  assert len(arch_acc)==len(pop)

  # get the novelty score
  tmp_novel_arch, novelty_scores, local_fitnesses = [], [], []
  for ind_idx, ind in enumerate(pop):
    novelty_score, knneighbors, local_fitness = novelty_metric.get_local_fitness_with_novelty( arch=ind.get('genotype'),
                                                                                arch_top1=ind.get('top1'),
                                                                                arch_idx=ind_idx,
                                                                                current_pop=current_pop,
                                                                                acc_df=acc_df)
    ind.set('novelty_score', novelty_score)
    ind.set('knneighbors', knneighbors)
    novelty_scores.append(-novelty_score)
    local_fitnesses.append(-local_fitness)
    tmp_novel_arch.append(ind.get('genotype'))
  assert len(novelty_scores)==len(pop)
  assert len(local_fitnesses)==len(pop)
  novelty_metric.update_archive(tmp_novel_arch)
  
  # objectives
  if args.local_fitness_flag:
    pop.set("F", np.column_stack([local_fitnesses, novelty_scores]))
    logging.info(f'[INFO] Using local fitness: {pop.get("F")}')
  else:
    pop.set("F", np.column_stack([arch_acc, novelty_scores]))
    logging.info(f'[INFO] Not using local fitness: {pop.get("F")}')

  # this line is necessary to set the CV and feasbility status - even for unconstrained
  set_cv(pop)
  
  # returned the evaluated individuals which have been evaluated or even modified
  algorithm.tell(infills=pop)
  logging.info(f'Algorithm generation #{algorithm.n_gen}')
  
  for idx, p in enumerate(pop):
    #logging.info(f'length of population: {len(pop)}')
    assert p.get('novelty_score') is not None, 'Novelty score is not assigned'
    assert p.get('knneighbors') is not None, 'k Nearest Neighbors is not assigned'
    if args.dataset == 'cifar10':
      #columns=['generation',  'arch', 'genotype', 'arch_loss', 'arch_top1', 'arch_top5', 'test_acc', 'valid_acc', 'novelty_score', 'knneighbors', 'fitness']
      d_tmp = { 'generation': n_gen+1, 'arch': p.X, 'genotype': p.get('genotype'),
                'arch_loss': p.get('losses'), 'arch_top1': p.get('top1'), 'arch_top5': p.get('top5'),
                'test_acc': get_arch_score(api=api, arch_str=p.get('genotype'), dataset='cifar10', acc_type=acc_type, use_012_epoch_training=False),
                'valid_acc': get_arch_score(api=api, arch_str=p.get('genotype'), dataset='cifar10-valid', acc_type=val_acc_type, use_012_epoch_training=False),
                'novelty_score': p.get('novelty_score'), 'knneighbors': p.get('knneighbors'),
                'fitness': p.F
                }
    else:
      d_tmp = { 'generation': n_gen+1, 'arch': p.X, 'genotype': p.get('genotype'),
                'arch_loss': p.get('losses'), 'arch_top1': p.get('top1'), 'arch_top5': p.get('top5'),
                'test_acc': get_arch_score(api=api, arch_str=p.get('genotype'), dataset=args.dataset, acc_type=acc_type, use_012_epoch_training=False),
                'valid_acc': get_arch_score(api=api, arch_str=p.get('genotype'), dataset=args.dataset, acc_type=val_acc_type, use_012_epoch_training=False),
                'novelty_score': p.get('novelty_score'), 'knneighbors': p.get('knneighbors'),
                'fitness': p.F
                }
    df = df.append(d_tmp, ignore_index=True)
  
  # do same more things, printing, logging, storing or even modifying the algorithm object
    
  last = time.time() - start_time
  logging.info("[INFO] {}/{} generation finished in {} minutes".format(n_gen + 1, args.epochs, last / 60))

# obtain the result objective from the algorithm
res = algorithm.result()

e = res.history[-1]
logging.info(f'[INFO] Best individual after the {e.n_gen} generations')
best_acc, best_valid, best_ind = 0.0, 0.0, None
test_tmp, valid_tmp, pareto_F = [], [], []
for ind in e.opt:
  arch_str = ind.get('genotype')
  pareto_F.append(ind.F)
  if args.dataset == 'cifar10':
    test_acc = get_arch_score(api=api, arch_str=arch_str, dataset='cifar10', acc_type=acc_type, use_012_epoch_training=False)
    valid_acc = get_arch_score(api=api, arch_str=arch_str, dataset='cifar10-valid', acc_type=val_acc_type, use_012_epoch_training=False)
  else:
    test_acc = get_arch_score(api=api, arch_str=arch_str, dataset=args.dataset, acc_type=acc_type, use_012_epoch_training=False)
    valid_acc = get_arch_score(api=api, arch_str=arch_str, dataset=args.dataset, acc_type=val_acc_type, use_012_epoch_training=False)
  if best_acc < test_acc:
    best_acc = test_acc
    best_valid = valid_acc
    best_ind = ind
  test_tmp.append(test_acc)
  valid_tmp.append(valid_acc)
writer.close()

last = time.time() - start
logging.info("[INFO] Architecture search finished in {} hours".format(last / 3600))

pd.DataFrame(novelty_metric.archive).to_csv(os.path.join(DIR, 'archive.json'))
df.to_json(os.path.join(DIR, 'all_population.json'))
acc_df.to_json(os.path.join(DIR, 'acc_df.json'))

logging.info(f'[INFO] Best Architecture after the search: {best_ind.get("genotype")}:: ({best_acc},{best_valid})')
logging.info(f'[INFO] Pareto set: length of pareto set: {len(test_tmp)} \n{test_tmp}\n{valid_tmp}, \n{pareto_F}')
logging.info(f'length Length of the result history: {len(res.history)}, length of df: {len(df)}')
with open(os.path.join(DIR, "result.pickle"), 'wb') as f:
  pickle.dump(res, f)

# Recording the best architecture informations after the search finishes
tmp_a = {'arch': best_ind.get("genotype"), 'dataset': args.dataset,
         'valid': best_valid, 'test': best_acc, 'time': last/3600}
if os.path.exists(os.path.join(args.output_dir, args.record_filename)):
  with open(os.path.join(args.output_dir, args.record_filename), 'a', newline='') as write_obj:
    dict_writer = DictWriter(write_obj, fieldnames=list(tmp_a.keys()) )
    dict_writer.writerow(tmp_a)
else:
  with open(os.path.join(args.output_dir, args.record_filename), 'w', newline='') as write_obj:
    dict_writer = DictWriter(write_obj, fieldnames=list(tmp_a.keys()) )
    dict_writer.writeheader()
    dict_writer.writerow(tmp_a)
