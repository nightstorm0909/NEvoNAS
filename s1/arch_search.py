import os
import sys
sys.path.insert(0, './s1')

import argparse
import copy
import logging
import random
import torch.nn as nn
import genotypes
import Novelty as novelty
import numpy as np
import pandas as pd
import pickle
import torch.backends.cudnn as cudnn
import torch.utils
import torch.nn.functional as F
import time
import ut as utils
import visualize

from genotypes                             import PRIMITIVES
from get_datasets                          import get_dataloader
from model_search                          import Network
from torch.utils.tensorboard               import SummaryWriter
from pymoo.core.problem                    import ElementwiseProblem
from pymoo.algorithms.moo.nsga2            import NSGA2
from pymoo.core.evaluator                  import set_cv
from pymoo.util.termination.no_termination import NoTermination
from pymoo.operators.crossover.sbx         import SimulatedBinaryCrossover
from pymoo.operators.mutation.pm           import PolynomialMutation

parser = argparse.ArgumentParser("S1")
parser.add_argument('--autoaug',           default=False, action='store_true')
parser.add_argument('--batch_size',        type = int, default = 64, help = 'batch size')
parser.add_argument('--config_path',       type=str, help='The config path.')
parser.add_argument('--config_root',       type=str, help='The root path of the config directory')
parser.add_argument('--cutout',            action = 'store_true', default = False, help = 'use cutout')
parser.add_argument('--cutout_length',     type = int, default = 16, help = 'cutout length')
parser.add_argument('--data_dir',          type = str, default = '../data', help = 'location of the data corpus')
parser.add_argument('--dataset',           type = str, default = 'cifar10', help = '["cifar10", "cifar100"]')
parser.add_argument('--epochs',            type = int, default = None, help = 'num of generations')
parser.add_argument('--gpu',               type = int, default = 0, help = 'gpu device id')
parser.add_argument('--grad_clip',         type = float, default = 5, help = 'gradient clipping')
parser.add_argument('--init_channels',     type = int, default = 16, help = 'num of init channels')
parser.add_argument('--knn',               type = int, default = 5, help = 'k-nearest neighbors')
parser.add_argument('--layers',            type = int, default = 8, help = 'total number of layers')
parser.add_argument('--learning_rate',     type = float, default = 0.025, help = 'init learning rate')
parser.add_argument('--learning_rate_min', type = float, default = 0.001, help = 'min learning rate')
parser.add_argument('--momentum',          type = float, default = 0.9, help = 'momentum')
parser.add_argument('--mutate_rate',       type = float, default = 0.1, help = 'mutation rate')
parser.add_argument('--output_dir',        type = str, default = None, help = 'location of trials')
parser.add_argument('--pop_size',          type = int, default = 20, help = 'population size')
parser.add_argument('--report_freq',       type = float, default = 50, help = 'report frequency')
parser.add_argument('--seed',              type = int, default = None, help = 'random seed')
parser.add_argument('--split_option',      type = int, default = 1, help = 'split option for CIFAR100')
parser.add_argument('--train_discrete',    default=False, action='store_true')
parser.add_argument('--train_epochs',      type = int, default = 0, help = 'num of training epochs')
parser.add_argument('--valid_batch_size',  type = int, default = 1024, help = 'validation batch size')
parser.add_argument('--weight_decay',      type = float, default = 3e-4, help = 'weight decay')
parser.add_argument('--workers',           type=int, default=2, help='number of data loading workers (default: 2)')
args = parser.parse_args()

def train(model, train_queue, criterion, optimizer, gen, device, pop=None):
  model.train()
  losses, top1, top5 = utils.AvgrageMeter(), utils.AvgrageMeter(), utils.AvgrageMeter()
  if pop is None:
    logging.info(f'In warm-up training')
    for step, (inputs, targets) in enumerate(train_queue):
      # Sample a random architecture
      rnd = model.random_alphas(discrete=False)
      assert model.check_alphas(rnd), "Given alphas has not been copied successfully to the model"
      if args.train_discrete:
        discrete_alphas = utils.discretize(alphas=rnd, arch_genotype=model.genotype())
        model.update_alphas(discrete_alphas)
        assert model.check_alphas(discrete_alphas), "Given alphas has not been copied successfully to the model"
    
      n = inputs.size(0)
      inputs = inputs.to(device)
      targets = targets.to(device)
      optimizer.zero_grad()
      logits = model(inputs)
      loss = criterion(logits, targets)
      loss.backward()
      nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
      optimizer.step()
      
      prec1, prec5 = utils.obtain_accuracy(logits.data, targets.data, topk = (1, 5))
      losses.update(loss.item(),  inputs.size(0))
      top1.update  (prec1.item(), inputs.size(0))
      top5.update  (prec5.item(), inputs.size(0))

      if (step) % args.report_freq == 0:
        logging.info(f"[Epoch #{gen}]: train_discrete: {args.train_discrete}")
        logging.info(f"Using Training batch #{step} with loss: {losses.avg:.5f}, prec1: {top1.avg:.5f}, prec5: {top5.avg:.5f}")
      #break
  else:
    for step, (inputs, targets) in enumerate(train_queue):
      #Copying and checking the discretized alphas
      tx = torch.tensor(pop[step % len(pop)].X).type(torch.float).to(device)
      tx = list(torch.chunk(tx, 2))
      tx = [ttx.reshape(model.arch_parameters()[0].shape) for ttx in tx]    
      model.update_alphas(tx)
      assert model.check_alphas(tx), "Given alphas has not been copied successfully to the model"
      # Discretizing the architecture
      discrete_alphas = utils.discretize(alphas=tx, arch_genotype=model.genotype(), device=device)
      model.update_alphas(discrete_alphas)
      assert model.check_alphas(discrete_alphas)
      #logging.info(f'step % len(pop): {step % len(pop)}')

      n = inputs.size(0)
      inputs = inputs.to(device)
      targets = targets.to(device)
      optimizer.zero_grad()
      logits = model(inputs)
      loss = criterion(logits, targets)
      loss.backward()
      nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
      optimizer.step()
      
      prec1, prec5 = utils.obtain_accuracy(logits.data, targets.data, topk = (1, 5))
      losses.update(loss.item(),  inputs.size(0))
      top1.update  (prec1.item(), inputs.size(0))
      top5.update  (prec5.item(), inputs.size(0))

      if (step) % args.report_freq == 0:
        logging.info(f"[Epoch #{gen}]: train_discrete: {args.train_discrete}")
        logging.info(f"Using Training batch #{step} with loss: {losses.avg:.5f}, prec1: {top1.avg:.5f}, prec5: {top5.avg:.5f}")
      #break
  logging.info(f"Training finished with loss: {losses.avg:.5f}, prec1: {top1.avg:.5f}, prec5: {top5.avg:.5f}")

class NAS(ElementwiseProblem):
  def __init__(self, n_var, n_obj, xl, xu):
    super().__init__(n_var=n_var, n_obj=n_obj, n_constr=0, xl=xl, xu=xu)

  def validation(self, ind, model, valid_queue, criterion, gen, ind_idx, pop_size, device):
    valid_start = time.time()
    
    tx = torch.tensor(ind.X).type(torch.float).to(device)
    tx = list(torch.chunk(tx, 2))
    tx = [ttx.reshape(model.arch_parameters()[0].shape) for ttx in tx]    
    #Copying and checking the discretized alphas
    model.update_alphas(tx)
    assert model.check_alphas(tx), "Given alphas has not been copied successfully to the model"
    g1 = model.genotype()
    # Discretizing the architecture
    discrete_alphas = utils.discretize(alphas=tx, arch_genotype=g1, device=device)
    model.update_alphas(discrete_alphas)
    assert model.check_alphas(discrete_alphas)
    assert utils.compare_genotypes(arch1=model.genotype(), arch2=g1), 'Something wrong with discretization'
    if not ('genotype' in ind.data): ind.set('genotype', g1)
    
    model.eval()
    losses, top1, top5 = utils.AvgrageMeter(), utils.AvgrageMeter(), utils.AvgrageMeter()
    with torch.no_grad():
      for step, (inputs, targets) in enumerate(valid_queue):
        n = inputs.size(0)
        inputs = inputs.to(device)
        targets = targets.to(device)
        logits = model(inputs)
        loss = criterion(logits, targets)
        prec1, prec5 = utils.obtain_accuracy(logits, targets, topk = (1, 5))
        losses.update(loss.item(),  n)
        top1.update  (prec1.item(), n)
        top5.update  (prec5.item(), n)

        #break
    #logging.info("Finished in {} seconds".format((time.time() - valid_start) ))

    logging.info(f"[{gen} Generation] {ind_idx}/{pop_size} finished with validation loss: {losses.avg:.5f}, prec1: {top1.avg:.5f}, prec5: {top5.avg:.5f}")
    return losses.avg, top1.avg, top5.avg, g1

if args.seed is None or args.seed < 0: args.seed = random.randint(1, 100000)
DIR = "search-S1-{}-{}".format(time.strftime("%Y%m%d-%H%M%S"), args.dataset)
if args.output_dir is not None:
  if not os.path.exists(args.output_dir):
    utils.create_exp_dir(args.output_dir)
  DIR = os.path.join(args.output_dir, DIR)
else:
  DIR = os.path.join(os.getcwd(), DIR)
utils.create_exp_dir(DIR)
utils.create_exp_dir(os.path.join(DIR, "weights"))
utils.create_exp_dir(os.path.join(DIR, "output_genotypes"))
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(DIR, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

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

logging.info(f'python {" ".join([ar for ar in sys.argv])}')
logging.info(f'torch version: {torch.__version__}, torchvision version: {torch.__version__}')
logging.info("gpu device = {}".format(args.gpu))
logging.info("args =  %s", args)
logging.info("[INFO] First Train and then evolve and repeat the cycle")

# Configuring dataset and dataloader
train_transform, valid_transform, train_loader, valid_loader = get_dataloader(args)
logging.info(f'train_transform: {train_transform}, \nvalid_transform: {valid_transform}')
if args.dataset == 'cifar10':    num_classes = 10
elif args.dataset == 'cifar100': num_classes = 100
logging.info("#classes: {}".format(num_classes))
train_queue, valid_queue = train_loader, valid_loader
logging.info('search_loader: {}, valid_loader: {}'.format(len(train_queue)*args.batch_size, len(valid_queue)*args.valid_batch_size))

# Model Initialization
model = Network(args.init_channels, num_classes, args.layers, device)
model = model.to(device)

# Configuring the optimizer and the scheduler
optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum = args.momentum, weight_decay = args.weight_decay)
criterion = nn.CrossEntropyLoss()
criterion.to(device)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.train_epochs+args.epochs), eta_min = args.learning_rate_min)
lr = scheduler.get_lr()[0]
logging.info(f'optimizer: {optimizer}\nCriterion: {criterion}')
logging.info(f'Scheduler: {scheduler}')

# Intializing the Novelty Metric
novelty_metric = novelty.Novelty(knn=args.knn)

# Initializing the problem
n_params = model.arch_parameters()[0].view(-1).shape[0] * 2
novelty_metric = novelty.Novelty(knn=3)
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

lr = scheduler.get_lr()[0]
df = pd.DataFrame(columns=['generation',  'arch', 'genotype', 'arch_top1', 'arch_top5', 'novelty_score', 'knneighbors', 'fitness'])
acc_df = pd.DataFrame(columns=['genotype', 'top1'])

# STAGE 1
start = time.time()
if args.train_epochs > 0: logging.info('[INFO] Training the Supernet (Warmup)')
for train_epoch in range(args.train_epochs):
  train_time = time.time()
  logging.info("[INFO] Epoch {} with learning rate {}".format(train_epoch + 1, scheduler.get_lr()[0]))
  #def train(model, train_queue, criterion, optimizer, gen, device, pop=None):
  train(model=model, train_queue=train_queue, criterion=criterion, optimizer=optimizer, gen=train_epoch+1, device=device)
  logging.info("[INFO] Training finished in {} minutes".format((time.time() - train_time) / 60))
  scheduler.step()
  #torch.save(model.state_dict(), "model.pt")
  utils.save(model, os.path.join(DIR, "weights","weights.pt"))

for n_gen in range(args.epochs):
  start_time = time.time()
  # ask the algorithm for the next solution to be evaluated
  pop = algorithm.ask()

  ## Training using the whole population
  logging.info("[INFO] Generation {} training with learning rate {}".format(n_gen + 1, scheduler.get_lr()[0]))
  #def train(model, train_queue, criterion, optimizer, gen, device, pop=None):
  train(model=model, train_queue=train_queue, criterion=criterion, optimizer=optimizer, gen=n_gen+1, device=device, pop=pop)
  logging.info("[INFO] Training finished in {} minutes".format((time.time() - start_time) / 60))
  utils.save(model, os.path.join(DIR, "weights","weights.pt"))
  #lr = scheduler.get_lr()[0]
  scheduler.step()
  
  # Evaluating the individuals in the population
  logging.info("[INFO] Evaluating Generation {} ".format(n_gen + 1))
  current_pop, arch_acc = [], []
  for ind_idx, ind in enumerate(pop):
    #losses.avg, top1.avg, top5.avg, g1 = validation(self, ind, model, valid_queue, criterion, gen, ind_idx, pop_size, device)
    losses, top1, top5, arch_genotype = nas.validation(ind=ind,
                                                  model=model,
                                                  valid_queue=valid_queue,
                                                  criterion=criterion,
                                                  gen=n_gen+1,
                                                  ind_idx=ind_idx+1,
                                                  pop_size=len(pop),
                                                  device=device)
    arch_acc.append(-top1/100)
    current_pop.append(arch_genotype)
    ind.set('losses', losses)
    ind.set('top1', top1)
    ind.set('top5', top5)
    ind.set('genotype', arch_genotype)
    d_tmp = {'genotype': arch_genotype, 'top1': top1}
    acc_df = acc_df.append(d_tmp, ignore_index=True)
  assert len(current_pop)==len(pop)
  assert len(arch_acc)==len(pop)

  # get the novelty score
  tmp_novel_arch, novelty_scores = [], []
  for ind_idx, ind in enumerate(pop):
    novelty_score, knneighbors = novelty_metric.how_novel(arch=ind.get('genotype'),
                                                                         arch_idx=ind_idx,
                                                                         current_pop=current_pop)
    ind.set('novelty_score', novelty_score)
    ind.set('knneighbors', knneighbors)
    novelty_scores.append(-novelty_score)
    tmp_novel_arch.append(ind.get('genotype'))
  assert len(novelty_scores)==len(pop)
  novelty_metric.update_archive(tmp_novel_arch)

  # objectives
  pop.set("F", np.column_stack([arch_acc, novelty_scores]))
  logging.info(f'[INFO] Fitness: {pop.get("F")}')
  
  # this line is necessary to set the CV and feasbility status - even for unconstrained
  set_cv(pop)
  
  # returned the evaluated individuals which have been evaluated or even modified
  algorithm.tell(infills=pop)
  logging.info(f'Algorithm generation #{algorithm.n_gen} completed')

  for idx, p in enumerate(pop):
    assert p.get('novelty_score') is not None, 'Novelty score is not assigned'
    assert p.get('knneighbors') is not None, 'k Nearest Neighbors is not assigned'
    #columns=['generation',  'arch', 'genotype', 'arch_top1', 'arch_top5', 'novelty_score', 'knneighbors', 'fitness']
    d_tmp = { 'generation': n_gen+1, 'arch': p.X, 'genotype': p.get('genotype'),
              'arch_loss': p.get('losses'), 'arch_top1': p.get('top1'), 'arch_top5': p.get('top5'),
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
pareto_F, genotype_list = [], []
for ind_idx, ind in enumerate(e.opt):
  arch_genotype = ind.get('genotype')
  if len(genotype_list) > 0:
    if not utils.search_genotype_list(arch=arch_genotype, arch_list=genotype_list):
      genotype_list.append(arch_genotype)
      with open(os.path.join(DIR, 'output_genotypes', f'genotype{ind_idx+1}.pickle'), 'wb') as f:
        pickle.dump(arch_genotype, f)
    else:
      logging.info(f'[INFO] Skipping the genotype_{ind_idx+1}')
  elif len(genotype_list)==0:
    genotype_list.append(arch_genotype)
    with open(os.path.join(DIR, 'output_genotypes', f'genotype{ind_idx+1}.pickle'), 'wb') as f:
      pickle.dump(arch_genotype, f)

  logging.info(f'[INFO] Genotype_{ind_idx+1}: fitness->{-ind.F[0]:.5f}, Novelty->{-ind.F[1]:.5f}')
  pareto_F.append(ind.F)

last = time.time() - start
logging.info("[INFO] Architecture search finished in {} hours".format(last / 3600))

pd.DataFrame(novelty_metric.archive).to_csv(os.path.join(DIR, 'archive.csv'))
df.to_json(os.path.join(DIR, 'all_population.json'))
acc_df.to_json(os.path.join(DIR, 'acc_df.json'))

#logging.info(f'[INFO] Best Architecture after the search: {best_ind.get("genotype")}:: ({best_acc},{best_valid})')
logging.info(f'[INFO] Pareto set: length of pareto set: {len(pareto_F)} \nFitness:\n{pareto_F}')
logging.info(f'length Length of the result history: {len(res.history)}, length of df: {len(df)}')
logging.info(f'length Length of the genotype list: {len(genotype_list)}')
with open(os.path.join(DIR, "result.pickle"), 'wb') as f:
  pickle.dump(res, f)
