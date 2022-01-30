import os
import numpy as np
import pandas as pd
import torch
import shutil
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import pickle
import genotypes
import torch.nn.functional as F
import pandas as pd
import yaml

class AverageMeter(object):     
  """Computes and stores the average and current value"""    
  def __init__(self):   
    self.reset()
  
  def reset(self):
    self.val   = 0.0
    self.avg   = 0.0
    self.sum   = 0.0
    self.count = 0.0
  
  def update(self, val, n=1): 
    self.val = val    
    self.sum += val * n     
    self.count += n
    self.avg = self.sum / self.count    

  def __repr__(self):
    return ('{name}(val={val}, avg={avg}, count={count})'.format(name=self.__class__.__name__, **self.__dict__))

class AvgrageMeter(object):

  def __init__(self):
    self.reset()

  def reset(self):
    self.avg = 0
    self.sum = 0
    self.cnt = 0

  def update(self, val, n=1):
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
  maxk = max(topk)
  batch_size = target.size(0)

  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))

  res = []
  for k in topk:
    #correct_k = correct[:k].view(-1).float().sum(0)
    correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
    res.append(correct_k.mul_(100.0/batch_size))
  return res

def obtain_accuracy(output, target, topk=(1,)):
  """Computes the precision@k for the specified values of k"""
  maxk = max(topk)
  batch_size = target.size(0)

  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))

  res = []
  for k in topk:
    correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
    res.append(correct_k.mul_(100.0 / batch_size))
  return res

class Cutout(object):
  def __init__(self, length):
    self.length = length

  def __call__(self, img):
    h, w = img.size(1), img.size(2)
    mask = np.ones((h, w), np.float32)
    y = np.random.randint(h)
    x = np.random.randint(w)

    y1 = np.clip(y - self.length // 2, 0, h)
    y2 = np.clip(y + self.length // 2, 0, h)
    x1 = np.clip(x - self.length // 2, 0, w)
    x2 = np.clip(x + self.length // 2, 0, w)

    mask[y1: y2, x1: x2] = 0.
    mask = torch.from_numpy(mask)
    mask = mask.expand_as(img)
    img *= mask
    return img

def data_transforms_cifar10(cutout, cutout_length, autoaug):
  CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
  CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

  train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip()
    ])
  
  if autoaug:
    train_transform.transforms.append(transforms.AutoAugment(policy=torchvision.transforms.autoaugment.AutoAugmentPolicy.CIFAR10))
  
  train_transform.transforms.append( transforms.ToTensor(),)
  train_transform.transforms.append( transforms.Normalize(CIFAR_MEAN, CIFAR_STD), )

  if cutout:
    train_transform.transforms.append(Cutout(cutout_length))

  valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
  return train_transform, valid_transform

def data_transforms_cifar100(cutout, cutout_length, autoaug):
  CIFAR_MEAN = [0.5071, 0.4867, 0.4408]
  CIFAR_STD = [0.2675, 0.2565, 0.2761]

  train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
  ])
  if autoaug:
    train_transform.transforms.append(transforms.AutoAugment(policy=torchvision.transforms.autoaugment.AutoAugmentPolicy.CIFAR10))
  
  train_transform.transforms.append( transforms.ToTensor(),)
  train_transform.transforms.append( transforms.Normalize(CIFAR_MEAN, CIFAR_STD), )

  if cutout:
    train_transform.transforms.append(Cutout(cutout_length))
  
  valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
  return train_transform, valid_transform

def _data_transforms_cifar10(args):
  CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
  CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

  train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
  ])
  if args.cutout:
    train_transform.transforms.append(Cutout(args.cutout_length))

  valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
  return train_transform, valid_transform

def _data_transforms_cifar100(args):
  CIFAR_MEAN = [0.5071, 0.4867, 0.4408]
  CIFAR_STD = [0.2675, 0.2565, 0.2761]

  train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
  ])
  if args.cutout:
    train_transform.transforms.append(Cutout(args.cutout_length))

  valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
  return train_transform, valid_transform
def count_parameters_in_MB(model):
  return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6


def save_checkpoint(state, is_best, save):
  filename = os.path.join(save, 'checkpoint.pth.tar')
  torch.save(state, filename)
  if is_best:
    best_filename = os.path.join(save, 'model_best.pth.tar')
    shutil.copyfile(filename, best_filename)

def save(model, model_path):
  torch.save(model.state_dict(), model_path)

def load(model, model_path, gpu = 0):
  model.load_state_dict(torch.load(model_path, map_location = 'cuda:{}'.format(gpu)), strict=False)

def drop_path(x, drop_prob):
  if drop_prob > 0.:
    keep_prob = 1.-drop_prob
    mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
    x.div_(keep_prob)
    x.mul_(mask)
  return x

def create_exp_dir(path, scripts_to_save=None):
  if not os.path.exists(path):
    os.mkdir(path)
  print('Experiment dir : {}'.format(path))

  if scripts_to_save is not None:
    os.mkdir(os.path.join(path, 'scripts'))
    for script in scripts_to_save:
      dst_file = os.path.join(path, 'scripts', os.path.basename(script))
      shutil.copyfile(script, dst_file)

def check_arch_flag(arch_flag, OSM, verbose=False):
  if verbose: print(f'number of cells: {len(OSM.cells)}')
  for idx in range(len(OSM.cells)):
    if idx in [len(OSM.cells)//3, 2*len(OSM.cells)//3]:
      # Reduction cell
      check_flag = arch_flag[1]
    else:
      # Normal cell
      check_flag = arch_flag[0]
    if not torch.all(check_flag==OSM.cells[idx].arch_flag).item(): return False
  return True

def get_cell_param_name(smallOSM_name, big_OSM, small_OSM, verbose=False):
  if verbose: print(f'Before: {smallOSM_name}')
  split_name = smallOSM_name.split('.')
  smallOSM_cell_number, smallOSM_edge_number, smallOSM_op_number  = int(split_name[1]), split_name[3], split_name[5]
  smallOSM_edge_name = small_OSM._rows_dict[int(smallOSM_edge_number)]
  smallOSM_op_name = small_OSM.cells[int(smallOSM_cell_number)].op_position[smallOSM_edge_name][smallOSM_op_number]
  for op_num, op in big_OSM.cells[smallOSM_cell_number].op_position[smallOSM_edge_name].items():
    if op == smallOSM_op_name:
      split_name[5] = op_num
      break
  res_name = '.'.join(split_name)
  if verbose: print(f'After: {res_name}')
  return res_name

def inherit_OSM_wts(big_OSM, small_OSM, verbose=False):
  big_model_dict = dict(big_OSM.named_parameters())
  count = 0
  for idx, (param_name2, param2) in enumerate(small_OSM.named_parameters()):        
    if verbose: print(f'{param_name2}: {param2.shape}', end='')
    if ('cell' in param_name2) and not('preprocess' in param_name2):
      inherit_name = get_cell_param_name(smallOSM_name=param_name2, big_OSM=big_OSM,
                                           small_OSM=small_OSM, verbose=False)
    else:
      inherit_name = param_name2
    param1 = big_model_dict[inherit_name]
    assert param1.shape == param2.shape, "Mismatched shape"
    param2.data.copy_(param1)
    assert torch.all(param1==param2).item(), "Copy failed"
    if verbose: print(f'->{inherit_name}: {param1.shape}: {param1.shape == param2.shape}', end='\n')
    if param1.shape == param2.shape: count += 1
  assert count == len(dict(small_OSM.named_parameters())), "Number of parameters copied failed"
  if verbose: print(f'count: {count}')

class NAS_config:
  '''
  Class for loading the configurations of the NAS algorithms
  '''
  def __init__(self, cfg_file, sectionName):
    self.sectionName = sectionName
    infile = open(cfg_file,'r')
    ymlcfg = yaml.safe_load(infile)
    infile.close()
    self.NAS_cfg = ymlcfg.get(self.sectionName,None)

def compare_s1genotype(g1, g2):
  for index, node1 in enumerate(g1):
    tmp_list = g2[int(index/2)*2: (int(index/2) + 1)*2]
    if node1 not in tmp_list:
      return False
  return True

def compare_genotypes(arch1, arch2):
  normal1, reduce1 = arch1.normal, arch1.reduce
  normal2, reduce2 = arch2.normal, arch2.reduce
  index = 0
  while index < len(normal1):
    if not (normal1[index] in normal2[index:index+2]) or not (normal1[index+1] in normal2[index:index+2]): return False
    if not (reduce1[index] in reduce2[index:index+2]) or not (reduce1[index+1] in reduce2[index:index+2]): return False
    index += 2
  return True

def search_genotype_list(arch, arch_list):
  # Return True if present arch present in arch_list otherwise False
  for each_arch in arch_list:
    if compare_genotypes(arch, each_arch): return True
  return False

def search_dataframe(df, g):
  if (not df.empty):
    for index, row in df.iterrows():
      if compare_s1genotype(row['genotype'].normal, g.normal):
        if compare_s1genotype(row['genotype'].reduce, g.reduce):
          return row
  return None

def converged(genotype_list):
  g1 = genotype_list[0]
  for g2 in genotype_list:
    if not compare_s1genotype(g1, g2): return False
  return True

def discretize(alphas, arch_genotype, device):
  genotype = genotypes.PRIMITIVES
  normal_cell = arch_genotype.normal
  reduction_cell = arch_genotype.reduce
  
  # Discretizing the normal cell
  index = 0
  offset = 0
  new_normal = torch.zeros_like(alphas[0]).to(device)
  while index < len(normal_cell):
    op, cell = normal_cell[index]
    idx = genotypes.PRIMITIVES.index(op)
    new_normal[int(offset + cell)][idx] = 1
    index += 1
    op, cell = normal_cell[index]
    idx = genotypes.PRIMITIVES.index(op)
    new_normal[int(offset + cell)][idx] = 1
    offset += (index // 2) + 2
    index += 1
  
  # Discretizing the reduction cell
  index = 0
  offset = 0
  new_reduce = torch.zeros_like(alphas[1]).to(device)
  while index < len(reduction_cell):
    op, cell = reduction_cell[index]
    idx = genotypes.PRIMITIVES.index(op)
    new_reduce[int(offset + cell)][idx] = 1
    index += 1
    op, cell = reduction_cell[index]
    idx = genotypes.PRIMITIVES.index(op)
    new_reduce[int(offset + cell)][idx] = 1
    offset += (index // 2) + 2
    index += 1
  return [new_normal, new_reduce]
