##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019 #
##################################################

import genotypes
import os
import torch
import pandas as pd
import yaml

from .configure_utils    import load_config, dict2config, configure2str
from .basic_args         import obtain_basic_args
from .attention_args     import obtain_attention_args
from .random_baseline    import obtain_RandomSearch_args
from .cls_kd_args        import obtain_cls_kd_args
from .cls_init_args      import obtain_cls_init_args
from .search_single_args import obtain_search_single_args
from .search_args        import obtain_search_args
# for network pruning
from .pruning_args       import obtain_pruning_args

# utility function
from .flop_benchmark     import get_model_infos, count_parameters_in_MB
from .evaluation_utils   import obtain_accuracy
from cell_operations     import ResNetBasicblock
from pymoo.core.problem  import ElementwiseProblem

# Custom functions added
def create_exp_dir(path):
  if not os.path.exists(path):
    os.mkdir(path)
  print('Experiment dir : {}'.format(path))
    
def model_save(model, model_path):
  torch.save(model.state_dict(), model_path)

def model_load(model, model_path, gpu = 0):
  model.load_state_dict(torch.load(model_path, map_location = 'cuda:{}'.format(gpu)), strict=False)

def get_arch_score(df, arch_str, dataset, acc_type):
  '''Gets the accuracy for given dataset and architecture from a customized pandas dataframe created from NAS201
  which contains only accuracy informations of all 15625 architectures in NAS201. This dataframe was created to
  reduce the RAM requirement for accessing the accuracy information from the original NAS201'''
  series = df[df['arch'] == arch_str]
  return series[dataset+'-'+acc_type].values[0]

def discretize(alphas, arch_genotype):
  genotype = genotypes.PRIMITIVES
  normal_cell = arch_genotype.normal
  reduction_cell = arch_genotype.reduce
  
  # Discretizing the normal cell
  index = 0
  offset = 0
  new_normal = torch.zeros_like(alphas[0])
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
  new_reduce = torch.zeros_like(alphas[1])
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

def compare_s1genotype(g1, g2):
  for index, node1 in enumerate(g1):
    tmp_list = g2[int(index/2)*2: (int(index/2) + 1)*2]
    if node1 not in tmp_list:
      return False
  return True

def search_dataframe(df, g):
  if (not df.empty):
    for index, row in df.iterrows():
      if compare_s1genotype(row['genotype'].normal, g.normal):
        if compare_s1genotype(row['genotype'].reduce, g.reduce):
          return row
  return None

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

def get_cell_param_name(name, big_OSM, small_OSM, verbose=False):
  if verbose: print(f'Before: {name}')
  split_name = name.split('.')
  cell_number, edge_name, op_number  = int(split_name[1]), split_name[3], split_name[4]
  if not isinstance(big_OSM.cells[cell_number], ResNetBasicblock):
    op_name = small_OSM.cells[cell_number].op_position[edge_name][op_number]    
    for op_num, op in big_OSM.cells[cell_number].op_position[edge_name].items():
      if op == op_name:
        split_name[4] = op_num
        break
  res_name = '.'.join(split_name)
  if verbose: print(f'After: {res_name}')
  return res_name

def inherit_OSM_wts(big_OSM, small_OSM, verbose=False):
  big_model_dict = dict(big_OSM.named_parameters())
  count = 0
  for idx, (param_name2, param2) in enumerate(small_OSM.named_parameters()):
    if 'arch' not in param_name2:
      if verbose: print(f'{param_name2}: {param2.shape}', end='')
      if 'cell' in param_name2:
        inherit_name = get_cell_param_name(name=param_name2, big_OSM=big_OSM,
                                                 small_OSM=small_OSM, verbose=False)
      else:
        inherit_name = param_name2
      param1 = big_model_dict[inherit_name]
      assert param1.shape == param2.shape, "Mismatched shape"
      param2.data.copy_(param1)
      assert torch.all(param1==param2).item(), "Copy failed"
      if verbose: print(f'->{inherit_name}: {param1.shape}: {param1.shape == param2.shape}', end='\n')
      if param1.shape == param2.shape: count += 1
  assert count == len(dict(small_OSM.named_parameters())) - 1, "Number of parameters copied failed"
  if verbose: print(count)

def shrink_space(arch_param, topk, arch_flag, edge2index, op_dict, op_position, verbose=False):
  arch_values, indices = torch.topk(arch_param, topk)
  new_arch_flag = torch.zeros_like(arch_flag)
  edge2index2 = {value:key for key, value in edge2index.items()}
  if verbose: print(f'Before::\narch flags: {arch_flag}\nindices: {indices}')
  for idx, row in enumerate(indices):
    edge = edge2index2[idx]
    for col in row:
      op_name = op_position[edge][str(col.item())]
      op_index = op_dict[op_name]
      new_arch_flag[idx][op_index] = True
      if verbose: print(f'({op_position[edge][str(col.item())]}, {op_index})', end=':')
    if verbose: print()
  if verbose: print(f'After::\narch flags: {new_arch_flag}')
  return new_arch_flag


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


