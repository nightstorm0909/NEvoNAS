import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *
from torch.autograd import Variable
from genotypes import PRIMITIVES
from genotypes import Genotype


class MixedOp(nn.Module):

  def __init__(self, C, stride):
    super(MixedOp, self).__init__()
    self._ops = nn.ModuleList()
    for primitive in PRIMITIVES:
      op = OPS[primitive](C, stride, False)
      if 'pool' in primitive:
        op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
      self._ops.append(op)

  def forward(self, x, weights):
    return sum(w * op(x) for w, op in zip(weights, self._ops))


class Cell(nn.Module):

  def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
    super(Cell, self).__init__()
    self.reduction = reduction

    if reduction_prev:
      self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
    else:
      self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
    self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
    self._steps = steps
    self._multiplier = multiplier

    self._ops = nn.ModuleList()
    self._bns = nn.ModuleList()
    for i in range(self._steps):
      for j in range(2+i):
        stride = 2 if reduction and j < 2 else 1
        op = MixedOp(C, stride)
        self._ops.append(op)

  def forward(self, s0, s1, weights):
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)

    states = [s0, s1]
    offset = 0
    for i in range(self._steps):
      s = sum(self._ops[offset+j](h, weights[offset+j]) for j, h in enumerate(states))
      offset += len(states)
      states.append(s)

    return torch.cat(states[-self._multiplier:], dim=1)


class Network(nn.Module):

  def __init__(self, C, num_classes, layers, device, steps=4, multiplier=4, stem_multiplier=3):
    super(Network, self).__init__()
    self._C = C
    self._num_classes = num_classes
    self._layers = layers
    #self._criterion = criterion
    self._steps = steps
    self._multiplier = multiplier
    self._device = device

    self._k = sum(1 for i in range(self._steps) for n in range(2+i))
    self._num_ops = len(PRIMITIVES)
    
    C_curr = stem_multiplier*C
    self.stem = nn.Sequential(
      nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
      nn.BatchNorm2d(C_curr)
    )
 
    C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
    self.cells = nn.ModuleList()
    reduction_prev = False
    for i in range(layers):
      if i in [layers//3, 2*layers//3]:
        C_curr *= 2
        reduction = True
      else:
        reduction = False
      cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
      reduction_prev = reduction
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, multiplier*C_curr

    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(C_prev, num_classes)

    self._initialize_alphas()
    self.prepare_dicts()

  #def set_criterion(self, criterion):
  #  self._criterion = criterion

  def copy_arch_parameters(self, parameters):
    for x, y in zip(self.arch_parameters(), parameters):
        #print("model arch shape: {}, pop arch shape: {}".format(x.data.shape, y.data.shape))
        x.data.copy_(y.data)

  '''def new(self):
    model_new = Network(self._C, self._num_classes, self._layers, self._criterion).to(self._device)
    for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
        x.data.copy_(y.data)
    return model_new'''

  def forward(self, input):
    s0 = s1 = self.stem(input)
    for i, cell in enumerate(self.cells):
      if cell.reduction:
        weights = F.softmax(self.alphas_reduce, dim=-1)
      else:
        weights = F.softmax(self.alphas_normal, dim=-1)
      s0, s1 = s1, cell(s0, s1, weights)
    out = self.global_pooling(s1)
    logits = self.classifier(out.view(out.size(0),-1))
    return logits

  def _loss(self, input, target):
    logits = self(input)
    return self._criterion(logits, target) 

  def _initialize_alphas(self):
    k = sum(1 for i in range(self._steps) for n in range(2+i))
    num_ops = len(PRIMITIVES)

    self.alphas_normal = Variable(1e-3*torch.rand(k, num_ops).to(self._device), requires_grad=False)
    self.alphas_reduce = Variable(1e-3*torch.rand(k, num_ops).to(self._device), requires_grad=False)
    self._arch_parameters = [
      self.alphas_normal,
      self.alphas_reduce,
    ]

  def arch_parameters(self):
    return self._arch_parameters

  def get_alphas(self):
    '''Returns the list of architecture weights copied to cpu'''
    alphas = [a.cpu() for a in self._arch_parameters]
    return alphas
  
  def update_alphas(self, alphas):
    '''Update the architecture weights
       alphas = [normal_wts, reduction_wts]
    '''
    #self.get_alphas()[0].data.copy_(alphas)
    for alpha, value in zip(self.arch_parameters(), alphas):
      alpha.data.copy_(value)
  
  def check_alphas(self, alphas):
    '''Verify if the architecture parameter is equal to the given alphas'''
    #return np.all(self.get_alphas()[0] == alphas.get())
    for x, y in zip(self.arch_parameters(), alphas):
      #print(x, y, x.eq(y))
      if not torch.all(x.eq(y)):
        return False
    return True
  
  def random_alphas(self, discrete=False, k = 1.0):
    '''Generate random values for the architecture parameters and update the architecture parametr of the model
       discrete: Whether to  discretize the alphas during training
       k = value for the operations in the discrete architecture parameter
    '''
    alphas_normal = torch.rand(self._k, self._num_ops, device=self._device)
    alphas_reduce = torch.rand(self._k, self._num_ops, device=self._device)
    tmp = [alphas_normal, alphas_reduce]

    if discrete:
      discrete_tmp = []
      for alpha in tmp:
        softmax_alpha  = nn.functional.softmax(alpha, dim=-1)
        index = softmax_alpha.max(-1, keepdim=True)[1]
        #alphas = torch.zeros_like(softmax_alphas).scatter_(-1, index, 1.0)
        discrete_tmp.append(torch.zeros_like(softmax_alpha).scatter_(-1, index, k))
      self.update_alphas(discrete_tmp)
      return discrete_tmp
    else:
      self.update_alphas(tmp)
      assert self.check_alphas(tmp), "random architecture parameter failed to be copied"
      return tmp

  def genotype(self):

    def _parse(weights):
      gene = []
      n = 2
      start = 0
      for i in range(self._steps):
        end = start + n
        W = weights[start:end].copy()
        edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[:2]
        for j in edges:
          k_best = None
          for k in range(len(W[j])):
            if k != PRIMITIVES.index('none'):
              if k_best is None or W[j][k] > W[j][k_best]:
                k_best = k
          gene.append((PRIMITIVES[k_best], j))
        start = end
        n += 1
      return gene

    gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())
    gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy())

    concat = range(2+self._steps-self._multiplier, self._steps+2)
    genotype = Genotype(
      normal=gene_normal, normal_concat=concat,
      reduce=gene_reduce, reduce_concat=concat
    )
    return genotype

  def prepare_dicts(self):
    self._nodes = ['c_{k-2}', 'c_{k-1}', '0', '1', '2', '3']
    self._nodes_dict = {key:i for key, i in  enumerate(self._nodes)}
    tmp = []
    self._rows = []
    for node in range(2, len(self._nodes)):
      for prev_node in range(node):
        self._rows.append('{}->{}'.format(self._nodes_dict[prev_node], node-2))
    self._rows_dict = {key:i for key, i in  enumerate(self._rows)}

  def show_alphas_dataframe(self):
    df_alphas = []
    for alpha in self.get_alphas():
      df = pd.DataFrame(columns=PRIMITIVES, index=self._rows)
      for index, row in enumerate(alpha):
        df.loc[self._rows_dict[index]] = row.numpy()
      df_alphas.append(df)
    return df_alphas

