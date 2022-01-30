import sys
sys.path.insert(0,'./')

import os
import glob
import numpy as np
import torch
import ut
import logging
import argparse
import torch.nn as nn
import genotypes
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from model import NetworkCIFAR as Network

import pickle

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--dir', type=str, default=None, help='location of population')
parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=20, help='total number of layers')
parser.add_argument('--model_path', type=str, default='EXP/model.pt', help='path of pretrained model')
parser.add_argument('--log_path', type=str, default=None, help='path of log file')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default=None, help='which architecture to use')
args = parser.parse_args()

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
if args.log_path is not None:
  fh = logging.FileHandler(os.path.join(args.log_path, 'evaluate.txt'))
  fh.setFormatter(logging.Formatter(log_format))
  logging.getLogger().addHandler(fh)

tmp=''
for arg in sys.argv:
  tmp+=' {}'.format(arg)
logging.info(f'python{tmp}')

CIFAR_CLASSES = 10

def main():
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  np.random.seed(args.seed)
  torch.cuda.set_device(args.gpu)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)
  logging.info('gpu device = %d' % args.gpu)
  logging.info("args = %s", args)
  
  print('---------Genotype---------')
  if args.arch is not None:
    genotype = eval("genotypes.%s" % args.arch)
  if args.dir is not None:
    if 'pickle' in args.dir:
      with open(os.path.join(args.dir), 'rb') as f:
        genotype = pickle.load(f)
    else:
      with open(os.path.join(args.dir, "genotype.pickle"), 'rb') as f:
        genotype = pickle.load(f)
    logging.info("Unpickling genotype.pickle")
  logging.info(genotype)
  print('--------------------------') 

  model = Network(args.init_channels, CIFAR_CLASSES, args.layers, args.auxiliary, genotype)
  ut.load(model, args.model_path, args.gpu)
  #model.load_state_dict(torch.load(args.model_path)['state_dict'], strict = False)
  model = model.cuda()

  logging.info("param size = %fMB", ut.count_parameters_in_MB(model))

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()

  _, test_transform = ut._data_transforms_cifar10(args)
  test_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=test_transform)

  test_queue = torch.utils.data.DataLoader(
      test_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2)

  model.drop_path_prob = 0.0
  #model.drop_path_prob = args.drop_path_prob
  test_acc, test_obj = infer(test_queue, model, criterion)
  logging.info('test_acc %f', test_acc)


def infer(test_queue, model, criterion):
  objs = ut.AvgrageMeter()
  top1 = ut.AvgrageMeter()
  top5 = ut.AvgrageMeter()
  model.eval()

  with torch.no_grad():
    for step, (input, target) in enumerate(test_queue):
      input = Variable(input, volatile=True).cuda()
      target = Variable(target, volatile=True).cuda()

      logits, _ = model(input)
      loss = criterion(logits, target)

      prec1, prec5 = ut.accuracy(logits, target, topk=(1, 5))
      n = input.size(0)
      objs.update(loss.data.item(), n)
      top1.update(prec1.data.item(), n)
      top5.update(prec5.data.item(), n)

      if step % args.report_freq == 0:
        logging.info('test %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg


if __name__ == '__main__':
  main() 

