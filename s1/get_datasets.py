import json
import torch
import torchvision
import torchvision.datasets as dset
import ut

from config_utils import load_config, dict2config

#def get_dataloader(dataset, batch_size, valid_batch_size, cutout, cutout_length, autoaug=False):
def get_dataloader(args):
  if args.dataset == 'cifar10':
    train_transform, valid_transform = ut._data_transforms_cifar10(args)
    #train_transform, valid_transform = ut.data_transforms_cifar10(args.cutout, args.cutout_length, args.autoaug)
    train_data = dset.CIFAR10(root = args.data_dir, train = True, download = True, transform = train_transform)
    cifar_split = load_config(f'{args.config_root}/cifar-split.txt', None, None)
    assert len(train_data) == 50000, "Something wrong with the dataset loading"
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size,
                                               pin_memory = False, num_workers = args.workers,
                                               sampler = torch.utils.data.sampler.SubsetRandomSampler(cifar_split.train)
                                              )
    valid_loader = torch.utils.data.DataLoader(train_data, batch_size=args.valid_batch_size,
                                               pin_memory = False, num_workers = args.workers,
                                               sampler = torch.utils.data.sampler.SubsetRandomSampler(cifar_split.valid)
                                              )
  elif args.dataset == 'cifar100':
    train_transform, valid_transform = ut.data_transforms_cifar100(args.cutout, args.cutout_length, args.autoaug)
    train_data = dset.CIFAR100(root=args.data_dir, train=True , transform=train_transform, download=True)
    assert len(train_data) == 50000, "Something wrong with the dataset loading"
    if args.split_option == 1:
      with open(f'{args.config_root}/cifar100_80-split.txt', 'r') as f:
        cifar100_split = json.load(f)
      cifar100_split = dict2config(cifar100_split, logger=None)
      assert len(cifar100_split.train)==40000, 'Something wrong with the training data split'
      assert len(cifar100_split.valid)==10000, 'Something wrong with the validation data split'
    else:
      with open(f'{args.config_root}/cifar100-split.txt', 'r') as f:
        cifar100_split = json.load(f)
      cifar100_split = dict2config(cifar100_split, logger=None)
      assert len(cifar100_split.train)==25000, 'Something wrong with the training data split'
      assert len(cifar100_split.valid)==25000, 'Something wrong with the validation data split'
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size,
                                               pin_memory = False, num_workers = args.workers,
                                               sampler = torch.utils.data.sampler.SubsetRandomSampler(cifar100_split.train)
                                              )
    valid_loader = torch.utils.data.DataLoader(train_data, batch_size=args.valid_batch_size,
                                               pin_memory = False, num_workers = args.workers,
                                               sampler = torch.utils.data.sampler.SubsetRandomSampler(cifar100_split.valid)
                                              )

  return train_transform, valid_transform, train_loader, valid_loader
