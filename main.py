import numpy as np
import torch
import argparse
from utils import str2bool
from solver import Solver


parser = argparse.ArgumentParser(description='TOY VIB')
parser.add_argument('--epoch', default = 200, type=int, help='epoch size')
parser.add_argument('--lr', default = 1e-4, type=float, help='learning rate')
parser.add_argument('--beta', default = 1e-3, type=float, help='beta')
parser.add_argument('--K', default = 256, type=int, 
        help='dimension of encoding Z')
parser.add_argument('--num_avg', default = 12, type=int, 
        help='the number of samples when perform multi-shot prediction')
parser.add_argument('--batch_size', default = 100, type=int, help='batch size')
parser.add_argument('--dataset', default='MNIST', type=str, 
        help='dataset name')
parser.add_argument('--dset_dir', default='datasets', type=str, 
        help='dataset directory path')
parser.add_argument('--summary_dir', default='summary', type=str, 
        help='summary directory path')
parser.add_argument('--ckpt_dir', default='checkpoints', type=str, 
        help='checkpoint directory path')
parser.add_argument('--load_ckpt',default='', type=str, help='checkpoint name')
parser.add_argument('--mode',default='train', type=str, help='train or test')
parser.add_argument('--device', type=str, default='cpu')
args = parser.parse_args()


#torch.backends.cudnn.enabled = True
#torch.backends.cudnn.benchmark = True


SEED = 950418
torch.manual_seed(SEED)
#torch.cuda.manual_seed(seed)
np.random.seed(SEED)

np.set_printoptions(precision=4)
torch.set_printoptions(precision=4)

print(args)
net = Solver(args)

if args.mode == 'train': 
    net.train()
elif args.mode == 'test': 
    net.test(save_ckpt=False)



