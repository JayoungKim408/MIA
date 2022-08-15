import os
import argparse
import warnings 
warnings.filterwarnings("ignore")

from mia.attacks.load_model import load_model
from mia.data import load_dataset

from mia.attacks.fbb import FullBlackBoxAttacker
from mia.attacks.wb import WhiteBoxAttacker


parser = argparse.ArgumentParser('Run Experiments on Attacks')
parser.add_argument('--gpu_num', type=str, default="1")

parser.add_argument('--attack', type=str, default='fbb')
parser.add_argument('--data', type=str, default='alphabank')
parser.add_argument('--target', type=str, default='ctgan')
parser.add_argument('--query_size', type=int, default=2000)

## WB arguments ##
parser.add_argument('--initialize_type', type=str, default ='random')
parser.add_argument('--lambda3', type=float, default=0.001) 
parser.add_argument('--batch_size', type=int, default=2) 
parser.add_argument('--maxfunc', type=int, default=1)
parser.add_argument('--lbfgs_lr', type=float, default=0.0001)

## FBB arguments ## 
parser.add_argument('--K', type=int, default=1000)

## DP arguments ##
parser.add_argument('--dp', type=bool, default=True)
parser.add_argument('--dp_type', type=str, choices=['sgd', 'gan'], default='sgd')


config = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"]=str(config.gpu_num)

z_dim = 128

train_data, test_data, meta_data, categorical_columns, ordinal_columns = load_dataset(config.data, benchmark=True)

victim_model = load_model(config.target, config.data, train_data, meta_data, config.query_size, categorical_columns, ordinal_columns, config.dp, config.dp_type)

if config.attack == 'wb':
    attacker = WhiteBoxAttacker(victim_model, config.data, config.query_size) 
    wb_auc, wb_ap = attacker.attack(train_data, test_data, meta_data)

    print('---- White-box attack result on {} for {} ----'.format(config.target, config.data))
    print('- Args:: lambda3: {}; query_size: {}; batch_size: {}; initialize_type: {}; maxfunc: {}; lbfgs_lr: {}'.format(config.lambda3, config.query_size, config.batch_size, config.initialize_type, config.maxfunc, config.lbfgs_lr))
    print('- WB AUC: ', wb_auc)
    print('- WB AP: ', wb_ap)
    print('------------------------------------------')

elif config.attack == 'fbb':
    attacker = FullBlackBoxAttacker(victim_model, config.data)
    fbb_auc, fbb_ap = attacker.attack(train_data, test_data, meta_data, config.K)

    print('---- Full-black-box attack result on {} for {} ----'.format(config.target, config.data))
    print('- Args:: K: {}; query_size: {}; batch_size: {}'.format(config.K, config.query_size, config.batch_size))
    print('- FBB AUC: ', fbb_auc)
    print('- FBB AP: ', fbb_ap)
    print('------------------------------------------')
