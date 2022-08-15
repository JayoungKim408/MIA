import pandas
from mia.evaluate import compute_scores
import torch
import argparse
import numpy as np
from mia.attacks.load_model import load_model
from mia.data import load_dataset
import warnings
warnings.filterwarnings(action='ignore')

randomSeed = 2021
torch.manual_seed(randomSeed)
torch.cuda.manual_seed(randomSeed)
torch.cuda.manual_seed_all(randomSeed) 
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(randomSeed)

parser = argparse.ArgumentParser('Test model')
parser.add_argument('--data', type=str, default='alphabank')
parser.add_argument('--synthesizer', type=str, default='ctgan')
parser.add_argument('--k', type=int, default=50000)
parser.add_argument('--test_iter', type=int, default=5)
parser.add_argument('--baseline', type=eval, default=True)

# dp arguments                    
parser.add_argument('--dp', type=eval, choices=[True, False], default=True)
parser.add_argument('--dp_type', type=str, choices=['sgd', 'gan'], default='sgd')

config = parser.parse_args()

train_data, test_data, meta_data, categorical_columns, ordinal_columns = load_dataset(config.data, benchmark=True)
synthesizer = load_model(config.synthesizer, config.data, train_data, meta_data, config.k, categorical_columns, ordinal_columns, config.dp, config.dp_type)

syn_data = synthesizer.sample(config.k)

scores = compute_scores(train_data, test_data, syn_data, meta_data, config)
scores = pandas.DataFrame(scores)
print('-----------------------------------------')
print('Test scores for {} on {}'.format(config.synthesizer, config.data))
print(scores.mean())
print('-----------------------------------------')