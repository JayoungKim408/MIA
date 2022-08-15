import os 
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_LAUNCH_BLOCKING"]="1"
import warnings
from mia.benchmark import run
from mia.save import *
import numpy as np
import argparse
import time 
import torch 

randomSeed = 2021
torch.manual_seed(randomSeed)
torch.cuda.manual_seed(randomSeed)
torch.cuda.manual_seed_all(randomSeed) 
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(randomSeed)

warnings.filterwarnings(action='ignore')
parser = argparse.ArgumentParser()
curr_time = time.time()

PATH = "./logs/csvs/"

# train & test environment 
parser.add_argument('--curr_time', type=str, default=curr_time)
parser.add_argument('--test_iter', type=int, default=5)
parser.add_argument('--save_loc', type=str, default='./result')
parser.add_argument('--baseline', type=eval, default=True, choices=[True, False])

# model parameters
parser.add_argument('--synthesizer', type=str, default='ctgan')
parser.add_argument('--dataset_name', type=str, default='alphabank')
parser.add_argument('--gpu_num', type=str, default="0")

# dp arguments                    
parser.add_argument('--dp', type=eval, choices=[True, False], default=True)
parser.add_argument('--dp_type', type=str, choices=['sgd', 'gan'], default='sgd')
parser.add_argument('--noise_multiplier', type=float, default=1.0)
parser.add_argument('--max_grad_norm', type=float, default=1.0)

config = parser.parse_args()    

for param, value in vars(config).items():
    print("{}: {}".format(param, value))

os.environ["CUDA_VISIBLE_DEVICES"]=config.gpu_num

# training and testing synthesizer.
if config.synthesizer == "ctgan":
    from mia.synthesizers.ctgan import CTGANSynthesizer as Synthesizer
elif config.synthesizer == "identity":
    from mia.synthesizers.identity import IdentitySynthesizer as Synthesizer
elif config.synthesizer == "tvae":
    from mia.synthesizers.tvae import TVAESynthesizer as Synthesizer
elif config.synthesizer == "tablegan":
    from mia.synthesizers.tablegan import TableganSynthesizer as Synthesizer
elif config.synthesizer == 'octgan':
    from mia.synthesizers.octgan import OCTGANSynthesizer as Synthesizer

path = config.save_loc + f"_{config.synthesizer}"
scores = run(Synthesizer, arguments=config, output_path=path)

end_time = time.strftime('%b%d_%H-%M-%S', time.localtime(time.time()))
print("finish time:", end_time)
