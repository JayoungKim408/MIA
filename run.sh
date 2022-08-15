#!bin/sh

python train_victim_model.py 
python test_model.py 
python main.py --attack fbb 
python main.py --attack wb
