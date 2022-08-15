import os
import torch

def make_dir(args):
    path = args.save_loc+'/'+args.dataset_name
    os.makedirs(path)
    for i in ['/csv','/logs','/model']:
        os.makedirs(path+i)

def save_model(generator, args=None ):
    if not os.path.exists(args.save_loc+'/'+args.dataset_name):
        make_dir(args)

    PATH = args.save_loc+"/{}/model/".format(args.dataset_name)
    
    model_dict = dict()

    model_dict['G'] = generator.state_dict()
    PATH += str(args.synthesizer)

    if args.dp:
        PATH += "_{}".format(args.dp_type)
    PATH += '.pth'
        
    torch.save(model_dict, PATH)
