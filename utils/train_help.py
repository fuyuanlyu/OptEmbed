import logging
import sys
from dataloader.tfloader import CriteoLoader, Avazuloader, KDD12loader
import torch
import pickle
import os

def get_dataloader(dataset, path):
    dataset = dataset.lower()
    if dataset == 'criteo':
        return CriteoLoader(path)
    elif dataset == 'avazu':
        return Avazuloader(path)
    elif dataset == 'kdd12':
        return KDD12loader(path)

def get_stats(path):
    defaults_path = os.path.join(path + "/defaults.pkl")
    with open(defaults_path, 'rb') as fi:
        defaults = pickle.load(fi)
    return [i+1 for i in list(defaults.values())] 

def get_model(opt):
    name = opt["model"].lower()
    from models.modules_train import FM, DeepFM, DCN, FNN, IPNN
    if name == "fm":
        model = FM(opt)
    elif name == "deepfm":
        model = DeepFM(opt)
    elif name == "dcn":
        model = DCN(opt)
    elif name == "fnn":
        model = FNN(opt)
    elif name == "ipnn":
        model = IPNN(opt)
    else:
        raise ValueError("Invalid model type: {}".format(name))
    return model

def get_evo(opt):
    name = opt["model"].lower()
    from models.modules_evo import FM, DeepFM, DCN, FNN, IPNN
    if name == "fm":
        model = FM(opt)
    elif name == "deepfm":
        model = DeepFM(opt)
    elif name == "dcn":
        model = DCN(opt)
    elif name == "fnn":
        model = FNN(opt)
    elif name == "ipnn":
        model = IPNN(opt)
    else:
        raise ValueError("Invalid model type: {}".format(name))
    return model

def get_optimizer(network, opt):
    arch_params, network_params = [], []
    arch_names, network_names = [], []
    for name, param in network.named_parameters():
        if name == "arch":
            arch_params.append(param)
            arch_names.append(name)
        else:
            network_params.append(param)
            network_names.append(name)
    
    print("arch_names:", arch_names)
    print("_"*30)
    print("network_names:", network_names)
    print("_"*30)
    
    arch_group = {
        "params": arch_params,
        "lr": opt["arch_lr"]
    }
    arch_optimizer = torch.optim.SGD([arch_group])

    network_group = {
        'params': network_params,
        'weight_decay': opt['wd'],
        'lr': opt['lr']
    }
    if opt['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD([network_group])
    elif opt['optimizer'] == 'adam':
        optimizer = torch.optim.Adam([network_group])
    else:
        print("Optimizer not supported.")
        sys.exit(-1)

    return arch_optimizer, optimizer

def get_cuda(enabled, device_id=0):
    if enabled:
        assert torch.cuda.is_available(), 'CUDA is not available'
        torch.cuda.set_device(device_id)

def get_log(name=""):
    FORMATTER = logging.Formatter(fmt="[{asctime}]:{message}", style= '{')
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setFormatter(FORMATTER)
    logger.addHandler(ch)
    return logger
