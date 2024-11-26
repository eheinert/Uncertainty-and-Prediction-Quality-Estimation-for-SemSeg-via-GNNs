import hydra
import torch

""" functions that are used to initialize segmentation networks"""

def init_deeplabwv3_plus(model, ckpt_path):
    print("Checkpoint file:", ckpt_path)
    print("Load PyTorch model", end="", flush=True)
    network = hydra.utils.instantiate(model)
    network = torch.nn.DataParallel(network)
    network.load_state_dict(torch.load(ckpt_path)['state_dict'], strict=False)
    network = network.cuda().eval()
    print("... ok")
    return network

def init_general_semsegnetwork(model, ckpt_path, num_classes):
    print("Checkpoint file:", ckpt_path)
    print("Load PyTorch model", end="", flush=True)
    network = hydra.utils.instantiate(model, num_classes=num_classes)
    ckpt = torch.load(ckpt_path)
    if 'state_dict' in ckpt.keys():
        state_dict = torch.load(ckpt_path)['state_dict']
    else:
        state_dict = torch.load(ckpt_path)
    if any('module' in key for key in state_dict.keys()):
        network = nn.DataParallel(network)
    network.load_state_dict(state_dict, strict=False)
    network = network.cuda().eval()
    print("... ok")
    return network