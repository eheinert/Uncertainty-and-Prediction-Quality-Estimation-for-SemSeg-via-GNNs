import hydra
from omegaconf import DictConfig
import numpy as np
from pathlib import Path
from tqdm import tqdm
from p_tqdm import p_map
import pickle
from functools import partial
import torch
from PIL import Image
import torch.nn.functional as F
from torchvision.transforms import Compose, ToTensor, Normalize

def get_metaseg_data_per_image(dataset, load_dir=None, worker=None):
    print("Load MetaSeg files from here: {}".format(Path(load_dir)))
    if worker is not None and worker.parallel:
        num_cpus = worker.num_cpus if "num_cpus" in worker else 1
        metaseg_data = p_map(partial(load_metaseg_data, load_dir=load_dir), dataset.all_basenames, num_cpus=num_cpus)
    else:
        print("single processing, still TODO")
        metaseg_data = [None for _ in dataset]
    return metaseg_data

def load_metaseg_data(basename, load_dir):
    load_path = Path(load_dir) / f"{basename}.p"
    return pickle.load(open(load_path, "rb"))

def concatenate_metrics(metrics):
    num_segments = sum(m.num_segments for m in metrics)
    all_metrics = {k: np.empty(num_segments, dtype=v.dtype) for k, v in metrics[0].metrics.items()}
    curr_idx = 0
    for m in metrics:
        next_idx = curr_idx + m.num_segments
        for k, v in m.metrics.items():
            all_metrics[k][curr_idx:next_idx] = v
        curr_idx = next_idx
    return all_metrics

def metrics_dict_as_predictors(metrics, exclude=None):
    if exclude is None:
        exclude = ["iou", "iou0", "class"]
    else:
        print("Exclude the metrics:", exclude)
    return np.array([v for k, v in metrics.items() if k not in exclude], dtype=np.float32).T.copy()

def get_dataset(dataset_class):
    """returns dataset:_target_ as defined in config"""
    return hydra.utils.instantiate(dataset_class)

def get_meta_model(meta_model_class):
    """returns meta_model:_target_ as defined in config"""
    return hydra.utils.instantiate(meta_model_class)

def init_segmentation_network_initialiser(cfg):
    model_name = cfg.semseg_model
    loader_path = cfg[model_name].loader
    loader_path = loader_path.replace('/','.')

    loader_module,_, loader_name = loader_path.rpartition(".")
    module = __import__(loader_module, fromlist=[loader_name])
    init_network = getattr(module,loader_name)

    return init_network

def get_softmax(image_path, net, transform):
    image = Image.open(image_path).convert('RGB')
    if transform is not None:
        image = transform(image)
    x = image.unsqueeze_(0).cuda()
    with torch.no_grad():
        logits = net(x)
    if type(logits) == tuple:
        logits = logits[0]
    softmax = torch.squeeze(F.softmax(logits, 1)).data.cpu()
    softmax = softmax.numpy().astype("float32")

    return softmax

def prepare_probs(net, cfg:DictConfig):
    """ create 'empty' dataset, softmax probabilities are yet to be computed."""
    dataset_name = cfg.dataset
    dataset = get_dataset(cfg[dataset_name])
    
    transform = Compose([ToTensor(), Normalize(dataset.mean, dataset.std)]) 
    print(f"Compute softmax probabilities for {dataset.mode}" + " data and save here: {}".format(cfg[dataset_name].probs_root))
    for image_path, basename in tqdm(zip(dataset.all_images, dataset.all_basenames), total=len(dataset.all_images)):
        softmax = get_softmax(image_path, net, transform)
        save_path = Path(cfg[dataset_name].probs_root) / f"{basename}.npy"
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(save_path, softmax)