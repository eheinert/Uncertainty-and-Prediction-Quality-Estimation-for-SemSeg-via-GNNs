import numpy as np
import pickle

from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm
from p_tqdm import p_map
from functools import partial
from easydict import EasyDict as edict

from .metrics import Metrics
from .utils import metrics_dict_as_predictors, concatenate_metrics

def compute_and_save_all_metaseg_data(dataset, save_dir, worker=None):
    print("Compute MetaSeg files, which will be saved here: {}".format(Path(save_dir)))
    if worker is not None and worker.parallel:
        num_cpus = worker.num_cpus if "num_cpus" in worker else 1
        p_map(partial(compute_metaseg_data, save_dir=save_dir, return_data=False),
              dataset, dataset.all_basenames, num_cpus=num_cpus)
    else:
        for dataset_item, basename in zip(dataset, dataset.all_basenames):
            compute_metaseg_data(dataset_item, basename, save_dir)

def compute_and_return_all_metaseg_data(dataset, worker=None):
    print("Compute MetaSeg files")
    if worker is not None and worker.parallel:
        num_cpus = worker.num_cpus if "num_cpus" in worker else 1
        metaseg_data = p_map(partial(compute_metaseg_data, return_data=True),
                             dataset, dataset.all_basenames, num_cpus=num_cpus)
    else:
        print("single processing, still TODO")
        metaseg_data = [None for _ in dataset]
    return metaseg_data

def compute_metaseg_data(dataset_item, basename, save_dir=None, return_data=True):
    metaseg_data_full = Metrics(*dataset_item)
    metaseg_data_slim = edict({"metrics": {k: v[1:] for k, v in metaseg_data_full.metrics.items()},
                               "segments": metaseg_data_full.segments,
                               "num_segments": metaseg_data_full.num_segments,
                               "boundary_mask": metaseg_data_full.boundary_mask,
                               "semantic": metaseg_data_full.labels,
                               "basename": basename})
    if save_dir is not None:
        dump_path = Path(save_dir) / f"{basename}.p"
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        pickle.dump(metaseg_data_slim, open(dump_path, "wb"))
    if return_data:
        return metaseg_data_slim

def prepare_meta_training_data(metaseg_data, target_key="iou"):
    metrics = concatenate_metrics(metaseg_data)
    x = metrics_dict_as_predictors(metrics)
    y = metrics[target_key]
    x = x[~np.isnan(y)]
    mu = x.mean(0)
    sigma = x.std(0) + 1e-10
    x = (x - mu) / sigma
    y = y[~np.isnan(y)]
    return x, y, mu, sigma

def prepare_meta_inference_data(metaseg_data):
    metrics = concatenate_metrics(metaseg_data)
    x = metrics_dict_as_predictors(metrics)
    mu = x.mean(0)
    sigma = x.std(0) + 1e-10
    x = (x - mu) / sigma
    return x, mu, sigma

def meta_inference(meta_model, metrics, mu=None, sigma=None, worker=None):
    print("Start meta inference")
    if worker is not None and worker.parallel:
        num_cpus = worker.num_cpus if "num_cpus" in worker else 1
        chunksize = worker.chunksize if "chunksize" in worker else 1
        pool_args = [(meta_model, metrics_per_image, mu, sigma) for metrics_per_image in metrics]
        with Pool(num_cpus) as pool:
            y_hat = pool.starmap(meta_prediction, tqdm(pool_args, total=len(pool_args)), chunksize=chunksize)
    return y_hat

def meta_prediction(meta_model, metrics, mu=None, sigma=None, standardize_predictors=True):
    x = metrics_dict_as_predictors(metrics)
    if standardize_predictors:
        try:
            x = (x - mu) / sigma
        except TypeError:
            print("Oops! No valid mu and sigma for standardization were provided. Please, try again...")
    return meta_model.predict(x)