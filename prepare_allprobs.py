import hydra
import torch
from omegaconf import DictConfig

from metaseg.utils import init_segmentation_network_initialiser, prepare_probs


@hydra.main(config_path=".", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    dataset_name = cfg.dataset
    
    """ load your semseg model """
    model_name = cfg.semseg_model

    init_segmentation_network = init_segmentation_network_initialiser(cfg)
    net = init_segmentation_network(cfg[model_name].model, cfg[model_name].checkpoint)

    """ prepare and save softmax probabilities for every sub dataset."""
    for mode in cfg[dataset_name].splits:
        cfg[dataset_name].mode = mode
        prepare_probs(net, cfg)

if __name__ == "__main__":
    main()