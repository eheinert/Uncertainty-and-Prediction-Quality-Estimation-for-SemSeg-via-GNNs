import hydra
from omegaconf import DictConfig
from pathlib import Path

from metaseg.utils import get_dataset
from metaseg.compute import compute_and_save_all_metaseg_data


@hydra.main(config_path=".", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    dataset_name = cfg.dataset

    """ prepare meta metrics for every sub dataset."""
    for mode in cfg[dataset_name].splits:
        cfg[dataset_name].mode = mode
        dataset = get_dataset(cfg[dataset_name])
        metaseg_data_directory = Path(cfg.save_roots.metaseg_data)
        compute_and_save_all_metaseg_data(dataset, metaseg_data_directory, worker = cfg.worker)

if __name__ == "__main__":
    main()