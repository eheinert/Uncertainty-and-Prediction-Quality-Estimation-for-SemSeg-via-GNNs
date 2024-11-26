import hydra
from os.path import join
from omegaconf import DictConfig
from pathlib import Path
from joblib import load

from metaseg.utils import get_dataset, get_metaseg_data_per_image
from metaseg.compute import prepare_meta_inference_data, meta_inference
from metaseg.visualize import visualize_and_save_meta_prediction


@hydra.main(config_path=".", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    print(f"Start meta inference of {cfg.meta_model} on {cfg.dataset}")
    
    """load meta_inference data loader and model"""
    dataset_name = cfg.dataset
    cfg[dataset_name].mode = "meta_infer"
    
    classifier = load(join(cfg.save_roots.meta_model, cfg.meta_model+ "/model.joblib"))
    metaseg_data_directory = Path(cfg.save_roots.metaseg_data)

    """load metaseg data for the given dataset"""
    dataset = get_dataset(cfg[dataset_name])
    metaseg_data = get_metaseg_data_per_image(dataset, metaseg_data_directory, cfg.worker)
    x, mu, sigma = prepare_meta_inference_data(metaseg_data)
    metrics_per_image = [data_per_image.metrics for data_per_image in metaseg_data]
    
    """meta inference  and visualization"""
    y_hat = meta_inference(classifier, metrics_per_image, mu, sigma, cfg.worker)
    visualize_and_save_meta_prediction(metaseg_data, y_hat, Path(cfg.save_roots.meta_inference), inf_blend = cfg.infer_blend \
                                                                ,image_paths = dataset.all_images, worker = cfg.worker)


if __name__ == "__main__":
    main()
