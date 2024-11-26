import hydra
import os
from os.path import join
import shutil
from omegaconf import OmegaConf

from omegaconf import DictConfig
from pathlib import Path
from sklearn.model_selection import train_test_split
from joblib import dump

from metaseg.utils import get_dataset, get_metaseg_data_per_image
from metaseg.compute import prepare_meta_training_data
from metaseg.train_utils import kfold_validation, train_full, validate
from metaseg.statistics_and_plots import scatterplot, confusion_matr, roc_fromclassifier, roc_fromprobs, feature_importances

@hydra.main(config_path=".", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    dataset_name = cfg.dataset
    print(f"Start fitting {cfg.meta_model} meta model on {dataset_name}!")
    metaseg_data_directory = Path(cfg.save_roots.metaseg_data)
    
    """set up plot directory and dump cfg there"""
    statistics_directory = join(cfg.save_roots.stats_and_plots,cfg.meta_model)
    if os.path.exists(statistics_directory):
        shutil.rmtree(statistics_directory)
    os.makedirs(statistics_directory)
    with open(join(statistics_directory,'config.yaml'), 'w') as f: 
        OmegaConf.save(cfg, f)

    """load metaseg data for the given dataset"""
    dataset = get_dataset(cfg[dataset_name])
    metaseg_data = get_metaseg_data_per_image(dataset, metaseg_data_directory, cfg.worker)
    feature_names = [f for f in list(metaseg_data[0]["metrics"].keys()) if f not in ['class','iou','iou0']]
    
    """prepare data to fit meta model on"""
    x, y, mu, sigma = prepare_meta_training_data(metaseg_data, target_key=cfg.meta_target)
       
    if cfg.val_multiparam == 0: 
        print("Fit on full meta_train set, no validation")
        classifier, train_performances = train_full(x,y,cfg)
        """plots:"""
        if cfg.meta_target == 'iou':
            scatterplot(y, classifier.predict(x), statistics_directory, datatype= 'training')
            feature_importances(classifier, feature_names, statistics_directory)
        elif cfg.meta_target == 'iou0':
            confusion_matr(y, classifier.predict(x), statistics_directory, datatype= 'training')
            roc_fromclassifier(classifier, x, y, statistics_directory, datatype = 'training')
            feature_importances(classifier, feature_names, statistics_directory)
    elif cfg.val_multiparam < 1 and cfg.val_multiparam > 0:
        print(f"Split meta_train data into train ({cfg.val_multiparam * 100} %) and val ({(1-cfg.val_multiparam) * 100} %). Train:")
        train_data, val_data = train_test_split(metaseg_data, test_size = 1-cfg.val_multiparam, random_state = cfg.seed)
        x_train, y_train, mu, sigma = prepare_meta_training_data(train_data, target_key=cfg.meta_target)
        x_val, y_val, mu, sigma = prepare_meta_training_data(val_data, target_key=cfg.meta_target)
        
        classifier, train_performances = train_full(x_train,y_train,cfg)
        print("Validate:")
        val_performances = validate(x_val, y_val, classifier, target_key = cfg.meta_target)

        """plots:"""
        if cfg.meta_target == 'iou':
            scatterplot(y_train, classifier.predict(x_train), statistics_directory, datatype= 'training')
            scatterplot(y_val, classifier.predict(x_val), statistics_directory, datatype= 'validation')
            feature_importances(classifier, feature_names, statistics_directory)
        elif cfg.meta_target == 'iou0':
            confusion_matr(y_train, classifier.predict(x_train), statistics_directory, datatype= 'training')
            confusion_matr(y_val, classifier.predict(x_val), statistics_directory, datatype= 'validation')
            roc_fromclassifier(classifier, x_val, y_val, statistics_directory, datatype = 'validation')
            feature_importances(classifier, feature_names, statistics_directory)
    elif cfg.val_multiparam == 1:
        print(f"Assuming you have an explicit meta validation set. Train on {cfg[dataset_name].splits.meta_train}")
        classifier, train_performances = train_full(x,y,cfg)
        cfg[dataset_name].mode = "meta_val"
        
        print(f"Validate on {cfg[dataset_name].splits.meta_val}")
        val_set = get_dataset(cfg[dataset_name])
        metaval_data = get_metaseg_data_per_image(val_set, metaseg_data_directory, cfg.worker)
        x_val, y_val, mu, sigma = prepare_meta_training_data(metaval_data, target_key = cfg.meta_target)
        val_performances = validate(x_val, y_val, classifier, target_key = cfg.meta_target)

        """plots:"""
        if cfg.meta_target == 'iou':
            scatterplot(y, classifier.predict(x), statistics_directory, datatype= 'training')
            scatterplot(y_val, classifier.predict(x_val), statistics_directory, datatype= 'validation')
            feature_importances(classifier, feature_names, statistics_directory)
        elif cfg.meta_target == 'iou0':
            confusion_matr(y, classifier.predict(x), statistics_directory, datatype= 'training')
            confusion_matr(y_val, classifier.predict(x_val), statistics_directory, datatype= 'validation')
            roc_fromclassifier(classifier, x_val, y_val, statistics_directory, datatype = 'validation')
            feature_importances(classifier, feature_names, statistics_directory)
    if cfg.val_multiparam > 1:
        """KFold validation, then fitting on full set"""
        KF_performances, y_kfresults = kfold_validation(x,y,cfg, return_kfresults=True)

        print("\nFinally fit on all meta_train data:")
        classifier, train_performances = train_full(x,y,cfg)

        """plots:"""
        if cfg.meta_target == 'iou':
            scatterplot(y, classifier.predict(x), statistics_directory, datatype= 'training')
            scatterplot(y, y_kfresults, statistics_directory, datatype= 'kf_cross_validation')
            feature_importances(classifier, feature_names, statistics_directory)
        elif cfg.meta_target == 'iou0':
            y_pred, y_proba = y_kfresults[0], y_kfresults[1]
            confusion_matr(y, classifier.predict(x), statistics_directory, datatype= 'training')
            confusion_matr(y, y_pred, statistics_directory, datatype= 'validation')
            roc_fromprobs(y, y_proba, statistics_directory, datatype = 'kf_cross_validation')
            feature_importances(classifier, feature_names, statistics_directory)

    print("Save meta model in " + join(cfg.save_roots.meta_model, cfg.meta_model+ "/model.joblib"))
    dump(classifier, join(cfg.save_roots.meta_model, cfg.meta_model+ "/model.joblib"))

if __name__ == "__main__":

    main()