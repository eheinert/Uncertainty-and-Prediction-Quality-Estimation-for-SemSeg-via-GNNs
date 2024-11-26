
from sklearn.metrics import average_precision_score, roc_auc_score, r2_score, mean_squared_error
from sklearn.model_selection import KFold
import numpy as np

from metaseg.utils import get_meta_model

def train_full(x,y,cfg):
    meta_model = get_meta_model(cfg[cfg.meta_model])
    classifier = meta_model.fit(x, y)

    perfs = {}
    if cfg.meta_target == 'iou0':
        auroc, auprc = roc_auc_score(y, classifier.predict_proba(x)[:, 1]), average_precision_score(~y, classifier.predict_proba(x)[:, 0])
        perfs["train_auroc"] = auroc
        perfs["train_auprc"] = auprc
        print("training auroc : {:.2f} %".format(auroc * 100))
        print("training auprc : {:.2f} %".format(auprc * 100))
    elif cfg.meta_target == 'iou':
        r2, rMSE = r2_score(y, classifier.predict(x)), np.sqrt(mean_squared_error(y, np.clip(classifier.predict(x), 0, 1)))
        perfs["train_r2"] = r2
        perfs["train_rMSE"] = rMSE
        print("training Rsquared: {:.2f} %".format(r2 * 100))
        print("training root rMSE: {:.4f} ".format(rMSE))
    
    return classifier, perfs

def validate(x_val,y_val,classifier,target_key = "iou"):
    perfs = {}
    if target_key == "iou0":
        val_auroc = roc_auc_score(y_val, classifier.predict_proba(x_val)[:, 1])
        perfs["val_auroc"] = val_auroc
        print("validation auroc: {:.2f} %".format(val_auroc * 100))
        val_auprc = average_precision_score(~y_val, classifier.predict_proba(x_val)[:, 0])
        perfs["val_auprc"] = val_auprc
        print("validation auprc: {:.2f} %".format(val_auprc * 100))

    elif target_key == "iou":
        val_r2 = r2_score(y_val, classifier.predict(x_val))
        perfs["val_r2"] = val_r2
        print("validation Rsquared: {:.2f} %".format(val_r2 * 100))
        val_rMSE = np.sqrt(mean_squared_error(y_val, np.clip(classifier.predict(x_val), 0, 1)))
        perfs["val_rMSE"] = val_rMSE
        print("validation root rMSE: {:.4f}".format(val_rMSE))
    return perfs

def kfold_validation(x,y,cfg, return_kfresults = False):
    k = cfg.val_multiparam
    print(f"Perform {k}Fold fitting to evaluate generalization:")

    kf = KFold(n_splits = k, shuffle = True, random_state = cfg.seed)
    perf_sums = {"train_auroc":0,"val_auroc":0,"train_auprc":0, "val_auprc":0} \
                    if cfg.meta_target == "iou0" else {"train_r2":0, "val_r2":0, "train_rMSE":0, "val_rMSE":0}
    
    """accelerate predicted y-values (and possibly probs) over kfold cross val process, always on non fitting data. init:"""
    if cfg.meta_target == "iou0":
        y_pred, y_proba = -2*np.ones_like(y),-2*np.ones_like(y, dtype='f')
    else:
        y_pred = -2*np.ones_like(y, dtype='f')

    for i, (train_idx, val_idx) in enumerate(kf.split(x)):
        print(f"K-Fold iteration {i+1} of {k}:")
        x_train, x_val = x[train_idx], x[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        """fit and evaluate meta model"""
        meta_model = get_meta_model(cfg[cfg.meta_model])
        classifier = meta_model.fit(x_train, y_train)

        if cfg.meta_target == "iou0":
            y_pred[val_idx], y_proba[val_idx] = classifier.predict(x_val), classifier.predict_proba(x_val)[:,1]
        else:
            y_pred[val_idx] = classifier.predict(x_val)

        """compute performance metrics"""
        if cfg.meta_target == "iou0":
            train_auroc, val_auroc = roc_auc_score(y_train, classifier.predict_proba(x_train)[:, 1]), \
                                         roc_auc_score(y_val, classifier.predict_proba(x_val)[:, 1])
            perf_sums["train_auroc"] += train_auroc
            perf_sums["val_auroc"] += val_auroc
            print("   training auroc:   {:.2f} %".format(train_auroc * 100))
            print("   validation auroc: {:.2f} %".format(val_auroc * 100))
            train_auprc, val_auprc = average_precision_score(~y_train, classifier.predict_proba(x_train)[:, 0]), \
                                         average_precision_score(~y_val, classifier.predict_proba(x_val)[:, 0])
            perf_sums["train_auprc"] += train_auprc
            perf_sums["val_auprc"] += val_auprc
            print("   training auprc:   {:.2f} %".format(train_auprc * 100))
            print("   validation auprc: {:.2f} %".format(val_auprc * 100))

        elif cfg.meta_target == "iou":
            train_r2, val_r2 = r2_score(y_train, classifier.predict(x_train)), r2_score(y_val, classifier.predict(x_val))
            perf_sums["train_r2"] += train_r2
            perf_sums["val_r2"] += val_r2
            print("   training Rsquared:   {:.2f} %".format(train_r2 * 100))
            print("   validation Rsquared: {:.2f} %".format(val_r2 * 100))
            train_rMSE, val_rMSE = np.sqrt(mean_squared_error(y_train, np.clip(classifier.predict(x_train), 0, 1))), \
                                 np.sqrt(mean_squared_error(y_val, np.clip(classifier.predict(x_val), 0, 1)))
            perf_sums["train_rMSE"] += train_rMSE
            perf_sums["val_rMSE"] += val_rMSE
            print("   training rMSE:       {:.4f}".format(train_rMSE))
            print("   validation rMSE:     {:.4f}".format(val_rMSE))
    
    print("\nKFold average performance metrics:")
    performance_metrics = {key: value/k for key, value in perf_sums.items()}
    if cfg.meta_target == "iou0":
        print("   training auroc:   {:.2f} %".format(performance_metrics["train_auroc"] * 100))
        print("   validation auroc: {:.2f} %".format(performance_metrics["val_auroc"] * 100))
        print("   training auprc:   {:.2f} %".format(performance_metrics["train_auprc"] * 100))
        print("   validation auprc: {:.2f} %".format(performance_metrics["val_auprc"] * 100))
    elif cfg.meta_target == "iou":
        print("   training Rsquared:   {:.2f} %".format(performance_metrics["train_r2"] * 100))
        print("   validation Rsquared: {:.2f} %".format(performance_metrics["val_r2"] * 100))
        print("   training rMSE:       {:.4f} ".format(performance_metrics["train_rMSE"]))
        print("   validation rMSE:     {:.4f} ".format(performance_metrics["val_rMSE"]))

    if return_kfresults and cfg.meta_target == 'iou0':
        return performance_metrics, [y_pred, y_proba] 
    elif return_kfresults: 
        return performance_metrics, y_pred
    else:
        return performance_metrics