import matplotlib.pyplot as plt
import numpy as np
import os
from os.path import join
import seaborn as sns
from sklearn.metrics import confusion_matrix, plot_roc_curve, roc_curve, auc 

def scatterplot(y,y_pred,directory,datatype = 'validation'):
    plt.scatter(y,y_pred, s = 1, alpha = 0.1)
    plt.xlabel('IoU')
    plt.ylabel('predicted IoU')
    plt.title('scatterplot ' + datatype + ' data')
    plt.xlim(0, 1)
    plt.ylim(min(y_pred)-0.01, max(y_pred)+0.01)
    plt.axis('scaled')

    os.makedirs(directory, exist_ok=True)
    plt.savefig(join(directory, 'scatterplot_' + datatype + '_data.png'))
    plt.close()

def confusion_matr(y,y_pred,directory,datatype = 'validation'):
    cm = np.flip(confusion_matrix(y, y_pred), axis =1)
    sns.heatmap(cm, annot=True, cmap='Blues',fmt='d',yticklabels =[1,0])
    plt.xlabel('IoU>0')
    plt.ylabel('predicted IoU>0')
    plt.title('confusion matrix ' + datatype + ' data')

    os.makedirs(directory, exist_ok=True)
    plt.savefig(join(directory, 'confusion matrix_' + datatype + '_data.png'))
    plt.close()

def roc_fromclassifier(classifier, x, y, directory, datatype = 'validation'):
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_roc_curve(classifier, x, y, ax=ax)

    ax.set_title('receiver operating characteristic (ROC) curve')
    ax.grid(True)
    plt.savefig(join(directory, 'roc_curve_' + datatype + '_data.png'))
    plt.close()

def roc_fromprobs(y, y_proba, directory, datatype = 'validation'):
    fpr, tpr, thresholds = roc_curve(y, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('receiver operating characteristic (ROC) curve')
    plt.legend(loc="lower right")
    plt.savefig(join(directory, 'roc_curve_' + datatype + '_data.png'))
    plt.close()

def feature_importances(classifier, features, directory):
    if hasattr(classifier,'feature_importances_'):
        importances = classifier.feature_importances_
    elif hasattr(classifier, 'coef_'):
        importances = classifier.coef_.ravel()
    else:
        print('Does your classifier have a feature_importances/coef_ like attribute? You might \
              consider adjusting metaseg.statistics_and_plots.feature_importances accordingly')
        exit()
    
    plt.figure(figsize=(22, 8))
    plt.bar(features, importances)
    plt.xticks(rotation=60)
    plt.xlabel('features')
    plt.ylabel('importance')
    plt.title('feature importance')
        
    plt.savefig(join(directory,'feature_importances.png'))
    plt.close()