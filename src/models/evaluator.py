from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import label_binarize
import numpy as np

def calculate_accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

def calculate_roc_auc(y_true, y_scores, num_classes):
    y_true_binarized = label_binarize(y_true, classes=np.arange(num_classes))
    roc_auc = roc_auc_score(y_true_binarized, y_scores, average='macro')
    return roc_auc