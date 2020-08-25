import numpy as np

from sklearn import metrics


def auc(y_true, y_pred, **kwargs):
    # y_pred.shape = (N, C)
    # AUC for melanoma (class 0)
    return {'auc': metrics.roc_auc_score((y_true==0).astype('float'), y_pred[:,0])}


def mel_auc(y_true, y_pred, **kwargs):
    # y_pred.shape = (N, C)
    # AUC for melanoma + nevi (class 0+1)
    return {'mel_auc': metrics.roc_auc_score((y_true<=1).astype('float'), y_pred[:,0]+y_pred[:,1])}


def mel_f1(y_true, y_pred, **kwargs):
    # y_pred.shape = (N, C)
    # AUC for melanoma + nevi (class 0+1)
    t = (y_true <= 1).astype('float')
    p = (y_pred[:,0] + y_pred[:,1]) >= 0.5
    p = p.astype('float')
    return {'mel_f1': metrics.f1_score(t, p)}


def accuracy(y_true, y_pred, **kwargs):
    return {'accuracy': np.mean(y_true == np.argmax(y_pred, axis=1))}


def auc2(y_true, y_pred, **kwargs):
    # y_pred.shape = (N, 2)
    # AUC for melanoma (class 1)
    return {'auc2': metrics.roc_auc_score(y_true, y_pred)}


def arc_auc(y_true, y_pred, **kwargs):
    # y_pred.shape = (N, 2)
    # AUC for melanoma (class 1)
    return {'arc_auc': metrics.roc_auc_score(y_true, y_pred)}


def auc3(y_true, y_pred, **kwargs):
    # y_pred.shape = (N, 3) - includes prediction for nevus
    t = (y_true == 2).astype('float')
    p = y_pred[:,2]
    return {'auc3': metrics.roc_auc_score(t, p)}