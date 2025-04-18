#!/usr/bin/env
# -*- coding: utf-8 -*-

"""
post_aly.py: 
"""
import numpy as np
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from numpy import interp
from itertools import cycle
from sklearn.metrics import auc, roc_auc_score, roc_curve, accuracy_score, f1_score, balanced_accuracy_score, brier_score_loss, log_loss, precision_score, recall_score
from dl import args, logger
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os.path as osp
from scipy.special import softmax
import pandas as pd
from pathlib import Path
import os



def process_pred(pred_dict):
    y_true, y_predscore = pred_dict["y_true"], pred_dict["y_predscore"]
    y_predscore = softmax(y_predscore, axis=1)
    label_map = {0:'positive', 1:'unknown'}
    classes = sorted([int(i) for i in label_map.keys()])

    labels = ['positive', 'unknown']    

    y_pred = np.argmax(y_predscore, axis=1)
    n_classes = len(classes)
    y_true_onehot = label_binarize(y_true, classes=classes)

    proc_pred_dict = {
        "y_true": y_true,
        "y_true_onehot": y_true_onehot,
        "y_pred": y_pred,
        "y_predscore": y_predscore,
        "classes": classes,
        "n_classes": n_classes,
        "labels": labels,
    }
    return proc_pred_dict


def cal_pred_metrics(pred_dict):
    proc_pred_dict = process_pred(pred_dict)
    y_true = proc_pred_dict["y_true"]
    y_true_onehot = proc_pred_dict["y_true_onehot"]

    y_pred = proc_pred_dict["y_pred"]
    y_predscore = proc_pred_dict["y_predscore"]

    metric_dict = dict()
    cal_basic_metric(y_true, y_pred, y_true_onehot, y_predscore, metric_dict)
    cal_auc(y_true, y_predscore[:,1], metric_dict)

    return metric_dict


class PlotAly(object):
    def __init__(self, pred_dict, plot_path):
        proc_pred_dict = process_pred(pred_dict)
        self.y_true = proc_pred_dict["y_true"]
        self.y_true_onehot = proc_pred_dict["y_true_onehot"]

        self.y_pred = proc_pred_dict["y_pred"]
        self.y_predscore = proc_pred_dict["y_predscore"]

        self.n_classes = proc_pred_dict["n_classes"]
        self.labels = proc_pred_dict["labels"]

        self.plot_path = plot_path

    def plot_metrics(self):
        fpr, tpr, roc_auc = get_fpr_tpr(self.y_true, self.y_predscore[:,1])
        plot_roc(fpr, tpr, roc_auc, self.plot_path)
        plot_confusion_mat(self.y_true, self.y_pred, self.labels, self.plot_path)


def get_fpr_tpr(y_test_onehot, y_test_predscore):
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    fpr, tpr, _ = roc_curve(y_test_onehot, y_test_predscore)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc


def cal_basic_metric(y_test, y_test_pred, y_test_onehot, y_test_predscore, metric_dict):

    bal_acc = balanced_accuracy_score(y_test, y_test_pred)
    logger.debug(f'Test Ballance Acc: {bal_acc}')
    
    pr = precision_score(y_test, y_test_pred, average = 'macro')
    logger.debug(f'Test Precision: {pr}')
    
    f1_macro = f1_score(y_test, y_test_pred, average='macro')
    logger.debug(f'Test F1: {f1_macro}')
    
       
    metric_dict['acc'] = bal_acc
    metric_dict['f1'] = f1_macro
    metric_dict['pr'] = pr

def cal_auc(y_test_onehot, y_test_predscore, metric_dict):
    macro_roc_auc = roc_auc_score(y_test_onehot, y_test_predscore,
                                  average="macro")
    print("One-vs-One ROC AUC scores:\n{:.6f} (macro)".format(macro_roc_auc))
    
    metric_dict['macro_auc'] = macro_roc_auc


def plot_roc(fpr, tpr, roc_auc, save_path):
    fig, ax = plt.subplots(figsize=(8, 8))
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=25)
    plt.ylabel('True Positive Rate', fontsize=25)
    plt.legend(loc="lower right")
    plt.show()
    fig.savefig(osp.join(save_path, "ROC.png"))
    roc = pd.DataFrame(columns=['fpr','tpr'])
    roc['fpr'] = fpr
    roc['tpr'] = tpr
    if not Path(f"../test/roc").exists():
        os.mkdir(f"../test/roc")
    roc.to_csv(f"../test/roc/roc_{args.model}.csv",index=False)

def plot_confusion_mat(y_true, y_pred, labels, save_path):
    conf_mat = confusion_matrix(y_true, y_pred, normalize="true")

    cm = pd.DataFrame(columns=['true','predict'])
    cm['true'] = y_true
    cm['predict'] = y_pred
    if not Path(f"../test/cm").exists():
        os.mkdir(f"../test/cm")
    cm.to_csv(f"../test/cm/cm_{args.model}.csv",index=False)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(conf_mat, annot=True, fmt=".3f", linewidths=.5, cmap="YlGnBu", ax=ax, annot_kws={"fontsize":30})
    ax.set_ylabel('True labels', fontsize=23)
    ax.set_xlabel('Predicted labels', fontsize=23)

    ax.set_xticklabels(['positive','unknown'], fontsize=20)
    ax.set_yticklabels(['positive','unknown'], fontsize=20)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_yticklabels(), rotation=0, ha="right",
             rotation_mode="anchor")
    plt.show()
    fig.savefig(osp.join(save_path, f"confusion_mat.png"))
    fig.savefig(osp.join(save_path, f"confusion_mat.eps"))