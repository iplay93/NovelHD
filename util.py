import torch
import random
import numpy as np
import pandas as pd
import os
import sys
import logging
from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix, accuracy_score
from shutil import copy

def set_requires_grad(model, dict_, requires_grad=True):
    for param in model.named_parameters():
        if param[0] in dict_:
            param[1].requires_grad = requires_grad


def fix_randomness(SEED):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def _calc_metrics(pred_labels, true_labels, log_dir, home_path):
    pred_labels = np.array(pred_labels).astype(int)
    true_labels = np.array(true_labels).astype(int)

    # save targets
    labels_save_path = os.path.join(log_dir, "labels")
    os.makedirs(labels_save_path, exist_ok=True)
    np.save(os.path.join(labels_save_path, "predicted_labels.npy"), pred_labels)
    np.save(os.path.join(labels_save_path, "true_labels.npy"), true_labels)

    r = classification_report(true_labels, pred_labels, digits=6, output_dict=True)
    cm = confusion_matrix(true_labels, pred_labels)
    df = pd.DataFrame(r)
    df["cohen"] = cohen_kappa_score(true_labels, pred_labels)
    df["accuracy"] = accuracy_score(true_labels, pred_labels)
    df = df * 100

    # save classification report
    exp_name = os.path.split(os.path.dirname(log_dir))[-1]
    training_mode = os.path.basename(log_dir)
    file_name = f"{exp_name}_{training_mode}_classification_report.xlsx"
    report_Save_path = os.path.join(home_path, log_dir, file_name)
    df.to_excel(report_Save_path)

    # save confusion matrix
    cm_file_name = f"{exp_name}_{training_mode}_confusion_matrix.torch"
    cm_Save_path = os.path.join(home_path, log_dir, cm_file_name)
    torch.save(cm, cm_Save_path)


def _logger(logger_name, level=logging.DEBUG):
    """
    Method to return a custom logger with the given name and level
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    # format_string = ("%(asctime)s — %(name)s — %(levelname)s — %(funcName)s:"
    #                 "%(lineno)d — %(message)s")
    format_string = "%(message)s"
    log_format = logging.Formatter(format_string)
    # Creating and adding the console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)
    # Creating and adding the file handler
    file_handler = logging.FileHandler(logger_name, mode='a')
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    return logger


def copy_Files(destination, data_type):
    destination_dir = os.path.join(destination, "model_files")
    os.makedirs(destination_dir, exist_ok=True)
    copy("main.py", os.path.join(destination_dir, "main.py"))
    copy("trainer/trainer.py", os.path.join(destination_dir, "trainer.py"))
    copy(f"config_files/{data_type}_Configs.py", os.path.join(destination_dir, f"{data_type}_Configs.py"))
    copy("data_preprocessing/augmentations.py", os.path.join(destination_dir, "augmentations.py"))
    copy("data_preprocessing/dataloader.py", os.path.join(destination_dir, "dataloader.py"))
    copy(f"models/model.py", os.path.join(destination_dir, f"model.py"))
    copy("models/loss.py", os.path.join(destination_dir, "loss.py"))
    copy("models/TC.py", os.path.join(destination_dir, "TC.py"))




def load_linear_checkpoint(logdir, mode='last'):
    if mode == 'last':
        linear_optim_path = os.path.join(logdir, 'last.linear_optim')
    elif mode == 'best':
        linear_optim_path = os.path.join(logdir, 'best.linear_optim')
    else:
        raise NotImplementedError()

    print("=> Loading linear optimizer checkpoint from '{}'".format(logdir))
    if os.path.exists(linear_optim_path):
        linear_optim_state = torch.load(linear_optim_path)
        return linear_optim_state
    else:
        return None


def save_linear_checkpoint(linear_optim_state, logdir):
    last_linear_optim = os.path.join(logdir, 'last.linear_optim')
    torch.save(linear_optim_state, last_linear_optim)

from ood_metrics import auroc, fpr_at_95_tpr
from sklearn.metrics import roc_auc_score,  f1_score,  roc_curve, auc

def calculate_acc(labels, scores):

    # Input: Lists
    
    perc_ths = (100*labels.count(0)/(labels.count(1)+labels.count(0))) 
    outputs = np.array(scores)
    #print(outputs)
    thres = np.percentile(outputs, perc_ths)    
    outputs = [1 if value > thres else 0 for value in outputs]
    #print(outputs)

    # f1
    f1 = f1_score(np.array(labels), outputs)

    fp, tn, fn, tp = 0, 0, 0, 0
    for k in range(len(outputs)):
        if outputs[k] == 1 and labels[k] == 0:  fp +=1
        elif outputs[k] == 0 and labels[k] == 0: tn += 1
        elif outputs[k] == 0 and labels[k] == 1: fn += 1
        elif outputs[k] == 1 and labels[k] == 1: tp += 1

    spec    = tn / (fp + tn)
    recall  = tp / (tp + fn)
    acc = (recall + spec) / 2
    precision = tp / (tp + fp)
    F1_rs =  2 * (precision * recall) / (precision + recall) #f1_score과 같음을 확인함

    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=0)
    auc_min = auc(fpr, tpr)

    print('='*45)
    print('0 labels: {} 1 labels: {} percentile thresh: {:.3f} thrsh num: {:.3f}'.format(labels.count(0), labels.count(1), perc_ths, thres))
    print('AUROC: {:.3f} and {:.3f} '.format(auroc(scores, labels), auc_min))
    print('fpr_at_95_tpr: {:.3f}'.format(fpr_at_95_tpr(scores, labels)))
    print('F1: {:.3f}'.format(f1))
    print('Balanced accuracy: {:.3f}'.format(acc))
    print('='*45)

    return auroc(scores, labels), fpr_at_95_tpr(scores, labels), f1, acc


def calculate_acc_rv(labels, scores):

    # Input: Lists
    
    perc_ths = (100*labels.count(0)/(labels.count(1)+labels.count(0))) 
    outputs = np.array(scores)
    #print(outputs)
    thres = np.percentile(outputs, perc_ths)    
    outputs = [1 if value > thres else 0 for value in outputs]
    #print(outputs)

    # f1
    f1 = f1_score(np.array(labels), outputs, pos_label=0)

    fp, tn, fn, tp = 0, 0, 0, 0
    for k in range(len(outputs)):
        if outputs[k] == 0 and labels[k] == 1:  fp +=1
        elif outputs[k] == 1 and labels[k] == 1: tn += 1
        elif outputs[k] == 1 and labels[k] == 0: fn += 1
        elif outputs[k] == 0 and labels[k] == 0: tp += 1

    spec    = tn / (fp + tn)
    recall  = tp / (tp + fn)
    acc = (recall + spec) / 2
    precision = tp / (tp + fp)
    F1_rs =  2 * (precision * recall) / (precision + recall) #f1_score과 같음을 확인함

    
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=0)
    auc_min = auc(fpr, tpr)

    print('='*45)
    print('0 labels: {} 1 labels: {} percentile thresh: {:.3f} thrsh num: {:.3f}'.format(labels.count(0), labels.count(1), perc_ths, thres))
    print('AUROC: {:.3f} and {:.3f} '.format(auroc(scores, labels), auc_min))
    print('fpr_at_95_tpr: {:.3f}'.format(fpr_at_95_tpr(scores, labels)))
    print('F1: {:.3f}'.format(f1))
    print('F1_2: {:.3f}'.format(F1_rs))
    print('Balanced accuracy: {:.3f}'.format(acc))
    print('='*45)

    return auroc(scores, labels), fpr_at_95_tpr(scores, labels), f1, acc



# visualization
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import RocCurveDisplay


def visualization_roc(class_num, y_onehot_test, y_score, vis_title, vis_path, pos_label = 1):
    
    # visualization        
    fig, ax = plt.subplots(figsize=(6, 6))
    if(len(class_num)<=5):
        colors = cycle(["tomato", "darkorange", "gold", "darkseagreen","dodgerblue"])
    else:
        colors = cycle(["firebrick", "tomato", "sandybrown", "darkorange", "olive", "gold", 
                        "darkseagreen", "darkgreen", "dodgerblue", "royalblue","slategrey",
                        "slateblue", "mediumpurple","indigo", "orchid", "hotpink"])
    for class_id, color in zip(range(len(class_num)), colors):
        #print(y_onehot_test[class_id])
        #print(y_score[class_id].tolist())
        if class_num[class_id] != -1:
            RocCurveDisplay.from_predictions(
                y_onehot_test[class_id],
                y_score[class_id],
                name=f"{(class_num[class_id]+1)}",
                pos_label = pos_label,
                color=color,
                ax=ax, 
            )
        else:
            RocCurveDisplay.from_predictions(
                y_onehot_test[class_id],
                y_score[class_id],
                pos_label = pos_label,
                name=f"Multi",
                color="black",
                ax=ax, 
            )

            
    plt.axis("square")
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Chance Level (0.5)')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(vis_title)
    plt.legend()
    plt.show()
    plt.savefig(vis_path)

import pandas as pd

def print_rs(final_1, final_2, final_3, final_4, final_5, save_path):
     # for extrating results to an excel file
    final_rs =[]
    for i in final_1:
        final_rs.append(i)
    for i in final_2:
        final_rs.append(i)
    for i in final_3:
        final_rs.append(i)
    for i in final_4:
        final_rs.append(i)
    for i in final_5:
        final_rs.append(i)
        
    df = pd.DataFrame(final_rs, columns=['mean', 'std'])
    df.to_excel(save_path, sheet_name='the results')