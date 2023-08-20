import argparse
from collections import namedtuple
import itertools
import sys

import numpy as np
from sklearn import svm
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score,roc_curve
from ood_metrics import auroc, fpr_at_95_tpr
from data_preprocessing.dataloader import count_label_labellist
import random, math
from sklearn.model_selection import train_test_split
import torch

from util import calculate_acc, visualization_roc, print_rs

import requests
import json

# visualization
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import RocCurveDisplay
from sklearn import preprocessing
from sklearn.preprocessing import LabelBinarizer

def send_slack_message(payload, webhook):

    return requests.post(webhook, json.dumps(payload))

def parse_args():
    parser = argparse.ArgumentParser(description='Anomaly Detection by One Class SVM')

    parser.add_argument('--data_path', default='./data/cifar10_cae.npz', type=str, help='path to dataset')

    parser.add_argument('--rate_anomaly_test', default=1, type=float,
                        help='rate of abnormal data versus normal data in test data. The default setting is 10:1(=0.1)')
    parser.add_argument('--test_rep_count', default=10, type=int,
                        help='counts of test repeats per one trained model. For a model, test data selection and evaluation are repeated.')
    parser.add_argument('--TRAIN_RAND_SEED', default=42, type=int, help='random seed used selecting training data')
    parser.add_argument('--TEST_RAND_SEED', default=[42, 89, 2, 156, 491, 32, 67, 341, 100, 279], type=list,
                        help='random seed used selecting test data.'
                             'The number of elements should equal to "test_rep_count" for reproductivity of validation.'
                             'When the length of this list is less than "test_rep_count", seed is randomly generated')
    
    parser.add_argument('--one_class_idx', type=int, default = 0, 
                    help='choose of one class label number that wants to deal with. -1 is for multi-classification')
    # for data loading 
    parser.add_argument('--padding', type=str, 
                    default='mean', help='choose one of them : no, max, mean')
    parser.add_argument('--timespan', type=int, 
                        default=1000, help='choose of the number of timespan between data points(1000 = 1sec, 60000 = 1min)')
    parser.add_argument('--min_seq', type=int, 
                        default=10, help='choose of the minimum number of data points in a example')
    parser.add_argument('--min_samples', type=int, default=20, 
                        help='choose of the minimum number of samples in each label')
    parser.add_argument('--dataset', default='lapras', type=str,
                        help='Dataset of choice: lapras, casas, opportunity, aras_a, aras_b')

    parser.add_argument('--test_ratio', type=float, default=0.2, help='choose the number of test ratio')

    args = parser.parse_args()

    return args


def load_data(data_to_path):
    """load data
    data should be compressed in npz
    """
    data = np.load(data_to_path)

    try:
        full_data = data['ae_out']
        full_labels = data['labels']
    except:
        print('Loading data should be numpy array and has "ae_out" and "labels" keys.')
        sys.exit(1)

    return full_data, full_labels


def prepare_data(full_data, full_labels, normal_label,TRAIN_RAND_SEED):
    """prepare data
    split data into anomaly data and normal data
    """
    TRAIN_DATA_RNG = np.random.RandomState(TRAIN_RAND_SEED)
    seed =   np.random.RandomState(TRAIN_RAND_SEED)

    # num_classes, entire_list, train_list, valid_list, test_list, entire_label_list, train_label_list, valid_label_list, test_label_list \
    # = splitting_data(args.selected_dataset, args.test_ratio, args.valid_ratio, args.padding, args.seed, \
    #                  args.timespan, args.min_seq, args.min_samples, args.aug_method, args.aug_wise)
    train_list, test_list, train_label_list, test_label_list = train_test_split(full_data, 
                                                                                full_labels, test_size = args.test_ratio, stratify = full_labels, random_state=seed) 
    print(f"Train Data: {len(train_list)} --------------")
    exist_labels, _ = count_label_labellist(train_label_list)    

    print(f"Test Data: {len(test_list)} --------------")
    count_label_labellist(test_label_list) 

    train_list = torch.tensor(train_list).cuda().cpu()
    train_label_list = torch.tensor(train_label_list).cuda().cpu()

    test_list = torch.tensor(test_list).cuda().cpu()
    test_label_list = torch.tensor(test_label_list).cuda().cpu()
    #print( train_list.shape, test_list.shape, test_label_list.shape)
  

    if (normal_label != -1):
        sup_class_idx = [x for x in exist_labels]
        known_class_idx  = [normal_label]
        novel_class_idx = [item for item in sup_class_idx if item not in set(known_class_idx)]

        train_x = train_list[np.where(train_label_list == normal_label)]

        testx_n = test_list [np.where(test_label_list == normal_label)]
        testy_n = test_label_list [np.where(test_label_list == normal_label)]
        ano_x = test_list [np.where(test_label_list != normal_label)]
        ano_y = test_label_list[np.where(test_label_list != normal_label)]

    else:
    # multi-class
        sup_class_idx = [x for x in exist_labels]
        random.seed(TRAIN_DATA_RNG)
        known_class_idx = random.sample(sup_class_idx, math.ceil(len(sup_class_idx)/2))
        #known_class_idx = [x for x in range(0, (int)(len(sup_class_idx)/2))]
        #known_class_idx = [0, 1]
        novel_class_idx = [item for item in sup_class_idx if item not in set(known_class_idx)]
        
        train_x = train_list[np.isin(train_label_list, known_class_idx)]

        testx_n = test_list [np.isin(test_label_list, known_class_idx)]
        testy_n = test_label_list [np.isin(test_label_list, known_class_idx)]        
        ano_x = test_list[np.isin(test_label_list, novel_class_idx)]
        ano_y = test_label_list[np.isin(test_label_list, novel_class_idx)]

    train_label_list[:] = 0
    
    # replace label : anomaly -> 1 : normal -> 0
    ano_y[:] = 1
    testy_n[:] = 0


    split_data = namedtuple('split_data', ('train_x', 'testx_n', 'testy_n', 'ano_x', 'ano_y'))     
    

    return split_data(
        train_x=train_x,
        testx_n=testx_n,
        testy_n=testy_n,
        ano_x=ano_x,
        ano_y=ano_y
    )


def make_test_data(split_data, RNG):
    """make test data which has specified mixed rate(rate_anomaly_test).
    shuffle and concatenate normal and abnormal data"""

    ano_x = split_data.ano_x
    ano_y = split_data.ano_y
    testx_n = split_data.testx_n
    testy_n = split_data.testy_n
    print('Test num',len(ano_x), len(testx_n))

    # anomaly data in test
    inds_1 = RNG.permutation(ano_x.shape[0])
    ano_x = ano_x[inds_1]
    ano_y = ano_y[inds_1]

    # index_1 = int(testx_n.shape[0])
    testx_a = ano_x
    testy_a = ano_y

    print('Test num',len(testx_a), len(testx_n))
    # concatenate test normal data and test anomaly data
    testx = np.concatenate([testx_a, testx_n], axis=0)
    testy = np.concatenate([testy_a, testy_n], axis=0)

    return testx, testy.tolist()


# def calc_metrics(testy, scores):
#     precision, recall, _ = precision_recall_curve(testy, scores)
#     roc_auc = roc_auc_score(testy, scores)
#     prc_auc = auc(recall, precision)

#     return roc_auc, prc_auc


def anomaly_detection(args):

     
    normal_label = args.one_class_idx

    data_type = args.dataset
   
    
    auroc_scores = []
    aupr_scores = []
    fpr_scores = []
    de_scores = []

    best_auroc = 0
    testy_rs = []
    scores_rs = []


    seed_num = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
        
    # nu : the upper limit ratio of anomaly data(0<=nu<=1)
    nus = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    # train model and evaluate with changing parameter nu
    for SEED in seed_num:

        # total_auroc  = 0
        # total_aupr = 0
        # total_fpr = 0
        # total_de = 0    
        data_path = './data/' + data_type + '_cae.npz' 
        # # ##### fix random seeds for reproducibility ########
        # SEED = args.seed = 20
        # np.random.seed(SEED)
        # #####################################################
        # load and prepare data
        full_data, full_labels = load_data(data_path)
                
        best_nu_rs = 0 
        best_nu = 0 

        # select test data and test
        TEST_SEED = np.random.RandomState(SEED)
        split_data = prepare_data(full_data, full_labels, normal_label, SEED)   
        
        scores_nu = []   
        labels_nu = []

        # repeat test by randomly selected data and evaluate   
        for nu in nus:
            
            print('='*45)
            print("True Class:", args.one_class_idx)            
            print("Dataset:", data_type)
            print("Seed:", SEED)
            print('nu',nu )
            print('='*45)

            # train with nu
            clf = svm.OneClassSVM(nu=nu, kernel='rbf', gamma='auto')

                     
            clf.fit(split_data.train_x)
            
            testx, l = make_test_data(split_data, TEST_SEED)

            s = clf.decision_function(testx).ravel() * (-1)
            
            scores_nu = scores_nu + s.tolist()
            labels_nu = labels_nu + l
   
        auroc_rs, fpr, f1, acc  = calculate_acc(labels_nu, scores_nu)   

            #find best nu for each seed
            #if auroc_rs_nu > best_nu_rs:  
            #    best_nu_rs = auroc_rs_nu
            #    best_nu = nu
        scores, labels = scores_nu, labels_nu

        print('Best nu,{} AUROC: {:.3f}'.format(best_nu, best_nu_rs))
           

        # calculate evaluation metrics       
        auroc_scores.append(auroc_rs)
        aupr_scores.append(fpr)
        fpr_scores.append(f1)
        de_scores.append(acc)

        testy_rs = testy_rs + labels
        scores_rs = scores_rs + scores     

    
    print(len(testy_rs), len(scores_rs))

    
    return [np.mean(auroc_scores), np.std(auroc_scores)],[np.mean(aupr_scores), np.std(aupr_scores)], \
           [np.mean(fpr_scores), np.std(fpr_scores)], [np.mean(de_scores), np.std(de_scores)], \
           testy_rs, scores_rs

if __name__ == '__main__':
#def start_test():
    # set parameters
    args = parse_args()
    final_auroc = []
    final_aupr  = []
    final_fpr   = []
    final_de    = []
    final_visu = []
    y_onehot_test=[]
    y_score = []
    validation = []

    if args.dataset == 'lapras': 
        class_num = [0, 1, 2, 3, -1]
    elif args.dataset == 'casas': 
        class_num = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, -1]
        args.aug_wise = 'Temporal2'
    elif args.dataset == 'opportunity': 
        class_num = [0, 1, 2, 3, 4, -1]
    elif args.dataset == 'aras_a': 
        class_num = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, -1]
    
    #lb = preprocessing.LabelBinarizer()

    for args.one_class_idx in class_num:        
        
        a,b,c,d, testy_rs, scores_rs = anomaly_detection(args)

        final_auroc.append(a)
        final_aupr.append(b)
        final_fpr.append(c)
        final_de .append(d)        
        #print(testy_rs)
        onehot_encoded = list()        
        label_binarizer = LabelBinarizer().fit(testy_rs)
        

        onehot_encoded = label_binarizer.transform(testy_rs)
        print(onehot_encoded.shape)
        print(label_binarizer.transform([1]))
        y_onehot_test.append(onehot_encoded)
        y_score.append(scores_rs)
        
        auroc_rs, aupr_rs, fpr_at_95_tpr_rs, detection_error_rs =  calculate_acc(testy_rs, scores_rs)
        validation.append([auroc_rs,0])

        # print(len(y_onehot_test))
        # print(len(y_score))

   

    # alert to slack
    webhook = "https://hooks.slack.com/services/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"
    payload = {"text": "Experiment_"+args.dataset+" Finished!"}
    send_slack_message(payload, webhook)

    # file save
    save_path = 'result_files/OCSVM_'+args.dataset+'.xlsx'
    print_rs(final_auroc, final_aupr, final_fpr, final_de, validation, save_path)


    # visualization
    vis_title = 'ROC curves of OC-SVM'
    vis_path = 'figure/OC-SVM_ROC_'+args.dataset+'.png'
    visualization_roc(class_num, y_onehot_test, y_score, vis_title, vis_path)

    print("Finished")
