import argparse
import itertools
import torch
import data_preprocessing.augmentations as ts
import models.opt_tc as tc
import numpy as np
from data_preprocessing.dataloader import loading_data, count_label_labellist
from sklearn.model_selection import train_test_split
import random, math
#from data_loader import Data_Loader
import pandas as pd

# visualization
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import RocCurveDisplay
from sklearn import preprocessing
from sklearn.preprocessing import LabelBinarizer


def transform_data(data, trans):
    trans_inds = np.tile(np.arange(trans.n_transforms), len(data))
    print(data.shape)
    trans_data = trans.transform_batch(np.repeat(data.numpy(), trans.n_transforms, axis=0), trans_inds)
    return trans_data, trans_inds

def data_generator_goad(args, datalist, labellist):

    test_ratio = args.test_ratio
    seed =  args.seed 
    # num_classes, entire_list, train_list, valid_list, test_list, entire_label_list, train_label_list, valid_label_list, test_label_list \
    # = splitting_data(args.selected_dataset, args.test_ratio, args.valid_ratio, args.padding, args.seed, \
    #                  args.timespan, args.min_seq, args.min_samples, args.aug_method, args.aug_wise)
    train_list, test_list, train_label_list, test_label_list = train_test_split(datalist, 
                                                                                labellist, test_size=test_ratio, stratify= labellist, random_state=seed) 
    print(f"Train Data: {len(train_list)} --------------")
    exist_labels, _ = count_label_labellist(train_label_list)    

    print(f"Test Data: {len(test_list)} --------------")
    count_label_labellist(test_label_list) 

    train_list = torch.tensor(train_list).cuda().cpu()
    train_label_list = torch.tensor(train_label_list).cuda().cpu()

    test_list = torch.tensor(test_list).cuda().cpu()
    test_label_list = torch.tensor(test_label_list).cuda().cpu()
    

    
    if (args.one_class_idx != -1):
        
        train_list = train_list[np.where(train_label_list == args.one_class_idx)]
        train_label_list = train_label_list[np.where(train_label_list == args.one_class_idx)]
        test_label_list  = np.array(test_label_list ) == args.one_class_idx

    else:
    # multi-class
        sup_class_idx = [x for x in exist_labels]
        random.seed(args.seed)
        known_class_idx = random.sample(sup_class_idx, math.ceil(len(sup_class_idx)/2))
        #known_class_idx = [x for x in range(0, (int)(len(sup_class_idx)/2))]
        #known_class_idx = [0, 1]
        novel_class_idx = [item for item in sup_class_idx if item not in set(known_class_idx)]
        
        train_list = train_list[np.isin(train_label_list, known_class_idx)]
        train_label_list = train_label_list[np.isin(train_label_list, known_class_idx)]
        test_label_list = [True if i in known_class_idx else False for i in test_label_list]

    # only use for testing novelty
    return train_list, test_list, torch.tensor(test_label_list).cuda().cpu()

def load_trans_data(args, trans, datalist, labellist):
    #dl = Data_Loader()
    #_, datalist, labellist = loading_data(args.dataset, args)
    x_train, x_test, y_test = data_generator_goad(args, datalist, labellist)
    
    x_train_trans, _ = transform_data(x_train, trans)
    x_test_trans, _ = transform_data(x_test, trans)

    x_test_trans, x_train_trans = x_test_trans.transpose(0, 2, 1), x_train_trans.transpose(0, 2, 1)

    #print(y_test)

    return x_train_trans, x_test_trans, y_test


def train_anomaly_detector(args, config, datalist, labellist):

    transformer = ts.get_transformer(args, config)
    x_train, x_test, y_test = load_trans_data(args, transformer, datalist, labellist)
    print("final shape:",x_train.shape, x_test.shape, y_test.shape, transformer.n_transforms)
    
    # train 은 하나의 idx를 기반으로 하기 때문에 test의 크기가 더 클 수 있음
    tc_obj = tc.TransClassifier(transformer.n_transforms, args, configs)
    auroc_rs, aupr_rs, fpr_rs, de_rs, scores, labels = tc_obj.fit_trans_classifier(x_train, x_test, y_test)
    
    return auroc_rs, aupr_rs, fpr_rs, de_rs, scores, labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GOAD for time series')
    # Model options
    parser.add_argument('--depth', default=10, type=int)
    parser.add_argument('--widen-factor', default=4, type=int)

    # Training options
    parser.add_argument('--batch_size', default=288, type=int)
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float)
    parser.add_argument('--epochs', default=16, type=int)

    # Trans options
    parser.add_argument('--type_trans', default='complicated', type=str)

    # CT options
    parser.add_argument('--lmbda', default=0.1, type=float)
    parser.add_argument('--m', default=1, type=float)
    parser.add_argument('--reg', default=True, type=bool)
    parser.add_argument('--eps', default=0, type=float)

    # Exp options
    #parser.add_argument('--class_ind', default=1, type=int)
    parser.add_argument('--dataset', default='lapras', type=str)
    parser.add_argument('--padding', type=str, 
                    default='mean', help='choose one of them : no, max, mean')
    parser.add_argument('--timespan', type=int, default=10000, 
                        help='choose of the number of timespan between data points (1000 = 1sec, 60000 = 1min)')
    parser.add_argument('--min_seq', type=int, 
                    default=10, help='choose of the minimum number of data points in a example')
    parser.add_argument('--min_samples', type=int, default=20, 
                    help='choose of the minimum number of samples in each label')
    parser.add_argument('--one_class_idx', type=int, default=0, 
                    help='choose of one class label number that wants to deal with. -1 is for multi-classification')
    parser.add_argument('--aug_method', type=str, default='AddNoise', 
                        help='choose the data augmentation method')
    parser.add_argument('--aug_wise', type=str, default='Temporal', 
                        help='choose the data augmentation wise')
    parser.add_argument('--test_ratio', type=float, default=0.2, 
                        help='choose the number of test ratio')
    parser.add_argument('--seed', default = 42, type=int,
                    help='seed value')
    
    args = parser.parse_args()
    
    data_type = args.dataset
    exec(f'from config_files.{data_type}_Configs import Config as Configs')
    configs = Configs()

    if data_type == 'lapras': 
        class_num = [0, 1, 2, 3, -1]
        args.timespan = 10000
    elif data_type == 'casas': 
        class_num = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, -1]
        args.aug_wise = 'Temporal2'
    elif data_type == 'opportunity': 
        class_num = [0, 1, 2, 3, 4, -1]
        args.timespan = 1000
    elif data_type == 'aras_a': 
        class_num = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, -1]
        args.timespan = 1000

    final_auroc = []
    final_aupr  = []
    final_fpr   = []
    final_de    = []

    y_onehot_test=[]
    y_score = []

    num_classes, datalist, labellist = loading_data(data_type, args)

    for args.one_class_idx in class_num:
        auroc_a = []
        aupr_a  = []
        fpr_a   = []
        de_a    = []

        testy_rs = []
        scores_rs = []
        
        seed_num = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]

        # Training for five seed #
        for test_num in seed_num :
            # ##### fix random seeds for reproducibility ########
            SEED = args.seed = test_num
            torch.manual_seed(SEED)
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = False
            np.random.seed(SEED)            
            #####################################################
            print("=" * 45)
            print("Dataset:", args.dataset)
            print("True Class:", args.one_class_idx)
            print(f'Seed:    {SEED}')
            print("=" * 45)
            one_class_total, one_class_aupr, one_class_fpr, one_class_de, scores, labels = train_anomaly_detector(args, configs, datalist, labellist)
            
            
            auroc_a.append(one_class_total)     
            aupr_a.append(one_class_aupr)   
            fpr_a.append(one_class_fpr)
            de_a.append(one_class_de)
            testy_rs.append(labels.tolist())
            scores_rs.append(scores.tolist())
        
        testy_rs, scores_rs = list(itertools.chain.from_iterable(testy_rs)), list(itertools.chain.from_iterable(scores_rs))
        final_auroc.append([np.mean(auroc_a), np.std(auroc_a)])
        final_aupr.append([np.mean(aupr_a), np.std(aupr_a)])
        final_fpr.append([np.mean(fpr_a), np.std(fpr_a)])
        final_de.append([np.mean(de_a), np.std(de_a)])

        # for visualization
        onehot_encoded = list()        
        label_binarizer = LabelBinarizer().fit(testy_rs)
        onehot_encoded = label_binarizer.transform(testy_rs)
        #print(onehot_encoded.shape)
        #print(label_binarizer.transform([1]))
        y_onehot_test.append(onehot_encoded)
        y_score.append(scores_rs)


    # for extrating results to an excel file
    
    final_rs =[]
    for i in final_auroc:
        final_rs.append(i)
    for i in final_aupr:
        final_rs.append(i)
    for i in final_fpr:
        final_rs.append(i)
    for i in final_de:
        final_rs.append(i)

    print("Finished")

    df = pd.DataFrame(final_rs, columns=['mean', 'std'])
    df.to_excel('result_files/GOAD_'+data_type+'.xlsx', sheet_name='the results')

    
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
                name=f"ROC curve for {(class_num[class_id]+1)}",
                color=color,
                ax=ax, 
            )
        else:
            RocCurveDisplay.from_predictions(
                y_onehot_test[class_id],
                y_score[class_id],
                name=f"ROC curve for Multi",
                color="black",
                ax=ax, 
            )

            
    plt.axis("square")
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Chance Level (0.5)')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curves of GOAD")
    plt.legend()
    plt.show()
    plt.savefig('figure/GOAD_ROC_'+args.dataset+'.png')



