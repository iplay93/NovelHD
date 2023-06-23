import argparse
from collections import namedtuple
import sys

import numpy as np
from sklearn import svm
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score
from ood_metrics import auroc, aupr, fpr_at_95_tpr, detection_error
from data_preprocessing.dataloader import count_label_labellist
import random, math

def parse_args():
    parser = argparse.ArgumentParser(description='Anomaly Detection by One Class SVM')

    parser.add_argument('--data_path', default='./data/cifar10_cae.npz', type=str, help='path to dataset')

    parser.add_argument('--rate_normal_train', default=0.82, type=float, help='rate of normal data to use in training')
    parser.add_argument('--rate_anomaly_test', default=0.1, type=float,
                        help='rate of abnormal data versus normal data in test data. The default setting is 10:1(=0.1)')
    parser.add_argument('--test_rep_count', default=10, type=int,
                        help='counts of test repeats per one trained model. For a model, test data selection and evaluation are repeated.')
    parser.add_argument('--TRAIN_RAND_SEED', default=42, type=int, help='random seed used selecting training data')
    parser.add_argument('--TEST_RAND_SEED', default=[42, 89, 2, 156, 491, 32, 67, 341, 100, 279], type=list,
                        help='random seed used selecting test data.'
                             'The number of elements should equal to "test_rep_count" for reproductivity of validation.'
                             'When the length of this list is less than "test_rep_count", seed is randomly generated')
    
    parser.add_argument('--one_class_idx', type=int, default=-1, 
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
    parser.add_argument('--selected_dataset', default='lapras', type=str,
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


def prepare_data(full_data, full_labels, normal_label, rate_normal_train, TRAIN_RAND_SEED):
    """prepare data
    split data into anomaly data and normal data
    """
    TRAIN_DATA_RNG = np.random.RandomState(TRAIN_RAND_SEED)
    
    if(normal_label != -1):
        # data whose label corresponds to anomaly label, otherwise treated as normal data
        ano_x = full_data[full_labels != normal_label]
        ano_y = full_labels[full_labels != normal_label]
        normal_x = full_data[full_labels == normal_label]
        normal_y = full_labels[full_labels == normal_label]
        
    elif (normal_label == -1):# multi-class
        exist_labels, _ = count_label_labellist(full_labels)
        sup_class_idx = [x for x in exist_labels]
        random.seed(TRAIN_RAND_SEED)
        known_class_idx = random.sample(sup_class_idx, math.ceil(len(sup_class_idx)/2))
        #known_class_idx = [x for x in range(0, (int)(len(sup_class_idx)/2))]
        #known_class_idx = [0, 1]
        novel_class_idx = [item for item in sup_class_idx if item not in set(known_class_idx)]
        print(novel_class_idx)
        print(known_class_idx)

        ano_x = full_data[np.isin(full_labels, novel_class_idx)]
        ano_y = full_labels[np.isin(full_labels, novel_class_idx)]
        normal_x = full_data[np.isin(full_labels, known_class_idx)]
        normal_y = full_labels[np.isin(full_labels, known_class_idx)]
        
    # replace label : anomaly -> 1 : normal -> 0
    ano_y[:] = 1
    normal_y[:] = 0

    # shuffle normal data and label
    inds = TRAIN_DATA_RNG.permutation(normal_x.shape[0])
    normal_x_data = normal_x[inds]
    normal_y_data = normal_y[inds]

    # split normal data into train and test
    index = int(normal_x.shape[0] * rate_normal_train)
    trainx = normal_x_data[:index]
    testx_n = normal_x_data[index:]
    testy_n = normal_y_data[index:]

    split_data = namedtuple('split_data', ('train_x', 'testx_n', 'testy_n', 'ano_x', 'ano_y'))

    return split_data(
        train_x=trainx,
        testx_n=testx_n,
        testy_n=testy_n,
        ano_x=ano_x,
        ano_y=ano_y
    )


def make_test_data(split_data, RNG, rate_anomaly_test):
    """make test data which has specified mixed rate(rate_anomaly_test).
    shuffle and concatenate normal and abnormal data"""

    ano_x = split_data.ano_x
    ano_y = split_data.ano_y
    testx_n = split_data.testx_n
    testy_n = split_data.testy_n

    # anomaly data in test
    inds_1 = RNG.permutation(ano_x.shape[0])
    ano_x = ano_x[inds_1]
    ano_y = ano_y[inds_1]

    index_1 = int(testx_n.shape[0] * rate_anomaly_test)
    testx_a = ano_x[:index_1]
    testy_a = ano_y[:index_1]

    # concatenate test normal data and test anomaly data
    testx = np.concatenate([testx_a, testx_n], axis=0)
    testy = np.concatenate([testy_a, testy_n], axis=0)

    return testx, testy


# def calc_metrics(testy, scores):
#     precision, recall, _ = precision_recall_curve(testy, scores)
#     roc_auc = roc_auc_score(testy, scores)
#     prc_auc = auc(recall, precision)

#     return roc_auc, prc_auc

def calc_metrics(testy, scores):
    return auroc(scores, testy), aupr(scores, testy), fpr_at_95_tpr(scores, testy), detection_error(scores, testy)


def anomaly_detection():
    # set parameters
    args = parse_args()
     
    normal_label = args.one_class_idx
    rate_normal_train = args.test_ratio
    rate_anomaly_test = args.rate_anomaly_test
    test_rep_count = args.test_rep_count
    TRAIN_RAND_SEED = args.TRAIN_RAND_SEED
    TEST_RAND_SEED = args.TEST_RAND_SEED

    data_type = args.selected_dataset
    data_path = './data/' + data_type + '_cae.npz' 
    # ##### fix random seeds for reproducibility ########
    SEED = args.seed = 20
    np.random.seed(SEED)
    #####################################################
    # load and prepare data
    full_data, full_labels = load_data(data_path)
    split_data = prepare_data(full_data, full_labels, normal_label, rate_normal_train, SEED)

   
    auroc_scores = []
    aupr_scores = []
    fpr_scores = []
    de_scores = []
    
    # nu : the upper limit ratio of anomaly data(0<=nu<=1)
    nus = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    # train model and evaluate with changing parameter nu
    for nu in nus:
        # train with nu
        clf = svm.OneClassSVM(nu=nu, kernel='rbf', gamma='auto')
        clf.fit(split_data.train_x)

        total_auroc  = 0
        total_aupr = 0
        total_fpr = 0
        total_de = 0


        seed_num = [20,40,60,80,100]
        # repeat test by randomly selected data and evaluate
        for SEED in seed_num:
            # select test data and test
            TEST_SEED = np.random.RandomState(SEED)

            testx, testy = make_test_data(split_data, TEST_SEED, rate_anomaly_test)
            scores = clf.decision_function(testx).ravel() * (-1)

            # calculate evaluation metrics
            auroc_rs, aupr_rs, fpr_at_95_tpr_rs, detection_error_rs = calc_metrics(testy, scores)

            total_auroc +=  auroc_rs
            total_aupr += aupr_rs            
            total_fpr +=  fpr_at_95_tpr_rs
            total_de +=  detection_error_rs 
            

        # calculate average
        total_auroc /= len(seed_num)
        total_aupr  /= len(seed_num)
        total_fpr   /= len(seed_num)
        total_de   /= len(seed_num)

       
        auroc_scores.append(total_auroc)
        aupr_scores.append(total_aupr)
        fpr_scores.append(total_fpr)
        de_scores.append(total_de)




        print('--- nu : ', nu, ' ---')
        print('ROC_AUC : ', total_auroc)
        print('AUPR_AUC : ', total_aupr)
        print('FPR_AUC : ', total_fpr)
        print('DE_AUC : ', total_de)

    print('***' * 5)
    print('ROC_AUC MAX : ', max(auroc_scores))
    print('AUPR_AUC MAX : ', max(aupr_scores))
    print('FPR_AUC MAX : ', max(fpr_scores))
    print('DE_AUC MAX : ', max(de_scores))

    print('ROC_MAX_NU : ', nus[int(np.argmax(auroc_scores))])


if __name__ == '__main__':
    anomaly_detection()