import argparse
import augmentations as ts
import models.opt_tc as tc
import numpy as np
from data_preprocessing.dataloader import loading_data
from dataloader import data_generator_goad
#from data_loader import Data_Loader

def transform_data(data, trans):
    trans_inds = np.tile(np.arange(trans.n_transforms), len(data))
    print(data.shape)
    trans_data = trans.transform_batch(np.repeat(data.numpy(), trans.n_transforms, axis=0), trans_inds)
    return trans_data, trans_inds

def load_trans_data(args, trans):
    #dl = Data_Loader()
    _, datalist, labellist = loading_data(args.dataset, args)
    x_train, x_test, y_test = data_generator_goad(args, datalist, labellist)
    
    x_train_trans, _ = transform_data(x_train, trans)
    x_test_trans, _ = transform_data(x_test, trans)

    x_test_trans, x_train_trans = x_test_trans.transpose(0, 2, 1), x_train_trans.transpose(0, 2, 1)
    y_test = np.array(y_test) == args.one_class_idx
    return x_train_trans, x_test_trans, y_test


def train_anomaly_detector(args, config):
    transformer = ts.get_transformer(args, config)
    x_train, x_test, y_test = load_trans_data(args, transformer)
    print("final shape:",x_train.shape, x_test.shape, y_test.shape, transformer.n_transforms)
    # train 은 하나의 idx를 기반으로 하기 때문에 test의 크기가 더 클 수 있음
    tc_obj = tc.TransClassifier(transformer.n_transforms, args, configs)
    tc_obj.fit_trans_classifier(x_train, x_test, y_test)


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

    for i in range(4):
        args.one_class_idx = i
        print("Dataset:", args.dataset)
        print("True Class:", args.one_class_idx)
        train_anomaly_detector(args, configs)