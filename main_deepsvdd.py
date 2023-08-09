import torch
import logging
import random
import numpy as np
import argparse
import math
import os

from deepSVDD import DeepSVDD
from data_preprocessing.dataloader import loading_data
from sklearn.model_selection import train_test_split
from data_preprocessing.dataloader import count_label_labellist
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import requests
import json

################################################################################
# Settings
################################################################################

class Load_Dataset(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, data_list, label_list):
        super(Load_Dataset, self).__init__()
        
        X_train = data_list
        y_train = label_list

        if len(X_train.shape) < 3:
            X_train = X_train.unsqueeze(2)

        # make sure the Channels in second dim
        X_train = np.transpose(X_train,(0, 2, 1))
        # (N, C, T)

        if isinstance(X_train, np.ndarray):
            self.x_data = torch.from_numpy(X_train)
            self.y_data = torch.from_numpy(y_train).long()
        else:
            self.x_data = X_train
            self.y_data = y_train

        self.len = X_train.shape[0]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index], index

    def __len__(self):
        return self.len


def send_slack_message(payload, webhook):

    return requests.post(webhook, json.dumps(payload))

def data_generator(args, configs, num_classes, datalist, labellist):
    test_ratio = args.test_ratio
    valid_ratio = args.valid_ratio
    seed =  args.seed 

    # Split train and valid dataset
    train_list, test_list, train_label_list, test_label_list = train_test_split(datalist, 
                                                                                labellist, test_size=test_ratio, stratify= labellist, random_state=seed) 
    if valid_ratio!=0:
        train_list, valid_list, train_label_list, valid_label_list = train_test_split(train_list, 
                                                                                      train_label_list, test_size=valid_ratio, stratify=train_label_list, random_state=seed)
    if valid_ratio == 0:
        valid_list = torch.Tensor(np.array([]))
        valid_label_list = torch.Tensor(np.array([]))

    print(f"Train Data: {len(train_list)} --------------")
    exist_labels, _ = count_label_labellist(train_label_list)
    
    print(f"Validation Data: {len(valid_list)} --------------")    
    count_label_labellist(valid_label_list)

    print(f"Test Data: {len(test_list)} --------------")
    count_label_labellist(test_label_list) 
    
    train_list = torch.tensor(train_list).cuda().cpu()
    train_label_list = torch.tensor(train_label_list).cuda().cpu()

    test_list = torch.tensor(test_list).cuda().cpu()
    test_label_list = torch.tensor(test_label_list).cuda().cpu()

 
    if(args.one_class_idx != -1): # one-class
        sup_class_idx = [x for x in exist_labels]
        known_class_idx = [args.one_class_idx]
        novel_class_idx = [item for item in sup_class_idx if item not in set(known_class_idx)]
        
        train_list = train_list[np.where(train_label_list == args.one_class_idx)]
        train_label_list = train_label_list[np.where(train_label_list == args.one_class_idx)]

        valid_list = test_list[np.where(test_label_list == args.one_class_idx)]
        valid_label_list = test_label_list[np.where(test_label_list == args.one_class_idx)]

        # only use for testing novelty
        test_list = test_list[np.where(test_label_list != args.one_class_idx)]
        test_label_list = test_label_list[np.where(test_label_list != args.one_class_idx)]

    else: # multi-class
        sup_class_idx = [x for x in exist_labels]
        random.seed(args.seed)
        known_class_idx = random.sample(sup_class_idx, math.ceil(len(sup_class_idx)/2))
        #known_class_idx = [x for x in range(0, (int)(len(sup_class_idx)/2))]
        #known_class_idx = [0, 1]
        novel_class_idx = [item for item in sup_class_idx if item not in set(known_class_idx)]
        
        train_list = train_list[np.isin(train_label_list, known_class_idx)]
        train_label_list = train_label_list[np.isin(train_label_list, known_class_idx)]
        valid_list = test_list[np.isin(test_label_list, known_class_idx)]
        valid_label_list =test_label_list[np.isin(test_label_list, known_class_idx)]

        # only use for testing novelty
        test_list = test_list[np.isin(test_label_list, novel_class_idx)]
        test_label_list = test_label_list[np.isin(test_label_list, novel_class_idx)]    


        # print(train_label_list)
        # print(valid_label_list)
        # print(test_label_list)
        
    train_label_list[:] = 0
    valid_label_list[:] = 0
    test_label_list[:] = 1
    # build data loader (N, T, C) -> (N, C, T)
    dataset = Load_Dataset(train_list, train_label_list)    
    train_loader = DataLoader(dataset, batch_size = configs.batch_size, shuffle=True)
    
    # dataset = Load_Dataset(train_list[math.ceil(len(train_list)/2):], train_label_list[math.ceil(len(train_list)/2):])    
    # train_loader2 = DataLoader(dataset, batch_size = configs.batch_size, shuffle=True)

    # dataset = Load_Dataset(valid_list,valid_label_list)
    # finetune_loader = DataLoader(dataset, batch_size = configs.batch_size, shuffle=True)

    # replace label : anomaly -> 1 : normal -> 0
    replace_list = np.concatenate((valid_list, test_list),axis=0)
    replace_label_list = np.concatenate((valid_label_list, test_label_list),axis=0)
    
    dataset = Load_Dataset(replace_list, replace_label_list)
    test_loader = DataLoader(dataset, batch_size=configs.batch_size, shuffle=True)
    
    
    
    # replace_list = np.concatenate((valid_list[math.ceil(len(valid_list)/2):], test_list[math.ceil(len(test_list)/2):]),axis=0)
    # replace_label_list = np.concatenate((valid_label_list[math.ceil(len(valid_list)/2):], test_label_list[math.ceil(len(test_list)/2):]),axis=0)
    
    # dataset = Load_Dataset(replace_list, replace_label_list)
    # test_loader2 = DataLoader(dataset, batch_size=configs.batch_size, shuffle=True)


    return train_loader, test_loader, novel_class_idx  


parser = argparse.ArgumentParser(description='DeepSVDD')

    
parser.add_argument('--xp_path', type=str)

parser.add_argument('--load_config', type=str, default=None,
                help='Config JSON-file path (default: None).')
parser.add_argument('--load_model', type=str, default=None,
                help='Model file path (default: None).')
parser.add_argument('--objective', type=str, default='one-class',
                help='Specify Deep SVDD objective ("one-class" or "soft-boundary").')
parser.add_argument('--nu', type=float, default=0.5, help='Deep SVDD hyperparameter nu (must be 0 < nu <= 1).')
parser.add_argument('--device', type=str, default='cuda', help='Computation device to use ("cpu", "cuda", "cuda:2", etc.).')
parser.add_argument('--seed', type=int, default=-1, help='Set seed. If -1, use randomization.')
parser.add_argument('--optimizer_name', type=str, default='adam',
                help='Name of the optimizer to use for Deep SVDD network training(adam, amsgrad).')
parser.add_argument('--lr', type=float, default=0.001,
                help='Initial learning rate for Deep SVDD network training. Default=0.001')
parser.add_argument('--n_epochs', type=int, default=100, help='Number of epochs to train.')

parser.add_argument('--lr_milestone', type=int, default=[], nargs='+',
                    help='Lr scheduler milestones at which lr is multiplied by 0.1. Can be multiple and must be increasing.')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size for mini-batch training.')
parser.add_argument('--weight_decay', type=float, default=1e-6,
                help='Weight decay (L2 penalty) hyperparameter for Deep SVDD objective.')
parser.add_argument('--pretrain', type=bool, default=True,
                help='Pretrain neural network parameters via autoencoder.')
parser.add_argument('--ae_optimizer_name', type=str, default='adam',
                help='Name of the optimizer to use for autoencoder pretraining(adam, amsgrad).')
parser.add_argument('--ae_lr', type=float, default=0.001,
                help='Initial learning rate for autoencoder pretraining. Default=0.001')
parser.add_argument('--ae_n_epochs', type=int, default=100, help='Number of epochs to train autoencoder.')
parser.add_argument('--ae_lr_milestone', type=int, default=[], nargs='+',
                    help='Lr scheduler milestones at which lr is multiplied by 0.1. Can be multiple and must be increasing.')
parser.add_argument('--ae_batch_size', type=int, default=128, help='Batch size for mini-batch autoencoder training.')
parser.add_argument('--ae_weight_decay', type=float, default=1e-6,
                help='Weight decay (L2 penalty) hyperparameter for autoencoder objective.')

parser.add_argument('--one_class_idx', type=int, default= 2, 
                    help='choose of one class label number that wants to deal with. -1 is for multi-classification')
   
parser.add_argument('--padding', type=str, 
                    default='mean', help='choose one of them : no, max, mean')
parser.add_argument('--timespan', type=int, 
                        default=10000, help='choose of the number of timespan between data points(1000 = 1sec, 60000 = 1min)')
parser.add_argument('--min_seq', type=int, 
                        default=10, help='choose of the minimum number of data points in a example')
parser.add_argument('--min_samples', type=int, default=20, 
                        help='choose of the minimum number of samples in each label')
parser.add_argument('--selected_dataset', default='lapras', type=str,
                        help='Dataset of choice: lapras, casas, opportunity, aras_a, aras_b')
parser.add_argument('--aug_method', type=str, default='AddNoise', help='choose the data augmentation method')
parser.add_argument('--aug_wise', type=str, default='Temporal', help='choose the data augmentation wise')

parser.add_argument('--test_ratio', type=float, default=0.2, help='choose the number of test ratio')
parser.add_argument('--valid_ratio', type=float, default=0.2, help='choose the number of test ratio')

args = parser.parse_args()

"""
    Deep SVDD, a fully deep method for anomaly detection.

    :arg DATASET_NAME: Name of the dataset to load.
    :arg NET_NAME: Name of the neural network to use.
    :arg XP_PATH: Export path for logging the experiment.
    :arg DATA_PATH: Root path of data.
"""
args = parser.parse_args()

torch.cuda.empty_cache()   
    
# Set seed
# ##### fix random seeds for reproducibility ########
SEED = args.seed = 40
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
#####################################################
    
data_type = args.selected_dataset
    
print(data_type)

exec(f'from config_files.{data_type}_Configs import Config as Configs')
configs = Configs()

if data_type == 'lapras': 
    args.timespan = 10000
    class_num = [0, 1, 2, 3, -1]
    seq_length = 598
    channel = 7
elif data_type == 'casas': 
    seq_length = 46
    args.aug_wise = 'Temporal2'
    class_num = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, -1]
    channel = 37
elif data_type == 'opportunity': 
    args.timespan = 1000
    class_num = [0, 1, 2, 3, 4, -1]
    seq_length = 169
    channel = 241
elif data_type == 'aras_a': 
    args.timespan = 1000
    seq_length = 63
    channel = 19
    class_num = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, -1]

final_auroc = []
final_aupr  = []
final_fpr   = []
final_de    = []


# Default device to 'cpu' if cuda is not available
if not torch.cuda.is_available(): device = 'cpu'
else: device ='cuda'
    

num_classes, datalist, labellist = loading_data(data_type, args)
    # Load data
    #dataset = load_dataset(dataset_name, data_path, normal_class)
for args.one_class_idx in class_num:
    auroc_a = []
    aupr_a  = []
    fpr_a   = []
    de_a    = []
    if args.one_class_idx != -1:
        seed_num =  [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
    else:
        seed_num = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]

    for test_num in seed_num :
        # ##### fix random seeds for reproducibility ########
        SEED = args.seed = test_num
        torch.manual_seed(SEED)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = False
        np.random.seed(SEED)            
        #####################################################

        print("Dataset:", data_type)
        print("True Class:", args.one_class_idx)
        print("Seed:", SEED)
        
        train_loader, test_loader, novel_class_idx = data_generator(args, configs, num_classes, datalist, labellist)


# Initialize DeepSVDD model and set neural network \phi
        deep_SVDD = DeepSVDD(args.objective, args.nu, seq_length, channel)

        #Train model on dataset
        deep_SVDD.train(train_loader, test_loader,
                            optimizer_name = args.optimizer_name,
                            lr = args.lr,
                            n_epochs=args.n_epochs,
                            lr_milestones=args.lr_milestone,
                            batch_size=args.batch_size,
                            weight_decay=args.weight_decay,
                            device=device)
        auroc_rs, aupr_rs, fpr_at_95_tpr_rs, detection_error_rs = deep_SVDD.test(test_loader, device=device)
        auroc_a.append(auroc_rs)     
        aupr_a.append(aupr_rs)   
        fpr_a.append(fpr_at_95_tpr_rs)
        de_a.append(detection_error_rs)

    final_auroc.append([np.mean(auroc_a), np.std(auroc_a)])
    final_aupr.append([np.mean(aupr_a), np.std(aupr_a)])
    final_fpr.append([np.mean(fpr_a), np.std(fpr_a)])
    final_de.append([np.mean(de_a), np.std(de_a)])

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
df.to_excel('result_files/DeepSVDD_'+data_type+'.xlsx', sheet_name='the results')

webhook = "https://hooks.slack.com/services/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"
payload = {"text": "Experiment_"+data_type+" Finished!"}
send_slack_message(payload, webhook)
    

# If specified, load Deep SVDD model (radius R, center c, network weights, and possibly autoencoder weights)
# if load_model:
#    deep_SVDD.load_model(model_path=load_model, load_ae=True)

# if args.pretrain:
#     # Pretrain model on dataset (via autoencoder)
#     deep_SVDD.pretrain(train_loader, test_loader,
#                         optimizer_name=args.ae_optimizer_name,
#                            lr=args.ae_lr,
#                            n_epochs=args.ae_n_epochs,
#                            lr_milestones= args.ae_lr_milestone,
#                            batch_size= args.ae_batch_size, 
#                            weight_decay=args.ae_weight_decay,
#                            device=device)


