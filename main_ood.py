import os
import numpy as np
from datetime import datetime
import argparse
from utils import _logger
#from trainer.trainer import Trainer, model_evaluate
from trainer.trainer_OODness import Trainer, model_evaluate
from models.TC import TC
from utils import _calc_metrics
from models.TFC import TFC, target_classifier
from data_preprocessing.dataloader import count_label_labellist, select_transformation

import random, math
import torch
from torch.utils.data import DataLoader, Dataset
import torch.fft as fft

from data_preprocessing.dataloader import loading_data
from tsaug import *
from data_preprocessing.augmentations import select_transformation
from sklearn.model_selection import train_test_split
import pandas as pd

import pickle

class Load_Dataset(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, data_list, label_list, config, training_mode, aug_method):
        super(Load_Dataset, self).__init__()
        self.training_mode = training_mode

        X_train = data_list
        y_train = label_list

        if len(X_train.shape) < 3:
            X_train = X_train.unsqueeze(2)

        #if X_train.shape.index(min(X_train.shape)) != 1:  # make sure the Channels in second dim
        
        X_train = X_train.permute(0, 2, 1)
        # (N, C, T)
        if isinstance(X_train, np.ndarray):
            self.x_data = torch.from_numpy(X_train)
            self.y_data = torch.from_numpy(y_train).long()
        else:
            self.x_data = X_train
            self.y_data = y_train

        self.len = X_train.shape[0]
        
        pos_aug = select_transformation(aug_method)
        # (N, C, T) -> (N, T, C)-> (N, C, T)
        self.aug1 = torch.from_numpy(np.array(pos_aug.augment(
            self.x_data.permute(0, 2, 1).cpu().numpy()))).permute(0, 2, 1)

        # (N, C, T)
        self.aug1_f = fft.fftn(self.aug1).abs()     
        
        # normal_aug = select_transformation('Drift')
        # self.x_data = torch.from_numpy(np.array(normal_aug.augment(
        #     self.x_data.permute(0, 2, 1).cpu().numpy()))).permute(0, 2, 1)

        # (N, C, T)
        self.x_data_f = fft.fftn(self.x_data).abs() #/(window_length) # rfft for real value inputs.

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index], self.aug1[index], self.x_data_f[index], self.aug1_f[index]

    def __len__(self):
        return self.len

# Args selections
start_time = datetime.now()

parser = argparse.ArgumentParser()

######################## Model parameters ########################
home_dir = os.getcwd()
parser.add_argument('--experiment_description', default='Exp1', type=str,
                    help='Experiment Description')
parser.add_argument('--run_description', default='run1', type=str,
                    help='Experiment Description')
parser.add_argument('--seed', default=0, type=int,
                    help='seed value')
parser.add_argument('--training_mode',  default='T', 
                        help='choose the data augmentation wise : "T (Temporal), F (Frequency)" ')

parser.add_argument('--selected_dataset', default='lapras', type=str,
                    help='Dataset of choice: lapras, casas, opportunity, aras_a, aras_b')
parser.add_argument('--logs_save_dir', default='experiments_logs', type=str,
                    help='saving directory')
parser.add_argument('--device', default='cuda', type=str,
                    help='cpu or cuda')
parser.add_argument('--home_path', default=home_dir, type=str,
                    help='Project home directory')

parser.add_argument('--padding', type=str, 
                    default='mean', help='choose one of them : no, max, mean')
parser.add_argument('--timespan', type=int, 
                    default=1000, help='choose of the number of timespan between data points(1000 = 1sec, 60000 = 1min)')
parser.add_argument('--min_seq', type=int, 
                    default=10, help='choose of the minimum number of data points in a example')
parser.add_argument('--min_samples', type=int, default=20, 
                    help='choose of the minimum number of samples in each label')
parser.add_argument('--one_class_idx', type=int, default = -1, 
                    help='choose of one class label number that wants to deal with. -1 is for multi-classification')

parser.add_argument("--ood_score", help='score function for OOD detection',
                        default = ['norm_mean'], nargs="+", type=str)
parser.add_argument("--ood_samples", help='number of samples to compute OOD score',
                        default = 1, type=int)
parser.add_argument("--print_score", help='print quantiles of ood score',
                        action='store_true')
parser.add_argument("--ood_layer", help='layer for OOD scores',
                    choices = ['penultimate', 'simclr', 'shift'],
                    default = ['simclr', 'shift'], nargs="+", type=str)

parser.add_argument('--version', type=str, 
                    default='CL', help='choose of version want to do : ND or CL')
parser.add_argument('--print_freq', type=int, 
                    default=1, help='print frequency')
parser.add_argument('--save_freq', type=int, 
                    default=50, help='save frequency')
parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset')
    
parser.add_argument('--aug_method', type=str, default='AddNoise', help='choose the data augmentation method')
parser.add_argument('--aug_wise', type=str, default='Temporal', 
                        help='choose the data augmentation wise : "None, Temporal, Sensor" ')

parser.add_argument('--test_ratio', type=float, default=0.2, 
                    help='choose the number of test ratio')
parser.add_argument('--valid_ratio', type=float, default=0.1, 
                    help='choose the number of vlaidation ratio')
parser.add_argument('--overlapped_ratio', type=int, default= 50, 
                    help='choose the number of windows''overlapped ratio')

# for training   
parser.add_argument('--loss', type=str, default='SupCon', help='choose one of them: crossentropy loss, contrastive loss')
parser.add_argument('--optimizer', type=str, default='', help='choose one of them: adam')
parser.add_argument('--patience', type=int, default=20, 
                    help='choose the number of patience for early stopping')
parser.add_argument('--batch_size', type=int, default=128, 
                    help='choose the number of batch size')
parser.add_argument('--lr', type=float, default=3e-5, 
                    help='choose the number of learning rate')
parser.add_argument('--gamma', type=float, default=0.7, 
                    help='choose the number of gamma')
parser.add_argument('--temp', type=float, default=0.07, 
                    help='temperature for loss function')
parser.add_argument('--warm', action='store_true',
                    help='warm-up for large batch training')

args = parser.parse_args()

torch.cuda.empty_cache()

device = torch.device(args.device)
experiment_description = args.experiment_description
data_type = args.selected_dataset
method = 'Test OOD-ness'
training_mode = args.training_mode
run_description = args.run_description


if training_mode == 'T':
    store_path = 'result_files/final_ood_'+ data_type +'_T.xlsx'
    store_path_2 = 'result_files/Summary_ood_'+ data_type +'_T.xlsx'
elif training_mode == 'F':
    store_path = 'result_files/final_ood_'+ data_type +'_F.xlsx'
    store_path_2 = 'result_files/Summary_ood_'+ data_type +'_F.xlsx'
else:
    raise ValueError

logs_save_dir = args.logs_save_dir
os.makedirs(logs_save_dir, exist_ok=True)

exec(f'from config_files.{data_type}_Configs import Config as Configs')
configs = Configs()

if data_type == 'lapras': 
    args.timespan = 10000
    class_num = [ 0, 1, 2, 3,-1]

elif data_type == 'casas':         
    class_num = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, -1]
    args.aug_wise = 'Temporal2'
    
elif data_type == 'opportunity': 
    args.timespan = 1000
    class_num = [0, 1, 2, 3, 4, -1]

elif data_type == 'aras_a': 
    args.timespan = 1000
    #args.aug_wise = 'Temporal2'
    class_num = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, -1]


final_acc = []
final_f1  = []
final_auroc = []

num_classes, datalist, labellist = loading_data(data_type, args)

args.K_shift = 2
pos_ths = 0.6
neg_ths  = 0.9

clssfication_arr = []
strong_set = []
# Training for each class
for args.one_class_idx in class_num:

    temp_strong_set = []
    for positive_aug in ['AddNoise', 'Convolve', 'Crop', 'Drift', 'Dropout', 
                        'Pool', 'Quantize', 'Resize', 'Reverse', 'TimeWarp', 'AddNoise2']:
        acc_rs = []
        f1_rs  = []
        auroc_rs = []

        # Training for five seed
        for test_num in [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]:
            # ##### fix random seeds for reproducibility ########
            SEED = args.seed = test_num
            torch.manual_seed(SEED)
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = False
            np.random.seed(SEED)
            #####################################################

            experiment_log_dir = os.path.join(logs_save_dir, experiment_description, 
                                            run_description, training_mode + f"_seed_{SEED}")
            os.makedirs(experiment_log_dir, exist_ok=True)

            # loop through domains
            counter = 0
            src_counter = 0

            # Logging
            log_file_name = os.path.join(experiment_log_dir, f"logs_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log")
            logger = _logger(log_file_name)
            logger.debug("=" * 45)
            logger.debug(f'Dataset: {data_type}')
            logger.debug(f'Method:  {method}')
            logger.debug(f'Mode:    {training_mode}')
            logger.debug(f'Seed:    {SEED}')
            logger.debug(f'Positive Augmentation:    {positive_aug}')
            logger.debug(f'one idx:    {args.one_class_idx}')


            # Load datasets
            data_path = f"./data/{data_type}"            

            test_ratio = args.test_ratio
            valid_ratio = args.valid_ratio
            seed =  args.seed 

            # Split train and valid dataset
            train_list, test_list, train_label_list, test_label_list = train_test_split(datalist, 
                        labellist, test_size=test_ratio, stratify= labellist, random_state=seed) 
        
            train_list, valid_list, train_label_list, valid_label_list = train_test_split(train_list, 
                        train_label_list, test_size=valid_ratio, stratify=train_label_list, random_state=seed)                                       

                                        
            train_list = torch.tensor(train_list).cuda().cpu()
            train_label_list = torch.tensor(train_label_list).cuda().cpu()
            valid_list = torch.tensor(valid_list).cuda().cpu()
            valid_label_list = torch.tensor(valid_label_list).cuda().cpu()
            test_list = torch.tensor(test_list).cuda().cpu()
            test_label_list = torch.tensor(test_label_list).cuda().cpu()

            print(f"Train Data: {len(train_list)} --------------")
            exist_labels, _ = count_label_labellist(train_label_list)
            
            if args.one_class_idx != -1: # one-class
                sup_class_idx = [x - 1 for x in num_classes]
                known_class_idx = [args.one_class_idx]
                novel_class_idx = [item for item in sup_class_idx if item not in set(known_class_idx)]

            else: # multi-class
                sup_class_idx = [x for x in exist_labels]
                random.seed(args.seed)
                known_class_idx = random.sample(sup_class_idx, math.ceil(len(sup_class_idx)/2))
                #known_class_idx = [x for x in range(0, (int)(len(sup_class_idx)/2))]
                #known_class_idx = [0, 1]
                novel_class_idx = [item for item in sup_class_idx if item not in set(known_class_idx)]
                
            train_list = train_list[np.isin(train_label_list, known_class_idx)]
            train_label_list = train_label_list[np.isin(train_label_list, known_class_idx)]
            valid_list = valid_list[np.isin(valid_label_list, known_class_idx)]
            valid_label_list =valid_label_list[np.isin(valid_label_list, known_class_idx)]

                # only use for testing novelty
            test_list = test_list[np.isin(test_label_list,  known_class_idx)]
            test_label_list = test_label_list[np.isin(test_label_list,  known_class_idx)]    

            print(f"Validation Data: {len(valid_list)} --------------")    
            count_label_labellist(valid_label_list)

            print(f"Test Data: {len(test_list)} --------------")
            count_label_labellist(test_label_list) 

            logger.debug(f'known class:    {known_class_idx}')
            logger.debug("=" * 45)

            # Build data loader
            dataset = Load_Dataset(train_list, train_label_list, configs, training_mode, positive_aug)    
            train_dl = DataLoader(dataset, batch_size=configs.batch_size, shuffle=True)

            dataset = Load_Dataset(valid_list, valid_label_list, configs, training_mode, positive_aug)
            valid_dl = DataLoader(dataset, batch_size=configs.batch_size, shuffle=True)

            dataset = Load_Dataset(test_list, test_label_list, configs, training_mode, positive_aug)
            test_dl = DataLoader(dataset, batch_size=configs.batch_size, shuffle=True)
            
            
            logger.debug("Data loaded ...")

            # Load Model
            model = TFC(configs, args).to(device)
            classifier = target_classifier(configs).to(device)

            model_optimizer = torch.optim.Adam(model.parameters(), lr=configs.lr, 
                            betas=(configs.beta1, configs.beta2), weight_decay=3e-4)
            classifier_optimizer = torch.optim.Adam(classifier.parameters(), 
                            lr=configs.lr, betas=(configs.beta1, configs.beta2), weight_decay=3e-4)

            # Trainer
            model = Trainer(model, model_optimizer, classifier, classifier_optimizer, 
                            train_dl, valid_dl, test_dl, device, logger, configs, experiment_log_dir, training_mode)

            # evaluate on the test set
            logger.debug('\nEvaluate on the Test set:')
            outs = model_evaluate(model, classifier, test_dl, device, training_mode)
            total_loss, total_acc, total_f1, auroc, pred_labels, true_labels = outs
            logger.debug(f'Test Loss : {total_loss:.4f}\t | \tTest Accuracy : {total_acc:2.4f}\n'
                        f'Test F1 : {total_f1:.4f}\t | \tTest AUROC : {auroc:2.4f}')
            
            # Testing        
            _calc_metrics(pred_labels, true_labels, experiment_log_dir, args.home_path)
            acc_rs.append(total_acc.item())
            f1_rs.append(total_f1.item())
            auroc_rs.append(auroc.item())
    
        
        print("Average of the Accuracy list =", round(sum(acc_rs)/len(acc_rs), 3))
        print("Average of the F1 list =", round(sum(f1_rs)/len(f1_rs), 3))
        print("Average of the AUROC list =", round(sum(auroc_rs)/len(auroc_rs), 3))
        final_acc.append([np.mean(acc_rs), np.std(acc_rs)])
        final_f1.append([np.mean(f1_rs), np.std(f1_rs)])
        final_auroc.append([np.mean(auroc_rs), np.std(auroc_rs)])

        if np.mean(auroc_rs) > neg_ths: # for ST
            clssfication_arr.append([args.one_class_idx, str(positive_aug), 'N'])
            temp_strong_set.append(positive_aug)
        if np.mean(auroc_rs) < pos_ths: 
            clssfication_arr.append([args.one_class_idx, str(positive_aug), 'P'])
        print("Class_RS", clssfication_arr)
    
    strong_set.append(temp_strong_set)

# for extrating results to an excel file
final_rs =[]
for i in final_auroc:
    final_rs.append(i)
for i in final_acc:
    final_rs.append(i)
for i in final_f1:
    final_rs.append(i)



logger.debug(f"Training time is : {datetime.now()-start_time}")
print("Finished")

df = pd.DataFrame(final_rs, columns=['mean', 'std'])
df.to_excel(store_path, sheet_name='the results')

df2 = pd.DataFrame(clssfication_arr, columns=['Class_num', 'Augmentation', 'Pos/ Neg'])
df2.to_excel(store_path_2, sheet_name='the results')


if training_mode == 'T':
    with open('./data/'+data_type+'.data', 'wb') as f:
        pickle.dump(strong_set, f)
elif training_mode == 'F':
    with open('./data/'+data_type+'_f.data', 'wb') as f:
        pickle.dump(strong_set, f)

print(strong_set)

torch.cuda.empty_cache()