import itertools
import torch
import os
import numpy as np
from datetime import datetime
import argparse
from data_preprocessing.dataloader import loading_data
from utils import _logger
from trainer.trainer_ND import Trainer

from models.TFC import target_classifier
from dataloader import data_generator_nd, data_generator_fold

import pandas as pd
from sklearn.preprocessing import LabelBinarizer
import random
import os.path

import requests
import json
import pickle

from eval_nd import eval_ood_detection
from util import calculate_acc_rv, visualization_roc, print_rs
from sklearn.model_selection import StratifiedKFold

def send_slack_message(payload, webhook):
    return requests.post(webhook, json.dumps(payload))

# Args selections
start_time = datetime.now()

parser = argparse.ArgumentParser()

######################## Model parameters ########################
home_dir = os.getcwd()
parser.add_argument('--experiment_description', default='exp1', type=str,
                    help='Experiment Description')
parser.add_argument('--run_description', default='run1', type=str,
                    help='Experiment Description')
parser.add_argument('--seed', default=0, type=int,
                    help='seed value')
parser.add_argument('--training_mode', default='supervised', type=str,
                    help='Modes of choice: random_init, supervised, self_supervised, fine_tune, train_linear')
parser.add_argument('--selected_dataset', default='lapras', type=str,
                    help='Dataset of choice: lapras, casas, opportunity, aras_a')
parser.add_argument('--logs_save_dir', default='experiments_logs', type=str,
                    help='saving directory')
parser.add_argument('--device', default='cuda', type=str,
                    help='cpu or cuda')
parser.add_argument('--home_path', default=home_dir, type=str,
                    help='Project home directory')

parser.add_argument('--padding', type=str, 
                    default='mean', help='choose one of them : no, max, mean')
parser.add_argument('--timespan', type=int, 
                    default=10000, help='choose of the number of timespan between data points (1000 = 1sec, 60000 = 1min)')
parser.add_argument('--min_seq', type=int, 
                    default=10, help='choose of the minimum number of data points in a example')
parser.add_argument('--min_samples', type=int, default=20, 
                    help='choose of the minimum number of samples in each label')
parser.add_argument('--one_class_idx', type=int, default=0, 
                    help='choose of one class label number that wants to deal with. -1 is for multi-classification')

parser.add_argument("--ood_score", help='score function for OOD detection',
                        default = ['NovelHD'], nargs="+", type=str)
parser.add_argument("--ood_samples", help='number of samples to compute OOD score',
                        default = 1, type=int)
parser.add_argument("--print_score", help='print quantiles of ood score',
                        action='store_true')
parser.add_argument("--ood_layer", help='layer for OOD scores',
                        choices = ['penultimate', 'simclr', 'shift'],
                        default = ['simclr', 'shift'], nargs="+", type=str)

parser.add_argument('--version', type=str, default='CL', help='choose of version want to do : ND or CL')
parser.add_argument('--print_freq', type=int, default=1, help='print frequency')
parser.add_argument('--save_freq', type=int, default=50, help='save frequency')
parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset')
    
parser.add_argument('--aug_method', type=str, default='AddNoise', help='choose the data augmentation method')
parser.add_argument('--K_shift', type=int, default=5 ,help='warm-up for large batch training')
parser.add_argument('--K_pos',  type=int, default=5,help='warm-up for large batch training')
parser.add_argument('--aug_wise', type=str, default='Temporal', 
                        help='choose the data augmentation wise : "None,  Temporal, Sensor" ')

parser.add_argument('--test_ratio', type=float, default=0.2, help='choose the number of test ratio')
parser.add_argument('--valid_ratio', type=float, default=0, help='choose the number of vlaidation ratio')
parser.add_argument('--overlapped_ratio', type=int, default= 50, help='choose the number of windows''overlapped ratio')
parser.add_argument('--lam_a', type=float, default= 1, help='choose lam_a ratio')
parser.add_argument('--train_num_ratio', type=float, default = 1, help='choose the number of test ratio')
parser.add_argument('--lam_score', type=float, default = 1, help='choose the number of test ratio')
parser.add_argument('--training_ver', type = str, default = 'Diverse', help='choose one of them: One, Diverse, Random')
parser.add_argument('--batch', type=int, default= 64, help='batch_size')
parser.add_argument('--neg_ths', type=float, default= 0.9, help='choose neg_thrshold ratio')

# for training   
parser.add_argument('--loss', type=str, default='SupCon', help='choose one of them: crossentropy loss, contrastive loss')
parser.add_argument('--optimizer', type=str, default='', help='choose one of them: adam')
#parser.add_argument('--lr', type=float, default=3e-5, help='choose the number of learning rate')
parser.add_argument('--temp', type=float, default=0.7, help='temperature for loss function')
parser.add_argument('--warm', action='store_true',help='warm-up for large batch training')

args = parser.parse_args()


torch.cuda.empty_cache()

device = torch.device(args.device)
experiment_description = args.experiment_description

method = 'Novel Human Activity'
training_mode = args.training_mode
run_description = args.run_description
positive_aug = 'AddNoise'

logs_save_dir = args.logs_save_dir
os.makedirs(logs_save_dir, exist_ok=True)

neg_ths  = args.neg_ths
data_type = args.selected_dataset

overall_st =[]
overall_st_f = []
with open('./data/'+data_type+'_s_'+str(neg_ths)+'.data', 'rb') as f:
    strong_set = pickle.load(f)

with open('./data/'+data_type+'_fs_'+str(neg_ths)+'.data', 'rb') as f:
    strong_set_f = pickle.load(f)

with open('./data/'+data_type+'_s_0.9.data', 'rb') as f:
    strong_set_90 = pickle.load(f)

with open('./data/'+data_type+'_fs_0.9.data', 'rb') as f:
    strong_set_f_90 = pickle.load(f)

for i in range(5, 10):
    with open('./data/'+data_type+'_s_'+str(round(i*0.1,1))+'.data', 'rb') as f:
        overall_st.append(pickle.load(f))
    with open('./data/'+data_type+'_fs_'+str(round(i*0.1,1))+'.data', 'rb') as f:
        overall_st_f.append(pickle.load(f))
for i in range(0, 4):
    for j in range(0, len(overall_st[i])):
        overall_st[i][j] = [x for x in overall_st[i][j] if x not in overall_st[i+1][j]]
        overall_st_f[i][j] = [x for x in overall_st_f[i][j] if x not in overall_st_f[i+1][j]]

#with open('./data/'+data_type+'_w._'+str(neg_ths)+'data', 'rb') as f:
#    weak_set = pickle.load(f)
    
#with open('./data/'+data_type+'_fw_'+str(neg_ths)+'.data', 'rb') as f:
#    weak_set_f = pickle.load(f)

with open('./data/'+data_type+'_multi._'+str(neg_ths)+'data', 'rb') as f:
    multiST = pickle.load(f)

with open('./data/'+data_type+'_multi_f_'+str(neg_ths)+'.data', 'rb') as f:
    multiST_f = pickle.load(f)

with open('./data/'+data_type+'_multi._0.9data', 'rb') as f:
    multiST_90 = pickle.load(f)

with open('./data/'+data_type+'_multi_f_0.9.data', 'rb') as f:
    multiST_f_90 = pickle.load(f)

overall_mul, overall_mul_f = [],[]
for i in range(5, 10):
    with open('./data/'+data_type+'_multi._'+str(round(i*0.1,1))+'data', 'rb') as f:
        overall_mul.append(pickle.load(f))
    with open('./data/'+data_type+'_multi_f_'+str(round(i*0.1,1))+'.data', 'rb') as f:
        overall_mul_f.append(pickle.load(f))
for i in range(0, 4):
    for j in range(0, len(overall_mul[i])):
        overall_mul[i][j] = [x for x in overall_mul[i][j] if x not in overall_mul[i+1][j]]
        overall_mul_f[i][j] = [x for x in overall_mul_f[i][j] if x not in overall_mul_f[i+1][j]]
# with open('./data/'+data_type+'_overall_'+str(neg_ths)+'.data', 'rb') as f:
#     overall_rs = pickle.load(f)

# with open('./data/'+data_type+'_overall_f_'+str(neg_ths)+'.data', 'rb') as f:
#     overall_rs_f = pickle.load(f)

args.data_size_ratio = 1
exec(f'from config_files.{data_type}_Configs import Config as Configs')
configs = Configs()

if data_type == 'lapras': 
    args.timespan = 10000
    class_num = [0, 1, 2, 3, -1]
    weak_transformation = ['AddNoise']
elif data_type == 'casas':         
    class_num = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, -1] 
    args.aug_wise = 'Temporal2'
    #strong_transformation = ['Convolve', 'Dropout', 'Drift', 'Crop', 'Pool', 'Quantize', 'Resize'] 
    weak_transformation = ['AddNoise']
elif data_type == 'opportunity': 
    args.timespan = 1000
    class_num = [0, 1, 2, 3, 4, -1]
    #strong_transformation = ['Convolve', 'Drift', 'Quantize', 'Pool', 'Crop'] 
    weak_transformation = ['AddNoise']
elif data_type == 'aras_a': 
    args.timespan = 1000
    #args.aug_wise = 'Temporal2'
    weak_transformation = ['AddNoise']
    class_num = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, -1]


num_classes, datalist, labellist = loading_data(data_type, args)

# each mode ood_score == ['T'], ['TCON'], ['TCLS'], ['FCON'], ['FCLS'], ['NovelHD'], ['NovelHD_TF']
# ['T'],['NovelHD'], ['NovelHD_TF']
#for args.ood_score in [['T']]:  

aug_num = 9
write_temp = False
write_lam = False
write_batch = False
use_fold = False
args.vis = False

if args.vis:
    from models.CLAN_atten import TFC 
else:
    from models.TFC import TFC 


if args.training_ver == "Random":
    store_path = 'result_files/' + str(args.ood_score[0])+'_'+ \
                    data_type+'_'+str(neg_ths)+'_random.xlsx'
    vis_path = 'figure/'+str(args.ood_score[0])+'_ROC_'+data_type+'_'+str(neg_ths)+'_random.png'
    vis_title ="ROC curves of "+str(args.ood_score[0])+" - random ST"
elif args.training_ver == "One":
    store_path = 'result_files/' + str(args.ood_score[0])+'_'+ \
                    data_type+'_'+str(neg_ths)+'_one.xlsx'
    vis_path = 'figure/'+str(args.ood_score[0])+'_ROC_'+data_type+'_'+str(neg_ths)+'_one.png'
    vis_title ="ROC curves of "+str(args.ood_score[0])+" - one ST"

elif args.training_ver == "Uniform":

    store_path = 'result_files/' + str(args.ood_score[0])+'_'+ \
                        data_type+'_'+str(neg_ths)+'_uniform.xlsx'
    vis_path = 'figure/'+str(args.ood_score[0])+'_ROC_'+data_type+'_'+str(neg_ths)+'_uniform.png'
    vis_title ="ROC curves of "+str(args.ood_score[0])+" - uniform ST"

elif args.training_ver == "Diverse":
    if write_temp:
        store_path = 'result_files/' + str(args.ood_score[0])+'_'+ \
                        data_type+'_'+str(neg_ths)+'_'+str(args.temp)+'_diverse.xlsx'
        vis_path = 'figure/'+str(args.ood_score[0])+'_ROC_'+data_type+'_'+str(neg_ths)+'_'+str(args.temp)+'_diverse.png'
        vis_title ="ROC curves of "+str(args.ood_score[0])+" - diverse ST"
    elif write_lam:
        store_path = 'result_files/' + str(args.ood_score[0])+'_'+ \
                        data_type+'_lam_'+str(args.lam_a)+'_diverse-v2.xlsx'
        vis_path = 'figure/'+str(args.ood_score[0])+'_ROC_'+data_type+'_lam_'+str(args.lam_a)+'_diverse-v2.png'
        vis_title ="ROC curves of "+str(args.ood_score[0])+" - diverse ST"
    elif write_batch:
        configs.batch_size = args.batch
        store_path = 'result_files/' + str(args.ood_score[0])+'_'+ \
                        data_type+'_batch_'+str(configs.batch_size)+'_diverse.xlsx'
        vis_path = 'figure/'+str(args.ood_score[0])+'_ROC_'+data_type+'_batch_'+str(configs.batch_size)+'_diverse.png'
        vis_title ="ROC curves of "+str(args.ood_score[0])+" - diverse ST"
    else:
        store_path = 'result_files/' + str(args.ood_score[0])+'_'+ \
                        data_type+'_'+str(neg_ths)+'_diverse.xlsx'
        vis_path = 'figure/'+str(args.ood_score[0])+'_ROC_'+data_type+'_'+str(neg_ths)+'_diverse.png'
        vis_title ="ROC curves of "+str(args.ood_score[0])+" - diverse ST"

elif args.training_ver == "Overall":
    store_path = 'result_files/' + str(args.ood_score[0])+'_'+ \
                    data_type+'_'+str(neg_ths)+'_overall.xlsx'
    vis_path = 'figure/'+str(args.ood_score[0])+'_ROC_'+data_type+'_'+str(neg_ths)+'_overall.png'
    vis_title ="ROC curves of "+str(args.ood_score[0])+" - same ST"
    
    with open('./data/'+data_type+'_overall_'+str(neg_ths)+'.data', 'rb') as f:
        overall_rs = pickle.load(f)

    with open('./data/'+data_type+'_overall_f_'+str(neg_ths)+'.data', 'rb') as f:
         overall_rs_f = pickle.load(f)

elif args.training_ver == "Num":
    store_path = 'result_files/' + str(args.ood_score[0])+'_'+ \
                    data_type+'_'+str(aug_num)+'_num.xlsx'
    vis_path = 'figure/'+str(args.ood_score[0])+'_ROC_'+data_type+'_'+str(aug_num)+'_num.png'
    vis_title ="ROC curves of "+str(args.ood_score[0])+" - "+str(aug_num)+" same ST"

else:
    raise ValueError


# # slack
# webhook = "https://hooks.slack.com/services/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"
# payload = {"text": "Experiment "+store_path+" Finished!"}
#2 ->8, 6->7

visualization = True
args.binary = True

# if simclr(0,1)
#for args.K_shift in range(0,1):
# for one-T
#for args.K_shift in range(1,2):
# for multiple- T, lapras = 4,5, casas =8,9 opportunity = 6,7 , aras_a = 7,8
kf = StratifiedKFold(n_splits=5)



# Starting

final_auroc = []
final_aupr  = []
final_fpr   = []
final_de    = []  
final_time = []

# for visualization
y_onehot_test=[]
y_score = []
validation = []

#for num_lam_a in range(1, 10):
#args.lam_a = round(num_lam_a * 0.1, 1)
#args.lam_a = 1
#weak_num = args.K_pos = 10 - args.K_shift
# weak_num = args.K_pos = 1
# strong_num = args.K_shift
# args.K_shift = args.K_shift + 1

#if randomness:
#    strong_transformation = ['AddNoise', 'AddNoise2','AddNoise3','AddNoise4', 'Convolve', 'Crop', 'Drift', 'Dropout', 'Pool', 'Quantize', 'Resize', 'Reverse', 'TimeWarp']


for args.one_class_idx in class_num:
    
    #num = 0        
# give weakly shifted transformation methods ['AddNoise', 'Convolve', 'Crop', 'Drift', 'Dropout', 'Pool', 'Quantize', 'Resize', 'Reverse', 'TimeWarp']
    #for S_tr in ['AddNoise', 'Convolve', 'Crop', 'Drift', 'Dropout', 'Pool', 'Quantize', 'Resize', 'Reverse', 'TimeWarp']:
    #for positive_aug in ['AddNoise']: #, 'Convolve', 'Crop', 'Drift', 'Dropout', 'Pool', 'Quantize', 'Resize', 'Reverse', 'TimeWarp']:
    #for shifted_aug in ['AddNoise', 'Convolve', 'Crop', 'Drift', 'Dropout', 'Pool', 'Quantize', 'Resize', 'Reverse', 'TimeWarp']:
                        
        # overall performance
    auroc_a = []
    aupr_a  = []
    fpr_a   = []
    de_a    = []
    time_a = []

    testy_rs = []
    scores_rs = []


    num = args.one_class_idx

    if args.one_class_idx != -1:
        if len(strong_set[num]) < 2:            
            with open('./data/'+data_type+'_s_'+str(neg_ths-0.1)+'.data', 'rb') as f:
                temp_strong_set = pickle.load(f)
                strong_set[num] = temp_strong_set[num]
        if len(strong_set_90[num]) < 2:
            with open('./data/'+data_type+'_s_'+str(0.8)+'.data', 'rb') as f:  
                temp_strong_set_90 = pickle.load(f)
                strong_set_90[num] = temp_strong_set_90[num]

        if len(strong_set_f[num]) < 2: 
            with open('./data/'+data_type+'_fs_'+str(neg_ths-0.1)+'.data', 'rb') as f:
                temp_strong_set_f = pickle.load(f) 
                strong_set_f[num] = temp_strong_set_f[num]

        if len(strong_set_f_90[num]) < 2:
            with open('./data/'+data_type+'_fs_'+str(0.8)+'.data', 'rb') as f:   
                temp_strong_set_f_90 = pickle.load(f)
                strong_set_f_90[num] = temp_strong_set_f_90[num]

    # if use_fold:                
    #     for i, (train_index, test_index) in enumerate(kf.split(datalist, labellist)):
    #         print(train_index, test_index)
    #         x_train, x_test = datalist[train_index], datalist[test_index]
    #         labellist = np.array(labellist)
    #         y_train, y_test = list(labellist[train_index]), list(labellist[test_index])
    #         train_dl, valid_dl, test_dl, ood_test_loader, novel_class = data_generator_fold(args, configs, training_mode, ['AddNoise'], 
    #                                             num_classes, x_train, x_test, y_train, y_test)
    # Training for five seed #
    for seed_n, test_num in enumerate([20, 40, 60, 80, 100, 120, 140, 160, 180, 200]) :
        # ##### fix random seeds for reproducibility ########
        SEED = args.seed = test_num #= seed_n = 20
        torch.manual_seed(SEED)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = False
        np.random.seed(SEED)
        random.seed(SEED)
        #####################################################
        positive_list  = ['AddNoise']

        # for one ST
        if args.training_ver == "One":
            negative_list = random.sample(strong_set[num], 1)
            negative_list_f  = random.sample(strong_set_f[num], 1) #= [S_tr]

            if args.one_class_idx == -1:
                negative_list = random.sample(multiST[seed_n], 1)
                negative_list_f = random.sample(multiST_f[seed_n], 1)

        elif args.training_ver == "Num":
            all_list = ['Convolve', 'Crop', 'Drift', 'Dropout', 'Pool', 'Quantize', 'Resize', 'Reverse', 'TimeWarp']
            negative_list = random.sample(all_list, aug_num)
            negative_list_f  = random.sample(all_list, aug_num) #= [S_tr]

            if args.one_class_idx == -1:
                negative_list = random.sample(all_list, aug_num)
                negative_list_f = random.sample(all_list, aug_num)

        elif args.training_ver == "Random":
            #assume 90
            all_list = ['AddNoise','Convolve', 'Crop', 'Drift', 'Dropout', 'Pool', 'Quantize', 'Resize', 'Reverse', 'TimeWarp']
            negative_list = random.sample(all_list, min(len(strong_set[num]),len(strong_set_90[num])))
            negative_list_f = random.sample(all_list, min(len(strong_set_f[num]),len(strong_set_f_90[num])))

            if args.one_class_idx == -1:
                if len(multiST_f_90[seed_n]) <2:
                    with open('./data/'+data_type+'_multi_f_'+str(0.8)+'.data', 'rb') as f:
                        temp_multiST_f_90 = pickle.load(f)
                        multiST_f_90[seed_n] = temp_multiST_f_90[seed_n]

                if len(multiST_f[seed_n]) <2:    
                    with open('./data/'+data_type+'_multi_f_'+str(0.8)+'.data', 'rb') as f:
                        temp_multiST_f = pickle.load(f)
                        multiST_f[seed_n] =  temp_multiST_f[seed_n]

                if len(multiST_90[seed_n]) < 2:
                    with open('./data/'+data_type+'_multi._'+str(0.8)+'data', 'rb') as f:
                        temp_multiST_90 = pickle.load(f)
                        multiST_90[seed_n] = temp_multiST_90[seed_n]

                if len(multiST[seed_n]) < 2:        
                    with open('./data/'+data_type+'_multi._'+str(0.8)+'data', 'rb') as f:
                        temp_multiST= pickle.load(f)
                        multiST[seed_n] = temp_multiST[seed_n]

                negative_list = random.sample(all_list, min(len(multiST[seed_n]), len(multiST_90[seed_n])))
                negative_list_f = random.sample(all_list, min(len(multiST_f[seed_n]), len(multiST_f_90[seed_n])))

        elif args.training_ver == "Uniform":
            uniform_exm = ['Resize', 'Resize2', 'Resize3','Resize4','Resize5','Resize6','Resize7','Resize8','Resize9','Resize10']
            negative_list = random.sample(uniform_exm, min((int)(len(strong_set[num])),len(strong_set_90[num])))
            negative_list_f = random.sample(uniform_exm, min((int)(len(strong_set_f[num])),len(strong_set_f_90[num])))
                
            if args.one_class_idx == -1:
                if len(multiST_f_90[seed_n]) <2:
                    with open('./data/'+data_type+'_multi_f_'+str(0.8)+'.data', 'rb') as f:
                        temp_multiST_f_90 = pickle.load(f)
                        multiST_f_90[seed_n] = temp_multiST_f_90[seed_n]

                if len(multiST_f[seed_n]) <2:    
                    with open('./data/'+data_type+'_multi_f_'+str(0.8)+'.data', 'rb') as f:
                        temp_multiST_f = pickle.load(f)
                        multiST_f[seed_n] =  temp_multiST_f[seed_n]

                if len(multiST_90[seed_n]) < 2:
                    with open('./data/'+data_type+'_multi._'+str(0.8)+'data', 'rb') as f:
                        temp_multiST_90 = pickle.load(f)
                        multiST_90[seed_n] = temp_multiST_90[seed_n]

                if len(multiST[seed_n]) < 2:        
                    with open('./data/'+data_type+'_multi._'+str(0.8)+'data', 'rb') as f:
                        temp_multiST= pickle.load(f)
                        multiST[seed_n] = temp_multiST[seed_n]

                negative_list = random.sample(uniform_exm, min((int)(len(multiST[seed_n])), len(multiST_90[seed_n])))
                negative_list_f = random.sample(uniform_exm, min((int)(len(multiST_f[seed_n])), len(multiST_f_90[seed_n])))
            

        elif args.training_ver == "Diverse":
            # negative_list = strong_set[num]
            # negative_list_f = strong_set_f[num]

            # if args.one_class_idx == -1:
            #     negative_list = multiST[seed_n]
            #     negative_list_f = multiST_f[seed_n]
            # # min for threshold test
            # for test
            if args.one_class_idx != -1:
                # if neg_ths != 0.9:
                #     result_list = overall_st[(int)(round(neg_ths-0.5, 1)*10)][num]
                #     result_list_f =overall_st_f[(int)(round(neg_ths-0.5, 1)*10)][num]

                #     if len(result_list) >= len(strong_set_90[num]):
                #         strong_set[num] = result_list 
                #     else:
                #         temp_ajent = []                           
                #         num_neg, num_neg_2 = 1, 1
                #         while len(temp_ajent) < (len(strong_set_90[num])-len(result_list)):
                #             print(neg_ths, round(neg_ths-0.5, 1)*10, num_neg, result_list, temp_ajent,strong_set_90[num])
                #             if (round(neg_ths-0.5, 1)*10+num_neg) < 5:
                #                 temp_ajent = temp_ajent + overall_st[(int)(round(neg_ths-0.5, 1)*10+num_neg)][num]
                #                 num_neg = num_neg+1

                #             if (round(neg_ths-0.5, 1)*10-num_neg_2) >= 0:
                #                 temp_ajent = temp_ajent + overall_st[(int)(round(neg_ths-0.5, 1)*10-num_neg_2)][num]
                #                 num_neg_2 = num_neg_2 + 1
                #             print(neg_ths, round(neg_ths-0.5, 1)*10, num_neg, result_list, temp_ajent,strong_set_90[num])

                #         print(neg_ths, round(neg_ths-0.5, 1)*10, num_neg, result_list, temp_ajent,strong_set_90[num])
                #         strong_set[num] = result_list + random.sample(temp_ajent, len(strong_set_90[num])-len(result_list))
                    
                #     if len(result_list_f) >= len(strong_set_f_90[num]):
                #         strong_set_f[num] = result_list_f
                #     else:
                #         temp_ajent = []                            
                #         num_neg, num_neg_2 = 1, 1
                #         while len(temp_ajent) < (len(strong_set_f_90[num])-len(result_list_f)):
                #             print(neg_ths, round(neg_ths-0.5, 1)*10, num_neg, temp_ajent,strong_set_f_90[num])
                #             if (round(neg_ths-0.5, 1)*10+num_neg) < 5:
                #                 temp_ajent = temp_ajent + overall_st_f[(int)(round(neg_ths-0.5, 1)*10+num_neg)][num]
                #                 num_neg = num_neg+1
                #             if (round(neg_ths-0.5, 1)*10-num_neg_2) >= 0:
                #                 temp_ajent = temp_ajent + overall_st_f[(int)(round(neg_ths-0.5, 1)*10-num_neg_2)][num]
                #                 num_neg_2 = num_neg_2 + 1
                #         print(neg_ths, round(neg_ths-0.5, 1)*10, num_neg, result_list_f, temp_ajent,strong_set_f_90[num])
                #         strong_set_f[num] = result_list_f + random.sample(temp_ajent, len(strong_set_f_90[num])-len(result_list_f))

                negative_list = random.sample(strong_set[num], min((int)(len(strong_set[num])),len(strong_set_90[num])))
                negative_list_f = random.sample(strong_set_f[num], min((int)(len(strong_set_f[num])),len(strong_set_f_90[num])))
                
                # negative_list = strong_set[num]
                # negative_list_f = strong_set_f[num]

            elif args.one_class_idx == -1:
                if len(multiST_f_90[seed_n]) <2:
                    with open('./data/'+data_type+'_multi_f_'+str(0.8)+'.data', 'rb') as f:
                        temp_multiST_f_90 = pickle.load(f)
                        multiST_f_90[seed_n] = temp_multiST_f_90[seed_n]

                if len(multiST_f[seed_n]) <2:    
                    with open('./data/'+data_type+'_multi_f_'+str(0.8)+'.data', 'rb') as f:
                        temp_multiST_f = pickle.load(f)
                        multiST_f[seed_n] =  temp_multiST_f[seed_n]

                if len(multiST_90[seed_n]) < 2:
                    with open('./data/'+data_type+'_multi._'+str(0.8)+'data', 'rb') as f:
                        temp_multiST_90 = pickle.load(f)
                        multiST_90[seed_n] = temp_multiST_90[seed_n]

                if len(multiST[seed_n]) < 2:        
                    with open('./data/'+data_type+'_multi._'+str(0.8)+'data', 'rb') as f:
                        temp_multiST= pickle.load(f)
                        multiST[seed_n] = temp_multiST[seed_n]
                
                # if neg_ths != 0.9:
                #     result_list = overall_mul[(int)(round(neg_ths-0.5, 1)*10)][seed_n]
                #     result_list_f =overall_mul_f[(int)(round(neg_ths-0.5, 1)*10)][seed_n]

                #     if len(result_list) >= len(multiST_90[seed_n]):
                #         multiST[seed_n] = result_list 
                #     else:
                #         temp_ajent = []                            
                #         num_neg, num_neg_2 = 1, 1
                #         while len(temp_ajent) < (len(multiST_90[seed_n])-len(result_list)):
                #             print(neg_ths, round(neg_ths-0.5, 1)*10, num_neg, temp_ajent,multiST_90[seed_n])

                #             if (round(neg_ths-0.5, 1)*10+num_neg) < 5:
                #                 temp_ajent = temp_ajent + overall_mul[(int)(round(neg_ths-0.5, 1)*10+num_neg)][seed_n]
                #                 num_neg = num_neg+1
                #             if (round(neg_ths-0.5, 1)*10-num_neg_2) >= 0:
                #                 temp_ajent = temp_ajent + overall_mul[(int)(round(neg_ths-0.5, 1)*10-num_neg_2)][seed_n]
                #                 num_neg_2 = num_neg_2 + 1
                #         print(neg_ths, round(neg_ths-0.5, 1)*10, num_neg, result_list, temp_ajent, multiST_90[seed_n])
                #         multiST[seed_n] = result_list + random.sample(temp_ajent, len(multiST_90[seed_n])-len(result_list))
                    
                #     if len(result_list_f) >= len(multiST_f_90[seed_n]):
                #         multiST_f[seed_n] = result_list_f
                #     else:
                #         temp_ajent = []                            
                #         num_neg, num_neg_2 = 1, 1
                #         while len(temp_ajent) < (len(multiST_f_90[seed_n])-len(result_list_f)):
                #             print(neg_ths, round(neg_ths-0.5, 1)*10, num_neg, temp_ajent,multiST_f_90[seed_n])
                #             if (round(neg_ths-0.5, 1)*10+num_neg) < 5:
                #                 temp_ajent = temp_ajent + overall_mul_f[(int)(round(neg_ths-0.5, 1)*10+num_neg)][seed_n]
                #                 num_neg = num_neg+1
                #             if (round(neg_ths-0.5, 1)*10-num_neg_2) >= 0:
                #                 temp_ajent = temp_ajent + overall_mul_f[(int)(round(neg_ths-0.5, 1)*10-num_neg_2)][seed_n]
                #                 num_neg_2 = num_neg_2 + 1
                #         print(neg_ths, round(neg_ths-0.5, 1)*10, num_neg, result_list_f, temp_ajent,multiST_f_90[seed_n])
                #         multiST_f[seed_n] = result_list_f + random.sample(temp_ajent, len(multiST_f_90[seed_n])-len(result_list_f))

                negative_list = random.sample(multiST[seed_n], min((int)(len(multiST[seed_n])), len(multiST_90[seed_n])))
                negative_list_f = random.sample(multiST_f[seed_n], min((int)(len(multiST_f[seed_n])), len(multiST_f_90[seed_n])))
                # negative_list = multiST[seed_n]
                # negative_list_f = multiST_f[seed_n]
            
        elif args.training_ver == "Overall":
            if args.one_class_idx == -1:

                if len(multiST_f_90[seed_n]) <2:
                    with open('./data/'+data_type+'_multi_f_'+str(0.8)+'.data', 'rb') as f:
                        temp_multiST_f_90 = pickle.load(f)
                        multiST_f_90[seed_n] = temp_multiST_f_90[seed_n]

                if len(multiST_f[seed_n]) <2:    
                    with open('./data/'+data_type+'_multi_f_'+str(neg_ths-0.1)+'.data', 'rb') as f:
                        temp_multiST_f = pickle.load(f)
                        multiST_f[seed_n] =  temp_multiST_f[seed_n]

                if len(multiST_90[seed_n]) < 2:
                    with open('./data/'+data_type+'_multi._'+str(0.8)+'data', 'rb') as f:
                        temp_multiST_90 = pickle.load(f)
                        multiST_90[seed_n] = temp_multiST_90[seed_n]

                if len(multiST[seed_n]) < 2:        
                    with open('./data/'+data_type+'_multi._'+str(neg_ths-0.1)+'data', 'rb') as f:
                        temp_multiST= pickle.load(f)
                        multiST[seed_n] = temp_multiST[seed_n]


                if len(strong_set[num]) >= len(multiST[seed_n]): 
                    negative_list = random.sample(strong_set[num], len(multiST[seed_n]))
                else:
                    dif = len(multiST[seed_n])-len(strong_set[num]) 
                    negative_list  = strong_set[num]
                    negative_list = negative_list + random.sample(multiST[seed_n], diff)
                
                if len(strong_set_f[num]) >= len(multiST_f[seed_n]): 
                    negative_list_f = random.sample(strong_set_f[num], len(multiST_f[seed_n]))
                else:
                    dif = len(multiST_f[seed_n])-len(strong_set_f[num]) 
                    negative_list_f  = strong_set_f[num]
                    negative_list_f = negative_list_f + random.sample(multiST_f[seed_n], diff)
            else:
                if len(overall_rs) >= len(strong_set[num]):                    
                    negative_list = random.sample(overall_rs, len(strong_set[num]))
                else:
                    diff = len(strong_set[num]) - len(overall_rs)
                    negative_list = overall_rs
                    negative_list = negative_list + random.sample(strong_set[num], diff)

                if len(overall_rs_f) >= len(strong_set_f[num]):                    
                    negative_list_f = random.sample(overall_rs_f, len(strong_set_f[num]))
                else:
                    diff = len(strong_set_f[num]) - len(overall_rs_f)
                    negative_list_f = overall_rs_f
                    negative_list_f = negative_list_f+ random.sample(strong_set_f[num], diff)
                                    
        else:
            raise ValueError
        
        positive_list = ['AddNoise'] #weak_set [num]
        positive_list_f = ['AddNoise'] #weak_set_f[num]          

        if str(args.ood_score[0]) == "simclr":
            weak_transformation= [ 'Convolve', 'Crop', 'Drift', 'Dropout', 'Pool', 'Quantize', 'Resize', 'Reverse', 'TimeWarp']
            positive_list = random.sample(weak_transformation, 1)  
            negative_list = []


        args.K_shift = len(negative_list) + 1
        args.K_shift_f = len(negative_list_f) + 1
        args.K_pos = len (positive_list) 
        args.K_pos_f = len (positive_list_f) 

        experiment_log_dir = os.path.join(logs_save_dir, experiment_description, run_description, training_mode + f"_seed_{SEED}")
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
        logger.debug(f'Positive Augmentation:    {positive_list}')
        logger.debug(f'Negative Augmentation:    {negative_list}')
        logger.debug(f'Positive_F Augmentation:    {positive_list_f}')
        logger.debug(f'Negative_F Augmentation:    {negative_list_f}')
        logger.debug(f'Seed:    {SEED}')
        logger.debug(f'Version:    {args.ood_score}')
        logger.debug(f'One_class_idx:    {args.one_class_idx}')
        logger.debug(f'Lambda:    {args.lam_a}')
        logger.debug(f'Neg ths:    {neg_ths}')
        logger.debug(f'Temperature:    {args.temp}')
        logger.debug(f'Batch:    {configs.batch_size}')
        logger.debug("=" * 45)

        # Load datasets
        data_path = f"./data/{data_type}"

        if use_fold != True:    
            train_dl, valid_dl, test_dl, ood_test_loader, novel_class = data_generator_nd(
                args, configs, training_mode, positive_list, num_classes, datalist, labellist)
        logger.debug("Data loaded ...")

        # Load Model
        model = TFC(configs, args).to(device)
        classifier = target_classifier(configs).to(device)
        model_optimizer = torch.optim.Adam(model.parameters(), 
                                            lr=configs.lr, betas=(configs.beta1, configs.beta2), weight_decay=3e-4)
        classifier_optimizer = torch.optim.Adam(classifier.parameters(), 
                                                lr=configs.lr, betas=(configs.beta1, configs.beta2), weight_decay=3e-4)

        #if training_mode == "self_supervised" and "novelty_detection":  # to do it only once
        #    copy_Files(os.path.join(logs_save_dir, experiment_description, run_description), data_type)

        # Trainer
        model = Trainer(model, model_optimizer, classifier, classifier_optimizer, 
                            train_dl, device, logger, configs, experiment_log_dir, args, negative_list, negative_list_f, positive_list, positive_list_f)
            
        # load saved model of this experiment
        path = os.path.join(os.path.join(logs_save_dir, experiment_description, 
                                    run_description, f"novelty_detection_seed_{args.seed}", "saved_models"))
                #chkpoint = torch.load(os.path.join(path, "ckp_last.pt"), map_location=device)
                #pretrained_dict = chkpoint["model_state_dict"]
                #model.load_state_dict(pretrained_dict)
                
                # Evlauation
        with torch.no_grad():
            auroc_dict, aupr_dict, fpr_dict, de_dict, one_class_total, one_class_aupr, one_class_fpr, one_class_de, scores, labels, infer_time\
                = eval_ood_detection(args, path, model, valid_dl, 
                                            ood_test_loader, args.ood_score, train_dl, negative_list, negative_list_f)

        auroc_a.append(one_class_total)     
        aupr_a.append(one_class_aupr)   
        fpr_a.append(one_class_fpr)
        de_a.append(one_class_de)
        time_a.append(infer_time)

        testy_rs = testy_rs + labels
        scores_rs = scores_rs + scores
                
        # final_auroc.append([one_class_total,0])
        # final_aupr.append([one_class_aupr,0])
        # final_fpr.append([one_class_fpr,0])
        # final_de.append([one_class_de,0])

        
    #testy_rs, scores_rs = np.concatenate(testy_rs), np.concatenate(scores_rs)
    final_auroc.append([np.mean(auroc_a),np.std(auroc_a)])
    final_aupr.append([np.mean(aupr_a),np.std(aupr_a)])
    final_fpr.append([np.mean(fpr_a),np.std(fpr_a)])
    final_de.append([np.mean(de_a),np.std(de_a)])
    final_time.append([np.mean(time_a),np.std(time_a)])            

    # for visualization
    onehot_encoded = list()        
    label_binarizer = LabelBinarizer().fit(testy_rs)                    
    onehot_encoded = label_binarizer.transform(testy_rs)
    y_onehot_test.append(onehot_encoded)
    y_score.append(scores_rs)

    auroc_rs, _,_,_ = calculate_acc_rv(testy_rs, scores_rs)
    validation.append([auroc_rs,0])

print_rs(final_auroc, final_aupr, final_fpr, final_de, final_time, store_path)

if visualization:
# visualization        
    visualization_roc(class_num, y_onehot_test, y_score, vis_title, vis_path)
        
print("Finished")

# # alert to slack
# send_slack_message(payload, webhook)     

print(final_time)

torch.cuda.empty_cache()

