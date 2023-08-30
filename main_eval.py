import itertools
import torch
import os
import numpy as np
from datetime import datetime
import argparse
from data_preprocessing.dataloader import loading_data
from utils import _logger
from trainer.trainer_ND import Trainer
from utils import _calc_metrics, copy_Files
from models.TFC import TFC, target_classifier
from dataloader import data_generator_nd
from eval_nd import eval_ood_detection
import pandas as pd
import openpyxl
from data_preprocessing.augmentations import select_transformation
from sklearn.preprocessing import LabelBinarizer
import random
import os.path

import requests
import json
import pickle

from util import calculate_acc_rv, visualization_roc, print_rs

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
                        default = ['NovelHD_TF'], nargs="+", type=str)
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
parser.add_argument('--lam_a', type=float, default= 0.5, help='choose lam_a ratio')
parser.add_argument('--train_num_ratio', type=float, default = 1, help='choose the number of test ratio')
parser.add_argument('--lam_score', type=float, default = 1, help='choose the number of test ratio')
parser.add_argument('--training_ver', type = str, default = 'Diverse', help='choose one of them: One, Diverse, Random')

# for training   
parser.add_argument('--loss', type=str, default='SupCon', help='choose one of them: crossentropy loss, contrastive loss')
parser.add_argument('--optimizer', type=str, default='', help='choose one of them: adam')
parser.add_argument('--patience', type=int, default=20, help='choose the number of patience for early stopping')
parser.add_argument('--batch_size', type=int, default=128, help='choose the number of batch size')
parser.add_argument('--lr', type=float, default=3e-5, help='choose the number of learning rate')
parser.add_argument('--gamma', type=float, default=0.7, help='choose the number of gamma')
parser.add_argument('--temp', type=float, default=0.07, help='temperature for loss function')
parser.add_argument('--warm', action='store_true',help='warm-up for large batch training')

args = parser.parse_args()


torch.cuda.empty_cache()

device = torch.device(args.device)
experiment_description = args.experiment_description
data_type = args.selected_dataset
method = 'Novel Human Activity'
training_mode = args.training_mode
run_description = args.run_description
positive_aug = 'AddNoise'

logs_save_dir = args.logs_save_dir
os.makedirs(logs_save_dir, exist_ok=True)


with open('./data/'+data_type+'.data', 'rb') as f:
    strong_set = pickle.load(f)

with open('./data/'+data_type+'_f.data', 'rb') as f:
    strong_set_f = pickle.load(f)

exec(f'from config_files.{data_type}_Configs import Config as Configs')
configs = Configs()

if data_type == 'lapras': 
    args.timespan = 10000
    class_num = configs.class_num
    weak_transformation = ['AddNoise']
elif data_type == 'casas':         
    class_num = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, -1]
    args.aug_wise = 'Temporal2'
    #strong_transformation = ['Convolve', 'Dropout', 'Drift', 'Crop', 'Pool', 'Quantize', 'Resize'] 
    strong_set = configs.ST
    strong_set_f = configs.ST_f
    weak_transformation = ['AddNoise']
elif data_type == 'opportunity': 
    args.timespan = 1000
    class_num = [0, 1, 2, 3, 4, -1]
    #strong_transformation = ['Convolve', 'Drift', 'Quantize', 'Pool', 'Crop'] 
    strong_set = configs.ST
    strong_set_f = configs.ST_f
  
    weak_transformation = ['AddNoise']
elif data_type == 'aras_a': 
    args.timespan = 1000
    #args.aug_wise = 'Temporal2'
    class_num = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, -1]
    strong_transformation = ['Convolve', 'Drift', 'Crop', 'Dropout', 'Pool', 'Quantize', 'Resize', 'TimeWarp'] 
    weak_transformation = ['AddNoise']

num_classes, datalist, labellist = loading_data(data_type, args)

# each mode ood_score == ['T'], ['TCON'], ['TCLS'], ['FCON'], ['FCLS'], ['NovelHD'], ['NovelHD_TF']
# ['T'],['NovelHD'], ['NovelHD_TF']
#for args.ood_score in [['T']]:  
# 

if args.training_ver == "Random":
    store_path = 'result_files/' + str(args.ood_score[0])+'_'+ \
                    data_type+'_random.xlsx'
    vis_path = 'figure/'+str(args.ood_score[0])+'_ROC_'+data_type+'_'+'_random.png'
    vis_title ="ROC curves of "+str(args.ood_score[0])+" - random ST"
elif args.training_ver == "One":
    store_path = 'result_files/' + str(args.ood_score[0])+'_'+ \
                    data_type+'_one.xlsx'
    vis_path = 'figure/'+str(args.ood_score[0])+'_ROC_'+data_type+'_'+'_one.png'
    vis_title ="ROC curves of "+str(args.ood_score[0])+" - one ST"

elif args.training_ver == "Diverse":
    store_path = 'result_files/' + str(args.ood_score[0])+'_'+ \
                    data_type+'_diverse.xlsx'
    vis_path = 'figure/'+str(args.ood_score[0])+'_ROC_'+data_type+'_'+'_diverse.png'
    vis_title ="ROC curves of "+str(args.ood_score[0])+" - diverse ST"

# slack
webhook = "https://hooks.slack.com/services/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"
payload = {"text": "Experiment "+store_path+" Finished!"}
#2 ->8, 6->7

visualization = True
args.binary = True


# if simclr(0,1)
#for args.K_shift in range(0,1):
# for one-T
#for args.K_shift in range(1,2):
# for multiple- T, lapras = 4,5, casas =8,9 opportunity = 6,7 , aras_a = 7,8
for args.K_shift in range(4,5):
    
    final_auroc = []
    final_aupr  = []
    final_fpr   = []
    final_de    = []  
    
    # for visualization
    y_onehot_test=[]
    y_score = []
    validation = []

#for num_lam_a in range(1, 10):
    #args.lam_a = round(num_lam_a * 0.1, 1)
    args.lam_a = 1
    #weak_num = args.K_pos = 10 - args.K_shift
   # weak_num = args.K_pos = 1
   # strong_num = args.K_shift
   # args.K_shift = args.K_shift + 1

    #if randomness:
    #    strong_transformation = ['AddNoise', 'AddNoise2','AddNoise3','AddNoise4', 'Convolve', 'Crop', 'Drift', 'Dropout', 'Pool', 'Quantize', 'Resize', 'Reverse', 'TimeWarp']
    if str(args.ood_score[0]) == "simclr":
        weak_transformation= [ 'Convolve', 'Crop', 'Drift', 'Dropout', 'Pool', 'Quantize', 'Resize', 'Reverse', 'TimeWarp']
    

    for num, args.one_class_idx in enumerate(class_num):

        
    # give weakly shifted transformation methods ['AddNoise', 'Convolve', 'Crop', 'Drift', 'Dropout', 'Pool', 'Quantize', 'Resize', 'Reverse', 'TimeWarp']
        for positive_aug in ['AddNoise']: #, 'Convolve', 'Crop', 'Drift', 'Dropout', 'Pool', 'Quantize', 'Resize', 'Reverse', 'TimeWarp']:
        #for shifted_aug in ['AddNoise', 'Convolve', 'Crop', 'Drift', 'Dropout', 'Pool', 'Quantize', 'Resize', 'Reverse', 'TimeWarp']:
                           
            # overall performance
            auroc_a = []
            aupr_a  = []
            fpr_a   = []
            de_a    = []

            testy_rs = []
            scores_rs = []


            # Training for five seed #
            for test_num in [20, 40, 60, 80, 100, 120, 140, 160, 180, 200] :
                # ##### fix random seeds for reproducibility ########
                SEED = args.seed = test_num
                torch.manual_seed(SEED)
                torch.backends.cudnn.deterministic = False
                torch.backends.cudnn.benchmark = False
                np.random.seed(SEED)
                random.seed(SEED)
                #####################################################
                positive_list  =['AddNoise']

                # for one ST
                if args.training_ver == "One":
                    negative_list = random.sample(strong_set[num], 1)
                    negative_list_f = random.sample(strong_set_f[num], 1)
                elif args.training_ver == "Random":
                    all_list = ['AddNoise','Convolve', 'Crop', 'Drift', 'Dropout', 'Pool', 'Quantize', 'Resize', 'Reverse', 'TimeWarp']
                    negative_list = random.sample(all_list, len(strong_set[num]))
                    negative_list_f = random.sample(all_list, len(strong_set_f[num]))
                else:
                    negative_list = strong_set[num]                
                    negative_list_f = strong_set_f[num]
            
                    

                # if strong_num > len(strong_transformation) :
                #     negative_list = [random.choice(strong_transformation) for i in range(strong_num - len(strong_transformation))]
                #     strong_num = len(strong_transformation)

                # negative_list += random.sample(strong_transformation, strong_num) #,'Dropout', 'Dropout', 'Dropout','Dropout']
                
                # if weak_num > len(weak_transformation):
                #     positive_list = [random.choice(weak_transformation) for i in range(weak_num- len(weak_transformation))]
                #     weak_num = len(weak_transformation)
                # positive_list += random.sample(weak_transformation, weak_num) #'AddNoise2'
               
                
                # Reset
                #strong_num = len (negative_list)
                #weak_num = len (positive_list) 


                args.K_shift = len(negative_list) + 1
                args.K_shift_f = len(negative_list_f) + 1
                args.K_pos = len (positive_list) 

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
                logger.debug(f'Seed:    {SEED}')
                logger.debug(f'Version:    {args.ood_score}')
                logger.debug(f'One_class_idx:    {args.one_class_idx}')
                logger.debug(f'Lambda A:    {args.lam_a}')
                logger.debug("=" * 45)

                # Load datasets
                data_path = f"./data/{data_type}"

                
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
                                train_dl, device, logger, configs, experiment_log_dir, args, negative_list, negative_list_f, positive_list)

                
                # load saved model of this experiment
                path = os.path.join(os.path.join(logs_save_dir, experiment_description, 
                                        run_description, f"novelty_detection_seed_{args.seed}", "saved_models"))
                    #chkpoint = torch.load(os.path.join(path, "ckp_last.pt"), map_location=device)
                    #pretrained_dict = chkpoint["model_state_dict"]
                    #model.load_state_dict(pretrained_dict)
                    
                    # Evlauation
                with torch.no_grad():
                    auroc_dict, aupr_dict, fpr_dict, de_dict, one_class_total, one_class_aupr, one_class_fpr, one_class_de, scores, labels\
                        = eval_ood_detection(args, path, model, valid_dl, 
                                             ood_test_loader, args.ood_score, train_dl, negative_list, negative_list_f)

                    auroc_a.append(one_class_total)     
                    aupr_a.append(one_class_aupr)   
                    fpr_a.append(one_class_fpr)
                    de_a.append(one_class_de)

                    testy_rs = testy_rs + labels
                    scores_rs = scores_rs + scores
                    

                    # mean_dict = dict()
                    # for ood_score in args.ood_score:
                    #     mean = 0
                    #     for ood in auroc_dict.keys():
                    #         mean += auroc_dict[ood][ood_score]
                    #     mean_dict[ood_score] = mean / len(auroc_dict.keys())
                    # auroc_dict['one_class_mean'] = mean_dict

                    # bests = []
                    # for ood in auroc_dict.keys():
                    #     print(ood)
                    #     message = ''
                    #     best_auroc = 0
                    #     for ood_score, auroc in auroc_dict[ood].items():
                    #         message += '[%s %s %.4f] ' % (ood, ood_score, auroc)
                    #         if auroc > best_auroc:
                    #             best_auroc = auroc
                    #     message += '[%s %s %.4f] ' % (ood, 'best', best_auroc)
                    #     if args.print_score:
                    #         print(message)
                    #     bests.append(best_auroc)

                    # bests = map('{:.4f}'.format, bests)
                    # print('\t'.join(bests))
                    # print("novel_class:", novel_class)

            
            #testy_rs, scores_rs = np.concatenate(testy_rs), np.concatenate(scores_rs)
            final_auroc.append([np.mean(auroc_a), np.std(auroc_a)])
            final_aupr.append([np.mean(aupr_a), np.std(aupr_a)])
            final_fpr.append([np.mean(fpr_a), np.std(fpr_a)])
            final_de.append([np.mean(de_a), np.std(de_a)])
            

            # for visualization
            onehot_encoded = list()        
            label_binarizer = LabelBinarizer().fit(testy_rs)                    
            onehot_encoded = label_binarizer.transform(testy_rs)
            y_onehot_test.append(onehot_encoded)
            y_score.append(scores_rs)

            auroc_rs, _,_,_ = calculate_acc_rv(testy_rs, scores_rs)
            validation.append([auroc_rs,0])

    print_rs(final_auroc, final_aupr, final_fpr, final_de, validation, store_path)

    if visualization:
    # visualization        
        visualization_roc(class_num, y_onehot_test, y_score, vis_title, vis_path)
            
print("Finished")

logger.debug(f"Training time is : {datetime.now()-start_time}")

# alert to slack
send_slack_message(payload, webhook)     

torch.cuda.empty_cache()

