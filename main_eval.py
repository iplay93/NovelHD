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
parser.add_argument('--aug_wise', type=str, default='Temporal', 
                        help='choose the data augmentation wise : "None,  Temporal, Sensor" ')

parser.add_argument('--test_ratio', type=float, default=0.2, help='choose the number of test ratio')
parser.add_argument('--valid_ratio', type=float, default=0, help='choose the number of vlaidation ratio')
parser.add_argument('--overlapped_ratio', type=int, default= 50, help='choose the number of windows''overlapped ratio')

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

exec(f'from config_files.{data_type}_Configs import Config as Configs')
configs = Configs()

if data_type == 'lapras': args.timespan = 10000
elif data_type == 'opportunity': args.timespan = 1000
elif data_type == 'aras_a': args.timespan = 10000
elif data_type == 'aras_b': args.timespan = 10000

num_classes, datalist, labellist = loading_data(data_type, args)

# each mode ood_score == ['T'], ['TCON'], ['TCLS'], ['FCON'], ['FCLS'], ['NovelHD'], ['NovelHD_TF']
# ['T'],['NovelHD'], ['NovelHD_TF']
for args.ood_score in [['T'], ['NovelHD'],['NovelHD_TF']]:    
        
    final_auroc = []
    final_aupr  = []
    final_fpr   = []
    final_de    = []

    if data_type == 'lapras': class_num = [0,1,2,3,-1]
    elif data_type == 'casas': 
        class_num = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, -1]
        args.aug_wise = 'Temporal2'
    elif data_type == 'opportunity': class_num = [0, 1, 2, 3, 4, -1]
    elif data_type == 'aras_a': class_num = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, -1]

    # lapras : [0, 1, 2, 3, -1]
    # casas : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, -1]
    # aras_a : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, -1]
    # opportunity : [0, 1, 2, 3, 4, -1]
    
    for args.one_class_idx in class_num:
    # give weakly shifted transformation methods ['AddNoise', 'Convolve', 'Crop', 'Drift', 'Dropout', 'Pool', 'Quantize', 'Resize', 'Reverse', 'TimeWarp']
        for positive_aug in ['TimeWarp']: #, 'Convolve', 'Crop', 'Drift', 'Dropout', 'Pool', 'Quantize', 'Resize', 'Reverse', 'TimeWarp']:
        #for shifted_aug in ['AddNoise', 'Convolve', 'Crop', 'Drift', 'Dropout', 'Pool', 'Quantize', 'Resize', 'Reverse', 'TimeWarp']:
            # overall performance
            auroc_a = []
            aupr_a  = []
            fpr_a   = []
            de_a    = []
            
            # give strongly shifted transformation
            shifted_aug = 'Dropout'
            
            if args.one_class_idx != -1:
                seed_num = [20, 40, 60, 80, 100]
            else:
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
                logger.debug(f'Positive Augmentation:    {positive_aug}')
                logger.debug(f'Negative Augmentation:    {shifted_aug}')
                logger.debug(f'Seed:    {SEED}')
                logger.debug(f'Version:    {args.ood_score}')
                logger.debug(f'One_class_idx:    {args.one_class_idx}')
                logger.debug("=" * 45)

                # Load datasets
                data_path = f"./data/{data_type}"

                
                train_dl, valid_dl, test_dl, ood_test_loader, novel_class = data_generator_nd(
                    args, configs, training_mode, positive_aug, num_classes, datalist, labellist)
                logger.debug("Data loaded ...")

                # Load Model
                model = TFC(configs).to(device)
                classifier = target_classifier(configs).to(device)

                model_optimizer = torch.optim.Adam(model.parameters(), 
                                                lr=configs.lr, betas=(configs.beta1, configs.beta2), weight_decay=3e-4)
                classifier_optimizer = torch.optim.Adam(classifier.parameters(), 
                                                    lr=configs.lr, betas=(configs.beta1, configs.beta2), weight_decay=3e-4)

                #if training_mode == "self_supervised" and "novelty_detection":  # to do it only once
                #    copy_Files(os.path.join(logs_save_dir, experiment_description, run_description), data_type)

                # Trainer
                model = Trainer(model, model_optimizer, classifier, classifier_optimizer, 
                                train_dl, device, logger, configs, experiment_log_dir, args, select_transformation(shifted_aug))

                
                # load saved model of this experiment
                path = os.path.join(os.path.join(logs_save_dir, experiment_description, 
                                        run_description, f"novelty_detection_seed_{args.seed}", "saved_models"))
                    #chkpoint = torch.load(os.path.join(path, "ckp_last.pt"), map_location=device)
                    #pretrained_dict = chkpoint["model_state_dict"]
                    #model.load_state_dict(pretrained_dict)
                    
                    # Evlauation
                with torch.no_grad():
                    auroc_dict, aupr_dict, fpr_dict, de_dict, one_class_total, one_class_aupr, one_class_fpr, one_class_de\
                        = eval_ood_detection(args, path, model, valid_dl, 
                                             ood_test_loader, args.ood_score, train_dl, select_transformation(shifted_aug))

                    auroc_a.append(one_class_total)     
                    aupr_a.append(one_class_aupr)   
                    fpr_a.append(one_class_fpr)
                    de_a.append(one_class_de)

                    mean_dict = dict()
                    for ood_score in args.ood_score:
                        mean = 0
                        for ood in auroc_dict.keys():
                            mean += auroc_dict[ood][ood_score]
                        mean_dict[ood_score] = mean / len(auroc_dict.keys())
                    auroc_dict['one_class_mean'] = mean_dict

                    bests = []
                    for ood in auroc_dict.keys():
                        print(ood)
                        message = ''
                        best_auroc = 0
                        for ood_score, auroc in auroc_dict[ood].items():
                            message += '[%s %s %.4f] ' % (ood, ood_score, auroc)
                            if auroc > best_auroc:
                                best_auroc = auroc
                        message += '[%s %s %.4f] ' % (ood, 'best', best_auroc)
                        if args.print_score:
                            print(message)
                        bests.append(best_auroc)

                    bests = map('{:.4f}'.format, bests)
                    print('\t'.join(bests))
                    print("novel_class:", novel_class)

            # # mean
            # print(f'{np.mean(auroc_a):.3f}')
            # print(f'{np.mean(aupr_a):.3f}')
            # print(f'{np.mean(fpr_a):.3f}')
            # print(f'{np.mean(de_a):.3f}')
            # # Standard deviation of list
            # print(f'{np.std(auroc_a):.3f}')
            # print(f'{np.std(aupr_a):.3f}')
            # print(f'{np.std(fpr_a):.3f}')
            # print(f'{np.std(de_a):.3f}')

            final_auroc.append([np.mean(auroc_a), np.std(auroc_a)])
            final_aupr.append([np.mean(aupr_a), np.std(aupr_a)])
            final_fpr.append([np.mean(fpr_a), np.std(fpr_a)])
            final_de.append([np.mean(de_a), np.std(de_a)])

    # overall
    # print(f'{auroc_a}')
    # print(f'{aupr_a}')
    # print(f'{fpr_a}')
    # print(f'{de_a}')

    # print(f'{final_auroc}')
    # print(f'{final_aupr}')
    # print(f'{final_fpr}')
    # print(f'{final_de}')

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
    df.to_excel('result_files/final_result_dataAug_' + str(args.ood_score[0])+'_'+data_type+'_comparison.xlsx', sheet_name='the results')

    logger.debug(f"Training time is : {datetime.now()-start_time}")

torch.cuda.empty_cache()

