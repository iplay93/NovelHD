import torch

import os
import numpy as np
from datetime import datetime
import argparse
from data_preprocessing.dataloader import splitting_data
from utils import _logger, set_requires_grad
#from trainer.trainer import Trainer, model_evaluate
from trainer.trainer_TFC import Trainer, model_evaluate
from models.TC import TC
from utils import _calc_metrics, copy_Files
from models.TFC import TFC, target_classifier
from dataloader import data_generator,data_generator_nd
from eval_nd import eval_ood_detection
    
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
parser.add_argument('--selected_dataset', default='Epilepsy', type=str,
                    help='Dataset of choice: sleepEDF, HAR, Epilepsy, pFD')
parser.add_argument('--logs_save_dir', default='experiments_logs', type=str,
                    help='saving directory')
parser.add_argument('--device', default='cuda', type=str,
                    help='cpu or cuda')
parser.add_argument('--home_path', default=home_dir, type=str,
                    help='Project home directory')

parser.add_argument('--padding', type=str, 
                    default='mean', help='choose one of them : no, max, mean')
parser.add_argument('--timespan', type=int, 
                    default=10000, help='choose of the number of timespan between data points(1000 = 1sec, 60000 = 1min)')
parser.add_argument('--min_seq', type=int, 
                    default=10, help='choose of the minimum number of data points in a example')
parser.add_argument('--min_samples', type=int, default=20, 
                    help='choose of the minimum number of samples in each label')
parser.add_argument('--one_class_idx', type=int, default=0, 
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

parser.add_argument('--version', type=str, default='CL', help='choose of version want to do : ND or CL')
parser.add_argument('--print_freq', type=int, default=1, help='print frequency')
parser.add_argument('--save_freq', type=int, default=50, help='save frequency')
parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset')
    
parser.add_argument('--aug_method', type=str, default='Dropout', help='choose the data augmentation method')
parser.add_argument('--aug_wise', type=str, default='Temporal', help='choose the data augmentation wise')

parser.add_argument('--test_ratio', type=float, default=0.3, help='choose the number of test ratio')
parser.add_argument('--valid_ratio', type=float, default=0.1, help='choose the number of vlaidation ratio')
parser.add_argument('--overlapped_ratio', type=int, default= 50, help='choose the number of windows''overlapped ratio')
parser.add_argument('--encoder', type=str, default='SupCon', help='choose one of them: simple, transformer')

    # for training   
parser.add_argument('--loss', type=str, default='SupCon', help='choose one of them: crossentropy loss, contrastive loss')
parser.add_argument('--optimizer', type=str, default='', help='choose one of them: adam')
parser.add_argument('--epochs', type=int, default= 20, help='choose the number of epochs')
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

logs_save_dir = args.logs_save_dir
os.makedirs(logs_save_dir, exist_ok=True)

args.abnormal_class = 3

exec(f'from config_files.{data_type}_Configs import Config as Configs')
configs = Configs()

# ##### fix random seeds for reproducibility ########
SEED = args.seed
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
logger.debug("=" * 45)

# Load datasets
data_path = f"./data/{data_type}"
if training_mode != "novelty_detection":
    train_dl, valid_dl, test_dl = data_generator(args, configs, training_mode)
else:
    train_dl, valid_dl, test_dl, ood_test_loader, novel_class = data_generator_nd(args, configs, training_mode)
logger.debug("Data loaded ...")

# Load Model
model = TFC(configs).to(device)
classifier = target_classifier(configs).to(device)

if training_mode == "fine_tune":
    # load saved model of this experiment
    load_from = os.path.join(os.path.join(logs_save_dir, experiment_description, run_description, f"self_supervised_seed_{SEED}", "saved_models"))
    chkpoint = torch.load(os.path.join(load_from, "ckp_last.pt"), map_location=device)
    pretrained_dict = chkpoint["model_state_dict"]
    model_dict = model.state_dict()
    del_list = ['logits']
    pretrained_dict_copy = pretrained_dict.copy()
    for i in pretrained_dict_copy.keys():
        for j in del_list:
            if j in i:
                del pretrained_dict[i]
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

if training_mode == "train_linear" or "tl" in training_mode:
    load_from = os.path.join(os.path.join(logs_save_dir, experiment_description, run_description, f"self_supervised_seed_{SEED}", "saved_models"))
    chkpoint = torch.load(os.path.join(load_from, "ckp_last.pt"), map_location=device)
    pretrained_dict = chkpoint["model_state_dict"]
    model_dict = model.state_dict()

    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    # delete these parameters (Ex: the linear layer at the end)
    del_list = ['logits']
    pretrained_dict_copy = pretrained_dict.copy()
    for i in pretrained_dict_copy.keys():
        for j in del_list:
            if j in i:
                del pretrained_dict[i]

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    set_requires_grad(model, pretrained_dict, requires_grad=False)  # Freeze everything except last layer.

if training_mode == "random_init":
    model_dict = model.state_dict()

    # delete all the parameters except for logits
    del_list = ['logits']
    pretrained_dict_copy = model_dict.copy()
    for i in pretrained_dict_copy.keys():
        for j in del_list:
            if j in i:
                del model_dict[i]
    set_requires_grad(model, model_dict, requires_grad=False)  # Freeze everything except last layer.


model_optimizer = torch.optim.Adam(model.parameters(), lr=configs.lr, betas=(configs.beta1, configs.beta2), weight_decay=3e-4)
classifier_optimizer = torch.optim.Adam(classifier.parameters(), lr=configs.lr, betas=(configs.beta1, configs.beta2), weight_decay=3e-4)

if training_mode == "self_supervised" and "novelty_detection":  # to do it only once
    copy_Files(os.path.join(logs_save_dir, experiment_description, run_description), data_type)

# Trainer
Trainer(model, model_optimizer, classifier, classifier_optimizer, train_dl, valid_dl, test_dl, device, logger, configs, experiment_log_dir, training_mode)



if training_mode != "self_supervised" and training_mode!="novelty_detection":
    # Testing
    outs = model_evaluate(model, classifier, test_dl, device, training_mode)
    total_loss, total_acc, total_f1, pred_labels, true_labels = outs
    _calc_metrics(pred_labels, true_labels, experiment_log_dir, args.home_path)

if training_mode == "novelty_detection":  
    # load saved model of this experiment
    path = os.path.join(os.path.join(logs_save_dir, experiment_description, 
                        run_description, f"novelty_detection_seed_{args.seed}", "saved_models"))
    chkpoint = torch.load(os.path.join(path, "ckp_last.pt"), map_location=device)
    pretrained_dict = chkpoint["model_state_dict"]
    model.load_state_dict(pretrained_dict)
    
    # Evlauation
    with torch.no_grad():    
        auroc_dict, aupr_dict, fpr_dict, f1_dict = eval_ood_detection(args, path, model,valid_dl, ood_test_loader, args.ood_score,
                                         train_loader=train_dl)
    
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


logger.debug(f"Training time is : {datetime.now()-start_time}")


torch.cuda.empty_cache()