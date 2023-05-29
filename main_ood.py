import torch

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
from torch.utils.data import DataLoader, Dataset
import torch.fft as fft
from data_preprocessing.dataloader import splitting_data
from tsaug import *

my_aug = (Pool(size=2))

class Load_Dataset(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, data_list, label_list, config, training_mode):
        super(Load_Dataset, self).__init__()
        self.training_mode = training_mode

        X_train = data_list
        y_train = label_list

        if len(X_train.shape) < 3:
            X_train = X_train.unsqueeze(2)

        #if X_train.shape.index(min(X_train.shape)) != 1:  # make sure the Channels in second dim
        X_train = X_train.permute(0, 2, 1)

        if isinstance(X_train, np.ndarray):
            self.x_data = torch.from_numpy(X_train)
            self.y_data = torch.from_numpy(y_train).long()
        else:
            self.x_data = X_train
            self.y_data = y_train

        self.x_data_f = fft.fft(self.x_data).abs() #/(window_length) # rfft for real value inputs.
        self.len = X_train.shape[0]
  
        self.aug1 = torch.from_numpy(np.array(my_aug.augment(self.x_data.cpu().numpy())))
        self.aug1_f = fft.fft(self.aug1).abs() 

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
    
parser.add_argument('--aug_method', type=str, default='AddNoise', help='choose the data augmentation method')
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
method = 'Test OOD-ness'
training_mode = args.training_mode
run_description = args.run_description

logs_save_dir = args.logs_save_dir
os.makedirs(logs_save_dir, exist_ok=True)

args.one_class_idx = 3

exec(f'from config_files.{data_type}_Configs import Config as Configs')
configs = Configs()
auroc_rs = []
f1_rs = []
# Training for five seed
for test_num in [20, 40, 60, 80, 100]:
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
    logger.debug("=" * 45)

    # Load datasets
    data_path = f"./data/{data_type}"

    num_classes, entire_list, train_list, valid_list, test_list, entire_label_list, train_label_list, valid_label_list, test_label_list \
    = splitting_data(args.selected_dataset, args.test_ratio, args.valid_ratio, args.padding, args.seed, \
                     args.timespan, args.min_seq, args.min_samples, args.aug_method, args.aug_wise)
    
    train_list = train_list.cpu()
    train_label_list = train_label_list.cpu()
    valid_list =valid_list.cpu()
    valid_label_list = valid_label_list.cpu()
    test_list = test_list.cpu()
    test_label_list = test_label_list.cpu()

    train_list = train_list[np.where(train_label_list == args.one_class_idx)]
    train_label_list = train_label_list[np.where(train_label_list == args.one_class_idx)]

    valid_list = valid_list[np.where(valid_label_list == args.one_class_idx)]
    valid_label_list = valid_label_list[np.where(valid_label_list == args.one_class_idx)]

    # only use for testing novelty
    test_list = test_list[np.where(test_label_list == args.one_class_idx)]
    test_label_list = test_label_list[np.where(test_label_list == args.one_class_idx)]
    
  
    # build data loader
    dataset = Load_Dataset(train_list, train_label_list, configs, training_mode)    
    train_dl = train_loader = DataLoader(dataset, batch_size=configs.batch_size, shuffle=True)

    dataset = Load_Dataset(valid_list, valid_label_list, configs, training_mode)
    valid_dl = DataLoader(dataset, batch_size=configs.batch_size, shuffle=True)

    dataset = Load_Dataset(test_list, test_label_list, configs, training_mode)
    test_dl = DataLoader(dataset, batch_size=configs.batch_size, shuffle=True)

    
    
    logger.debug("Data loaded ...")

    # Load Model
    model = TFC(configs).to(device)
    classifier = target_classifier(configs).to(device)

    model_optimizer = torch.optim.Adam(model.parameters(), lr=configs.lr, 
                                       betas=(configs.beta1, configs.beta2), weight_decay=3e-4)
    classifier_optimizer = torch.optim.Adam(classifier.parameters(), 
                                        lr=configs.lr, betas=(configs.beta1, configs.beta2), weight_decay=3e-4)

    # Trainer
    model = Trainer(model, model_optimizer, classifier, classifier_optimizer, train_dl, valid_dl, test_dl, device, logger, configs, experiment_log_dir, training_mode)


    # Testing
    outs = model_evaluate(model, classifier, test_dl, device, training_mode)
    total_loss, total_acc, total_f1, auroc, pred_labels, true_labels = outs
    _calc_metrics(pred_labels, true_labels, experiment_log_dir, args.home_path)
    auroc_rs.append(auroc.item())
    f1_rs.append(total_f1.item())

print("Average of the AUROC list =", round(sum(auroc_rs)/len(auroc_rs), 3))
print("Average of the F1 list =", round(sum(f1_rs)/len(f1_rs), 3))

torch.cuda.empty_cache()