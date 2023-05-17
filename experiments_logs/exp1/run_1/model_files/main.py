import torch

import os
import numpy as np
from datetime import datetime
import argparse
from data_preprocessing.dataloader import splitting_data
from utils import _logger, set_requires_grad
from trainer.trainer import Trainer, model_evaluate
from models.TC import TC
from utils import _calc_metrics, copy_Files
from models.model import base_Model
from torch.utils.data import Dataset,DataLoader

def DataTransform(sample, config):

    weak_aug = scaling(sample.cpu(), config.augmentation.jitter_scale_ratio)
    strong_aug = jitter(permutation(sample.cpu(), max_segments=config.augmentation.max_seg), config.augmentation.jitter_ratio)

    return weak_aug, strong_aug


def jitter(x, sigma=0.8):
    # https://arxiv.org/pdf/1706.00527.pdf
    return x + np.random.normal(loc=0., scale=sigma, size=x.shape)


def scaling(x, sigma=1.1):
    # https://arxiv.org/pdf/1706.00527.pdf
    factor = np.random.normal(loc=2., scale=sigma, size=(x.shape[0], x.shape[2]))
    ai = []
    for i in range(x.shape[1]):
        xi = x[:, i, :]
        ai.append(np.multiply(xi, factor[:, :])[:, np.newaxis, :])
    return np.concatenate((ai), axis=1)


def permutation(x, max_segments=5, seg_mode="random"):
    orig_steps = np.arange(x.shape[2])

    num_segs = np.random.randint(1, max_segments, size=(x.shape[0]))

    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        if num_segs[i] > 1:
            if seg_mode == "random":
                split_points = np.random.choice(x.shape[2] - 2, num_segs[i] - 1, replace=False)
                split_points.sort()
                splits = np.split(orig_steps, split_points)
            else:
                splits = np.array_split(orig_steps, num_segs[i])
            warp = np.concatenate(np.random.permutation(splits)).ravel()
            ret[i] = pat[0,warp]
        else:
            ret[i] = pat
    return torch.from_numpy(ret)


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

        self.len = X_train.shape[0]
        if training_mode == "self_supervised":  # no need to apply Augmentations in other modes
            self.aug1, self.aug2 = DataTransform(self.x_data, config)

    def __getitem__(self, index):
        if self.training_mode == "self_supervised":
            return self.x_data[index], self.y_data[index], self.aug1[index], self.aug2[index]
        else:
            return self.x_data[index], self.y_data[index], self.x_data[index], self.x_data[index]

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

parser.add_argument('--padding', type=str, default='mean', help='choose one of them : no, max, mean')
parser.add_argument('--timespan', type=int, default=10000, help='choose of the number of timespan between data points(1000 = 1sec, 60000 = 1min)')
parser.add_argument('--min_seq', type=int, default=10, help='choose of the minimum number of data points in a example')
parser.add_argument('--min_samples', type=int, default=20, help='choose of the minimum number of samples in each label')
parser.add_argument('--arg_ood', type=int, default=-1, help='choose of label number that wants to delete')
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
method = 'TS-TCC'
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
num_classes, entire_list, train_list, valid_list, test_list, entire_label_list, train_label_list, valid_label_list, test_label_list \
    = splitting_data(data_type, args.test_ratio, args.valid_ratio, args.padding, args.seed, \
                     args.timespan, args.min_seq, args.min_samples, args.aug_method, args.aug_wise, args.arg_ood, args.version)

train_list = train_list.cpu()
train_label_list = train_label_list.cpu()
valid_list =valid_list.cpu()
valid_label_list = valid_label_list.cpu()
test_list = test_list.cpu()
test_label_list = test_label_list.cpu()

# train_list = train_list[np.where(train_label_list!=args.abnormal_class)]
# train_label_list = train_label_list[np.where(train_label_list!=args.abnormal_class)]


# valid_list = valid_list[np.where(valid_label_list!=args.abnormal_class)]
# valid_label_list =valid_label_list[np.where(valid_label_list!= args.abnormal_class)]

# print(valid_list)
# print(valid_label_list)

# test_list = test_list[np.where(test_label_list==args.abnormal_class)]
# test_label_list = test_label_list[np.where(test_label_list==args.abnormal_class)]

# build data loader
dataset = Load_Dataset(train_list,train_label_list, configs, training_mode)
train_dl = DataLoader(dataset, batch_size=configs.batch_size, shuffle=True)

dataset = Load_Dataset(valid_list,valid_label_list, configs, training_mode)
valid_dl = DataLoader(dataset, batch_size=configs.batch_size, shuffle=True)

dataset = Load_Dataset(test_list,test_label_list, configs, training_mode)
test_dl = DataLoader(dataset, batch_size=configs.batch_size, shuffle=True)

logger.debug("Data loaded ...")

# Load Model
model = base_Model(configs).to(device)
temporal_contr_model = TC(configs, device).to(device)

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
temporal_contr_optimizer = torch.optim.Adam(temporal_contr_model.parameters(), lr=configs.lr, betas=(configs.beta1, configs.beta2), weight_decay=3e-4)

if training_mode == "self_supervised":  # to do it only once
    copy_Files(os.path.join(logs_save_dir, experiment_description, run_description), data_type)

# Trainer
Trainer(model, temporal_contr_model, model_optimizer, temporal_contr_optimizer, train_dl, valid_dl, test_dl, device, logger, configs, experiment_log_dir, training_mode)

if training_mode != "self_supervised":
    # Testing
    outs = model_evaluate(model, temporal_contr_model, test_dl, device, training_mode)
    total_loss, total_acc, total_f1, pred_labels, true_labels = outs
    _calc_metrics(pred_labels, true_labels, experiment_log_dir, args.home_path)

logger.debug(f"Training time is : {datetime.now()-start_time}")


torch.cuda.empty_cache()