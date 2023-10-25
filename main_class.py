import os
import numpy as np
from datetime import datetime
import argparse
from utils import _logger
#from trainer.trainer import Trainer, model_evaluate
from trainer.trainer_class import Trainer_class, model_evaluate_class
from utils import _calc_metrics
from models.TFC import TFC_class, target_classifier, TFC
from data_preprocessing.dataloader import count_label_labellist, select_transformation
from trainer.trainer_ND import Trainer
import random, math
import torch
from torch.utils.data import DataLoader, Dataset
import torch.fft as fft
from data_preprocessing.dataloader import loading_data
from sklearn.model_selection import train_test_split
import pandas as pd
from eval_nd_combine import eval_ood_detection
import pickle

class Load_Dataset_nd(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, data_list, label_list, args, positive_list):
        super(Load_Dataset_nd, self).__init__()
        X_train = data_list
        y_train = label_list

        if len(X_train.shape) < 3:
            X_train = X_train.unsqueeze(2)

        # make sure the Channels in second dim
        X_train = X_train.permute(0, 2, 1)
        # (N, C, T)

        if isinstance(X_train, np.ndarray):
            self.x_data = torch.from_numpy(X_train)
            self.y_data = torch.from_numpy(y_train).long()
        else:
            self.x_data = X_train
            self.y_data = y_train

        # (N, C, T)
        self.x_data_f = fft.fft(self.x_data).abs() #/(window_length) # rfft for real value inputs.
        self.len = X_train.shape[0]
    
        # select positive transformation method        
        pos_aug = select_transformation(positive_list[0],  X_train.shape[2])
        # (N, C, T) -> (N, T, C)-> (N, C, T)
        self.aug1 = torch.from_numpy(np.array(pos_aug.augment(self.x_data.permute(0, 2, 1).cpu().numpy()))).permute(0, 2, 1)
        # (N, C, T)
        self.aug1_f = fft.fft(self.aug1).abs()

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index], self.aug1[index], self.x_data_f[index], self.aug1_f[index]

    def __len__(self):
        return self.len
    
class Load_Dataset(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, data_list, label_list):
        super(Load_Dataset, self).__init__()

        X_train = data_list
        y_train = label_list

        if len(X_train.shape) < 3:
            X_train = X_train.unsqueeze(2)
        
        X_train = X_train.permute(0, 2, 1)
        # (N, C, T)
        if isinstance(X_train, np.ndarray):
            self.x_data = torch.from_numpy(X_train)
            self.y_data = torch.from_numpy(y_train).long()
        else:
            self.x_data = X_train
            self.y_data = y_train

        self.len = X_train.shape[0]
        

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

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

parser.add_argument('--overlapped_ratio', type=int, default= 50, 
                    help='choose the number of windows''overlapped ratio')

# for training   
parser.add_argument('--loss', type=str, default='SupCon', help='choose one of them: crossentropy loss, contrastive loss')
parser.add_argument('--optimizer', type=str, default='', help='choose one of them: adam')
parser.add_argument('--patience', type=int, default=20, 
                    help='choose the number of patience for early stopping')
parser.add_argument('--lr', type=float, default=3e-5, 
                    help='choose the number of learning rate')
parser.add_argument('--gamma', type=float, default=0.7, 
                    help='choose the number of gamma')
parser.add_argument('--temp', type=float, default=0.07, 
                    help='temperature for loss function')
parser.add_argument('--warm', action='store_true',
                    help='warm-up for large batch training')

parser.add_argument('--training_ver', type=int, default=0, 
                    help='0: standard / 1: Retrain / 2 : Retrain + Novelty Detection ')

args = parser.parse_args()

torch.cuda.empty_cache()

device = torch.device(args.device)
experiment_description = args.experiment_description
data_type = args.selected_dataset
method = 'Test classification'
run_description = args.run_description
args.neg_ths = neg_ths  = 0.9

if args.training_ver == 0:
    retrain = False #False
    ND = False
elif args.training_ver == 1:
    retrain = True #False
    ND = False
elif args.training_ver == 2:
    retrain = False #False
    ND = True

    with open('./data/'+data_type+'_s_'+str(neg_ths)+'-novel.data', 'rb') as f:
        strong_set = pickle.load(f)

    with open('./data/'+data_type+'_fs_'+str(neg_ths)+'-novel.data', 'rb') as f:
        strong_set_f = pickle.load(f)

else:
    raise ValueError



if retrain:
    store_path = 'result_files/final_class_'+ data_type +'- retrain.xlsx'
elif ND:
    store_path = 'result_files/final_class_'+ data_type +'- novel.xlsx'
else:
    store_path = 'result_files/final_class_'+ data_type +'.xlsx'


logs_save_dir = args.logs_save_dir
os.makedirs(logs_save_dir, exist_ok=True)

exec(f'from config_files.{data_type}_Configs import Config as Configs')
configs = Configs()

if data_type == 'lapras': 
    args.timespan = 10000
    class_num = [0, 1, 2, 3,-1]

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

args.K_shift_f = args.K_shift = 2

clssfication_arr = []



# Training for except one class_idx
for num, args.one_class_idx in enumerate(class_num):

    acc_rs = []
    f1_rs  = []
    auroc_rs = []
    seed_set = []

    for seed_n, test_num in enumerate([20, 40, 60, 80, 100]):
    # Training for five seed
        
        # ##### fix random seeds for reproducibility ########
        SEED = args.seed = test_num
        torch.manual_seed(SEED)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = False
        np.random.seed(SEED)
        random.seed(args.seed)
        #####################################################
    # Classification
        experiment_log_dir = os.path.join(logs_save_dir, experiment_description, run_description, f"novelty_detection_seed_{args.seed}")
        os.makedirs(experiment_log_dir, exist_ok=True)

        # Logging
        log_file_name = os.path.join(experiment_log_dir, f"logs_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log")
        logger = _logger(log_file_name)
        logger.debug("=" * 45)
        logger.debug(f'Dataset: {data_type}')
        logger.debug(f'Method:  {method}')
        logger.debug(f'Seed:    {SEED}')
        logger.debug(f'one idx:    {args.one_class_idx}')


        test_ratio = args.test_ratio
        seed =  args.seed 

        # Split train and test dataset
        train_list, test_list, train_label_list, test_label_list = train_test_split(datalist, 
                        labellist, test_size=test_ratio, stratify= labellist, random_state=seed) 
        
                                        
        train_list = torch.tensor(train_list).cuda().cpu()
        train_label_list = torch.tensor(train_label_list).cuda().cpu()
        test_list = torch.tensor(test_list).cuda().cpu()
        test_label_list = torch.tensor(test_label_list).cuda().cpu()

        
        exist_labels, _ = count_label_labellist(train_label_list)
            
        if args.one_class_idx != -1: # one-class
            sup_class_idx = [x - 1 for x in num_classes]
            novel_class_idx = [args.one_class_idx]
            known_class_idx = [item for item in sup_class_idx if item not in set(novel_class_idx)]             

        else: # multi-class
            sup_class_idx   = [x - 1 for x in num_classes]            
            novel_class_idx = random.sample(sup_class_idx, math.ceil(len(sup_class_idx)/2))
            known_class_idx = [item for item in sup_class_idx if item not in set(novel_class_idx)] 
                
        novel_list = train_list[np.isin(train_label_list, novel_class_idx)]
        novel_label_list = train_label_list[np.isin(train_label_list, novel_class_idx)]

        train_list = train_list[np.isin(train_label_list, known_class_idx)]
        train_label_list = train_label_list[np.isin(train_label_list, known_class_idx)]        

        # only use for testing novelty
        test_list = test_list[np.isin(test_label_list,  known_class_idx)]
        test_label_list = test_label_list[np.isin(test_label_list,  known_class_idx)]    

        print(f"Train Data: {len(train_list)} --------------")
        count_label_labellist(train_label_list)

        print(f"Novel Data: {len(novel_list)} --------------")
        count_label_labellist(novel_label_list)  
        
        print(f"Test Data: {len(test_list)} --------------")
        count_label_labellist(test_label_list) 

        logger.debug(f'known class:    {known_class_idx}')
        logger.debug("=" * 45)

        
        # 클래스 번호를 연속적으로 매핑
        class_mapping = {class_label: idx for idx, class_label in enumerate(known_class_idx)}

        mapped_class_labels = [class_mapping[class_label] for class_label in known_class_idx]
        
        print("Original Class Labels:", known_class_idx)
        print("Mapped Class Labels:", mapped_class_labels)

        # 매핑된 클래스 번호 출력
        train_label_list = torch.tensor([class_mapping[label.item()] for label in train_label_list])
        test_label_list = torch.tensor([class_mapping[label.item()] for label in test_label_list])
       
        #print("Original Batch Labels:", train_label_list)
        #print("Remapped Batch Labels:", test_label_list)
       
                               
        
        # Build data loader
        dataset = Load_Dataset(train_list, train_label_list)    
        train_dl = DataLoader(dataset, batch_size=configs.batch_size, shuffle=True)

        dataset = Load_Dataset(test_list, test_label_list)
        test_dl = DataLoader(dataset, batch_size=configs.batch_size, shuffle=True)
                        
        logger.debug("Data loaded ...")           

        # Load Model
        model = TFC_class(configs, args, len(known_class_idx)).to(device)
        classifier = target_classifier(configs).to(device)

        model_optimizer = torch.optim.Adam(model.parameters(), lr=configs.lr, 
                        betas=(configs.beta1, configs.beta2), weight_decay=3e-4)
        classifier_optimizer = torch.optim.Adam(classifier.parameters(), 
                        lr=configs.lr, betas=(configs.beta1, configs.beta2), weight_decay=3e-4)

        # Trainer
        model = Trainer_class(model, model_optimizer, classifier, classifier_optimizer, 
                            train_dl, device, logger, configs, experiment_log_dir)
        
        # Novelty detection
        

        if retrain:
            
            
            # # (N, C, T)
            #if isinstance(novel_list, np.ndarray):
            #     novel_list = torch.from_numpy(novel_list)

            _, s_t = model(novel_list.permute(0, 2, 1).float().to(device))
            novel_label_list = torch.tensor(np.argmax(s_t.detach().cpu().numpy(), axis=1)).cuda().cpu()
            train_list = torch.cat((train_list, novel_list),0)
            train_label_list = torch.cat((train_label_list, novel_label_list),0)

            # _, s_t = model(train_list.permute(0, 2, 1).float().to(device))
            # train_label_list = torch.tensor(np.argmax(s_t.detach().cpu().numpy(), axis=1)).cuda().cpu()

            model = TFC_class(configs, args, len(known_class_idx)).to(device)
            model_optimizer = torch.optim.Adam(model.parameters(), lr=configs.lr, 
                        betas=(configs.beta1, configs.beta2), weight_decay=3e-4)

            print("#"*45)
            print(train_list.shape, train_label_list.shape)
            # Build data loader
            dataset = Load_Dataset(train_list, train_label_list)    
            train_dl = DataLoader(dataset, batch_size=configs.batch_size, shuffle=True)
            model = Trainer_class(model, model_optimizer, classifier, classifier_optimizer, 
                            train_dl, device, logger, configs, experiment_log_dir)
            
        elif ND:
            positive_list = ['AddNoise'] 
            positive_list_f = ['AddNoise']
                
            negative_list = strong_set[num]
            negative_list_f = strong_set_f[num]

            args.ood_score = ['NovelHD']
            args.temp = 0.7
            args.training_mode = 'novelty_detection'
            args.K_shift = len(negative_list) + 1
            args.K_shift_f = len(negative_list_f) + 1
            args.K_pos = len (positive_list) 
            args.K_pos_f = len (positive_list_f) 
            args.lam_a = 0.5
            logger.debug(f'Mode:    {args.training_mode}')
            logger.debug(f'Positive Augmentation:    {positive_list}')
            logger.debug(f'Negative Augmentation:    {negative_list}')
            logger.debug(f'Positive_F Augmentation:    {positive_list_f}')
            logger.debug(f'Negative_F Augmentation:    {negative_list_f}')
            logger.debug(f'Seed:    {SEED}')
            logger.debug(f'Version:    {args.ood_score}')
            logger.debug(f'One_class_idx:    {args.one_class_idx}')
            logger.debug(f'Neg ths:    {args.neg_ths}')
            logger.debug(f'Temperature:    {args.temp}')
            logger.debug("=" * 45)           
            
            # train_list_nd는 train이자 valid로 쓰임
            train_list_nd  = train_list
            test_list_nd  = novel_list

            train_label_list_nd = train_label_list
            test_label_list_nd =  novel_label_list

            train_label_list_nd [:] = 0            
            test_label_list_nd [:] = 1

            ood_test_loader = dict()
            ood_test_set = Load_Dataset_nd(test_list_nd, test_label_list_nd, args, positive_list)            
            ood = f'one_class_{1}'  # change save name
            ood_test_loader[1] = DataLoader(ood_test_set, batch_size=configs.batch_size, shuffle=True)  

            dataset = Load_Dataset_nd(train_list_nd, train_label_list_nd, args, positive_list)    
            train_loader = DataLoader(dataset, batch_size = configs.batch_size, shuffle=True)

            model_nd = TFC(configs, args).to(device)
            classifier_nd = target_classifier(configs).to(device)
            model_optimizer_nd = torch.optim.Adam(model_nd.parameters(), 
                                                lr=configs.lr, betas=(configs.beta1, configs.beta2), 
                                                weight_decay=3e-4)
            classifier_optimizer_nd = torch.optim.Adam(classifier_nd.parameters(), 
                                                    lr=configs.lr, betas=(configs.beta1, configs.beta2), 
                                                    weight_decay=3e-4)
            
            model_nd = Trainer(model_nd, model_optimizer_nd, classifier_nd, classifier_optimizer_nd, 
                    train_loader, device, logger, configs, experiment_log_dir, args, negative_list, 
                    negative_list_f, positive_list, positive_list_f)
            
            data_path = f"./data/{data_type}"
            path = os.path.join(os.path.join(logs_save_dir, experiment_description, 
                                        run_description, f"novelty_detection_seed_{args.seed}", "saved_models"))
               
            with torch.no_grad():
                outputs = eval_ood_detection(args, path, model_nd, train_loader, 
                                             ood_test_loader, args.ood_score, train_loader, 
                                             negative_list, negative_list_f)
            
            train_list = torch.cat((train_list, novel_list),0)
            print(train_list.shape)

            # train_list와 output의 길이가 같아야 합니다.
            if len(train_list) == len(outputs):
                train_list = train_list.tolist()
                train_list = [train_list[i] for i in range(len(train_list)) if outputs[i] != 0]
                train_list = torch.tensor(train_list)
            else:
                print("train_list와 outputs의 길이가 다릅니다.")

            print(train_list.shape)

            _, s_t = model(train_list.permute(0, 2, 1).float().to(device))
            train_label_list = torch.tensor(np.argmax(s_t.detach().cpu().numpy(), axis=1)).cuda().cpu()

            model = TFC_class(configs, args, len(known_class_idx)).to(device)
            model_optimizer = torch.optim.Adam(model.parameters(), lr=configs.lr, 
                        betas=(configs.beta1, configs.beta2), weight_decay=3e-4)
            
            print("#"*45)
            print(train_list.shape, train_label_list.shape)
            # Build data loader
            dataset = Load_Dataset(train_list, train_label_list)    
            train_dl = DataLoader(dataset, batch_size=configs.batch_size, shuffle=True)
            model = Trainer_class(model, model_optimizer, classifier, classifier_optimizer, 
                            train_dl, device, logger, configs, experiment_log_dir)
            



        # evaluate on the test set
        logger.debug('\nEvaluate on the Test set:')
        outs = model_evaluate_class(model, classifier, test_dl, device)
        total_acc, total_auc, total_f1, trgs = outs
        logger.debug(f'Test Accuracy : {total_acc:2.4f} | Test F1 : {total_f1:.4f}\t | \tTest AUROC : {total_auc:2.4f}')
            
        # Testing        
        #_calc_metrics(pred_labels, true_labels, experiment_log_dir, args.home_path)
        acc_rs.append(total_acc.item())
        f1_rs.append(total_f1.item())
        auroc_rs.append(total_auc.item())    
        
    print("Average of the Accuracy list =", round(sum(acc_rs)/len(acc_rs), 3))
    print(np.mean(acc_rs))
    print("Average of the F1 list =", round(sum(f1_rs)/len(f1_rs), 3))
    print(np.mean(f1_rs))
    print("Average of the AUROC list =", round(sum(auroc_rs)/len(auroc_rs), 3))
    print(np.mean(auroc_rs))
    final_acc.append([np.mean(acc_rs), np.std(acc_rs)])
    final_f1.append([np.mean(f1_rs), np.std(f1_rs)])
    final_auroc.append([np.mean(auroc_rs), np.std(auroc_rs)])

# for extrating results to an excel file
final_rs =[]
for i in final_auroc:
    final_rs.append(i)
for i in final_acc:
    final_rs.append(i)
for i in final_f1:
    final_rs.append(i)

print("Finished")

df = pd.DataFrame(final_rs, columns=['mean', 'std'])
df.to_excel(store_path, sheet_name='the results')


torch.cuda.empty_cache()