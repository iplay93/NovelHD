import torch
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import torch.fft as fft
from data_preprocessing.dataloader import count_label_labellist, select_transformation
from data_preprocessing.augmentations import *
import random
from tsaug import *
from sklearn.model_selection import train_test_split
import math

class Load_Dataset(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, data_list, label_list, args, training_mode, positive_list):
        super(Load_Dataset, self).__init__()
        self.training_mode = training_mode

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

def data_generator_nd(args, configs, training_mode, positive_list, 
                                            num_classes, datalist, labellist):
    test_ratio = args.test_ratio 
    valid_ratio = args.valid_ratio
    seed =  args.seed 

    # Split train and valid dataset
    train_list, test_list, train_label_list, test_label_list = train_test_split(datalist, 
                                                                                labellist, test_size=test_ratio, stratify= labellist, random_state=seed) 
    print("Before len:", len(train_list))
    if args.train_num_ratio !=1 :
        train_list, _, train_label_list, _ = train_test_split(train_list,
                                                       train_label_list, test_size=(1-args.train_num_ratio), stratify= train_label_list, random_state=seed) 
    print("After len:", len(train_list))
    if len(train_list)< 1:
           raise ValueError("The training num is less than 1")

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

    # entire_list = entire_list.cpu()
    # entire_label_list = torch.tensor(entire_label_list).cuda().cpu()
 
    if(args.one_class_idx != -1): # one-class
        sup_class_idx = [x - 1 for x in num_classes]
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
    
    
    if args.binary:
        # for binary classification
        train_label_list[:] = 0
        valid_label_list[:] = 0
        test_label_list[:] = 1
        
    ood_test_loader = dict()

    if args.binary:
        for ood in [1]:
            # one class idx exit
            ood_test_set = Load_Dataset(test_list[np.where(test_label_list == ood)],
                                            test_label_list[np.where(test_label_list == ood)], 
                                            args, training_mode, positive_list)
            ood = f'one_class_{ood}'  # change save name

            ood_test_loader[ood] = DataLoader(ood_test_set, batch_size=configs.batch_size, shuffle=True)      

    if args.binary is not True:
        for ood in novel_class_idx:
            # one class idx exit
            ood_test_set = Load_Dataset(test_list[np.where(test_label_list == ood)],
                                        test_label_list[np.where(test_label_list == ood)], 
                                        args, training_mode, positive_list)
            ood = f'one_class_{ood}'  # change save name

            ood_test_loader[ood] = DataLoader(ood_test_set, batch_size=configs.batch_size, shuffle=True)          
        
   
    print("Length of OOD test loader", len(ood_test_loader))
   
    # build data loader (N, T, C) -> (N, C, T)
    dataset = Load_Dataset(train_list, train_label_list, args, training_mode, positive_list)
    total_size = len(dataset)
    split_size = int(args.data_size_ratio * total_size)
    remaining_size = total_size - split_size
    small_dataset, _ = random_split(dataset, [split_size, remaining_size])
    train_loader = DataLoader(small_dataset, batch_size = configs.batch_size, shuffle=True)

    dataset = Load_Dataset(valid_list,valid_label_list, args, training_mode, positive_list)
    finetune_loader = DataLoader(dataset, batch_size = configs.batch_size, shuffle=True)

    dataset = Load_Dataset(test_list, test_label_list, args, training_mode, positive_list)
    test_loader = DataLoader(dataset, batch_size=configs.batch_size, shuffle=True)


    return train_loader, finetune_loader, test_loader, ood_test_loader, novel_class_idx  


def generate_freq(dataset, config):
    X_train = dataset["samples"]
    y_train = dataset['labels']
    # shuffle
    data = list(zip(X_train, y_train))
    np.random.shuffle(data)
    data = data[:10000] # take a subset for testing.
    X_train, y_train = zip(*data)
    X_train, y_train = torch.stack(list(X_train), dim=0), torch.stack(list(y_train), dim=0)

    if len(X_train.shape) < 3:
        X_train = X_train.unsqueeze(2)

    if X_train.shape.index(min(X_train.shape)) != 1:  # make sure the Channels in second dim
        X_train = X_train.permute(0, 2, 1)

    """Align the TS length between source and target datasets"""
    X_train = X_train[:, :1, :int(config.TSlength_aligned)] # take the first 178 samples

    if isinstance(X_train, np.ndarray):
        x_data = torch.from_numpy(X_train)
    else:
        x_data = X_train

    """Transfer x_data to Frequency Domain. If use fft.fft, the output has the same shape; if use fft.rfft, 
    the output shape is half of the time window."""

    x_data_f = fft.fft(x_data).abs() #/(window_length) # rfft for real value inputs.
    return (X_train, y_train, x_data_f)



class Load_Dataset_2(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, data_list, label_list, config, training_mode):
        super(Load_Dataset_2, self).__init__()
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
        


def data_generator(args, configs, training_mode, aug_method):
    # num_classes, entire_list, train_list, valid_list, test_list, entire_label_list, train_label_list, valid_label_list, test_label_list \
    # = splitting_data(args.selected_dataset, args.test_ratio, args.valid_ratio, args.padding, args.seed, \
    #                  args.timespan, args.min_seq, args.min_samples, args.aug_method, args.aug_wise)
    
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
    dataset = Load_Dataset(train_list,train_label_list, configs, training_mode, aug_method)    
    train_loader = DataLoader(dataset, batch_size=configs.batch_size, shuffle=True)

    dataset = Load_Dataset(valid_list,valid_label_list, configs, training_mode, aug_method)
    finetune_loader = DataLoader(dataset, batch_size=configs.batch_size, shuffle=True)

    dataset = Load_Dataset(test_list,test_label_list, configs, training_mode, aug_method)
    test_loader = DataLoader(dataset, batch_size=configs.batch_size, shuffle=True)


    return train_loader, finetune_loader, test_loader




def data_generator_2(args, configs, training_mode):
    # num_classes, entire_list, train_list, valid_list, test_list, entire_label_list, train_label_list, valid_label_list, test_label_list \
    # = splitting_data(args.selected_dataset, args.test_ratio, args.valid_ratio, args.padding, args.seed, \
    #                  args.timespan, args.min_seq, args.min_samples, args.aug_method)
    
    train_list = train_list.cpu()
    train_label_list = train_label_list.cpu()
    valid_list =valid_list.cpu()
    valid_label_list = valid_label_list.cpu()
    test_list = test_list.cpu()
    test_label_list = test_label_list.cpu()    

    """In pre-training: 
    train_dataset: [371055, 1, 178] from SleepEEG.    
    finetune_dataset: [60, 1, 178], test_dataset: [11420, 1, 178] from Epilepsy"""
   
    # build data loader
    dataset = Load_Dataset_2(train_list,train_label_list, configs, training_mode)    
    train_loader = DataLoader(dataset, batch_size=configs.batch_size, shuffle=True)

    dataset = Load_Dataset_2(valid_list,valid_label_list, configs, training_mode)
    finetune_loader = DataLoader(dataset, batch_size=configs.batch_size, shuffle=True)

    dataset = Load_Dataset_2(test_list,test_label_list, configs, training_mode)
    test_loader = DataLoader(dataset, batch_size=configs.batch_size, shuffle=True)


    return train_loader, finetune_loader, test_loader





def data_generator_fold(args, configs, training_mode, positive_list, 
                                            num_classes, train_list, test_list, train_label_list, test_label_list ):

    print(f"Train Data: {len(train_list)} --------------")
    exist_labels, _ = count_label_labellist(train_label_list)
    
    # print(f"Validation Data: {len(valid_list)} --------------")    
    # count_label_labellist(valid_label_list)

    print(f"Test Data: {len(test_list)} --------------")
    count_label_labellist(test_label_list) 
    
    train_list = torch.tensor(train_list).cuda().cpu()
    train_label_list = torch.tensor(train_label_list).cuda().cpu()

    test_list = torch.tensor(test_list).cuda().cpu()
    test_label_list = torch.tensor(test_label_list).cuda().cpu()

    # entire_list = entire_list.cpu()
    # entire_label_list = torch.tensor(entire_label_list).cuda().cpu()
 
    if(args.one_class_idx != -1): # one-class
        sup_class_idx = [x - 1 for x in num_classes]
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
    
    
    if args.binary:
        # for binary classification
        train_label_list[:] = 0
        valid_label_list[:] = 0
        test_label_list[:] = 1
        
    ood_test_loader = dict()

    if args.binary:
        for ood in [1]:
            # one class idx exit
            ood_test_set = Load_Dataset(test_list[np.where(test_label_list == ood)],
                                            test_label_list[np.where(test_label_list == ood)], 
                                            args, training_mode, positive_list)
            ood = f'one_class_{ood}'  # change save name

            ood_test_loader[ood] = DataLoader(ood_test_set, batch_size=configs.batch_size, shuffle=True)      

    if args.binary is not True:
        for ood in novel_class_idx:
            # one class idx exit
            ood_test_set = Load_Dataset(test_list[np.where(test_label_list == ood)],
                                        test_label_list[np.where(test_label_list == ood)], 
                                        args, training_mode, positive_list)
            ood = f'one_class_{ood}'  # change save name

            ood_test_loader[ood] = DataLoader(ood_test_set, batch_size=configs.batch_size, shuffle=True)          
        
   
    print("Length of OOD test loader", len(ood_test_loader))
   
    # build data loader (N, T, C) -> (N, C, T)
    dataset = Load_Dataset(train_list, train_label_list, args, training_mode, positive_list)    
    train_loader = DataLoader(dataset, batch_size = configs.batch_size, shuffle=True)

    dataset = Load_Dataset(valid_list,valid_label_list, args, training_mode, positive_list)
    finetune_loader = DataLoader(dataset, batch_size = configs.batch_size, shuffle=True)

    dataset = Load_Dataset(test_list, test_label_list, args, training_mode, positive_list)
    test_loader = DataLoader(dataset, batch_size=configs.batch_size, shuffle=True)


    return train_loader, finetune_loader, test_loader, ood_test_loader, novel_class_idx  
