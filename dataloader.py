import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
import numpy as np
from augmentations import DataTransform_FD, DataTransform_TD
import torch.fft as fft
from data_preprocessing.dataloader import splitting_data
from augmentations import *
import random

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
    
        if training_mode == "pre_train":  # no need to apply Augmentations in other modes
            self.aug1 = DataTransform_TD(self.x_data, config)
            #self.aug1_f = DataTransform_FD(self.x_data_f, config) # [7360, 1, 90]
            self.aug1_f = fft.fft(self.aug1).abs() 
    def __getitem__(self, index):
        if self.training_mode == "pre_train":
            return self.x_data[index], self.y_data[index], self.aug1[index], self.x_data_f[index], self.aug1_f[index]
        else:
            return self.x_data[index], self.y_data[index], self.x_data[index], self.x_data_f[index], self.x_data_f[index]

    def __len__(self):
        return self.len

def data_generator(args, configs, training_mode):
    num_classes, entire_list, train_list, valid_list, test_list, entire_label_list, train_label_list, valid_label_list, test_label_list \
    = splitting_data(args.selected_dataset, args.test_ratio, args.valid_ratio, args.padding, args.seed, \
                     args.timespan, args.min_seq, args.min_samples, args.aug_method, args.aug_wise)
    
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
    dataset = Load_Dataset(train_list,train_label_list, configs, training_mode)    
    train_loader = DataLoader(dataset, batch_size=configs.batch_size, shuffle=True)

    dataset = Load_Dataset(valid_list,valid_label_list, configs, training_mode)
    finetune_loader = DataLoader(dataset, batch_size=configs.batch_size, shuffle=True)

    dataset = Load_Dataset(test_list,test_label_list, configs, training_mode)
    test_loader = DataLoader(dataset, batch_size=configs.batch_size, shuffle=True)


    return train_loader, finetune_loader, test_loader

def data_generator_2(args, configs, training_mode):
    num_classes, entire_list, train_list, valid_list, test_list, entire_label_list, train_label_list, valid_label_list, test_label_list \
    = splitting_data(args.selected_dataset, args.test_ratio, args.valid_ratio, args.padding, args.seed, \
                     args.timespan, args.min_seq, args.min_samples, args.aug_method)
    
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


def data_generator_nd(args, configs, training_mode):
    num_classes, entire_list, train_list, valid_list, test_list, entire_label_list, train_label_list, valid_label_list, test_label_list \
    = splitting_data(args.selected_dataset, args.test_ratio, 0, args.padding, args.seed, \
                     args.timespan, args.min_seq, args.min_samples, args.aug_method, args.aug_wise)
    
    train_list = train_list.cpu()
    train_label_list = train_label_list.cpu()

    test_list = test_list.cpu()
    test_label_list = test_label_list.cpu()

    entire_list = entire_list.cpu()
    entire_label_list = entire_label_list.cpu()

    if(args.one_class_idx != -1):    
        train_list = train_list[np.where(train_label_list == args.one_class_idx)]
        train_label_list = train_label_list[np.where(train_label_list == args.one_class_idx)]

        valid_list = test_list[np.where(test_label_list == args.one_class_idx)]
        valid_label_list =test_label_list[np.where(test_label_list == args.one_class_idx)]

        # only use for testing novelty
        test_list = entire_list[np.where(entire_label_list != args.one_class_idx)]
        test_label_list = entire_label_list[np.where(entire_label_list != args.one_class_idx)]

    else:
        sup_class_idx = num_classes
        print(sup_class_idx)
        known_class_idx = random.sample(sup_class_idx, (int)(len(num_classes)/2))
        print(known_class_idx)
        novel_class_idx = [item for item in sup_class_idx if item not in set(known_class_idx)]

        for k in range(len(novel_class_idx)):  
            one_class_idx = novel_class_idx[k]-1
            train_list = train_list[np.where(train_label_list != one_class_idx)]
            train_label_list = train_label_list[np.where(train_label_list != one_class_idx)]
            if k == 0:
                valid_list = test_list[np.where(test_label_list != one_class_idx)]
                valid_label_list =test_label_list[np.where(test_label_list != one_class_idx)]
            else: 
                valid_list = valid_list[np.where(valid_label_list != one_class_idx)]
                valid_label_list =valid_label_list[np.where(valid_label_list != one_class_idx)]


        for k in range(len(known_class_idx)):
            one_class_idx = known_class_idx[k]-1
            if k == 0:
            # only use for testing novelty
                test_list = entire_list[np.where(entire_label_list != one_class_idx)]
                test_label_list = entire_label_list[np.where(entire_label_list != one_class_idx)]
            else:
                test_list = test_list[np.where(test_label_list != one_class_idx)]
                test_label_list = test_label_list[np.where(test_label_list != one_class_idx)]
     
        

    """In pre-training: 
    train_dataset: [371055, 1, 178] from SleepEEG.    
    finetune_dataset: [60, 1, 178], test_dataset: [11420, 1, 178] from Epilepsy"""
   
    # build data loader
    dataset = Load_Dataset(train_list,train_label_list, configs, training_mode)    
    train_loader = DataLoader(dataset, batch_size=configs.batch_size, shuffle=True)

    dataset = Load_Dataset(valid_list,valid_label_list, configs, training_mode)
    finetune_loader = DataLoader(dataset, batch_size=configs.batch_size, shuffle=True)

    dataset = Load_Dataset(test_list,test_label_list, configs, training_mode)
    test_loader = DataLoader(dataset, batch_size=configs.batch_size, shuffle=True)


    return train_loader, finetune_loader, test_loader
