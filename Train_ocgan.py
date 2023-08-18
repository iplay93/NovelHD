from __future__ import print_function
import math
import os
import matplotlib as mpl
import tarfile
import matplotlib.image as mpimg
from matplotlib import pyplot as plt

import torch
import torch.nn as nn

import numpy as np
import random
from random import shuffle

from datetime import datetime
import time
import logging
from OCGAN.models import set_network

from utils import Logger, AverageMeter

# for modifying
from torch.utils.data import DataLoader, Dataset
from data_preprocessing.dataloader import loading_data, count_label_labellist
from sklearn.model_selection import train_test_split
from ood_metrics import auroc, aupr, fpr_at_95_tpr, detection_error
from sklearn.metrics import roc_auc_score, roc_curve, auc
import pandas as pd
import numpy as np
import torch.optim as optim

# visualization
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import RocCurveDisplay
from sklearn import preprocessing
from sklearn.preprocessing import LabelBinarizer

import torch.autograd


import argparse

def train_options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expname", default="expce", help="Name of the experiment")

    parser.add_argument("--epochs", default=101, type=int,
                        help="Number of epochs for training")
    parser.add_argument("--use_gpu", default=1, type=int,  help="1 to use GPU  ")

    parser.add_argument("--lr", default="0.005", type=float, help="Base learning rate")

    parser.add_argument("--beta1", default=0.5, type=float, help="Parameter for Adam")
    parser.add_argument("--lambda1", default=500, type=float, help="Weight of reconstruction loss")
    parser.add_argument("--datapath", default='/users/pramudi/Documents/data/', help="Data path")

    parser.add_argument("--continueEpochFrom", default=-1,
                        help="Continue training from specified epoch")
    parser.add_argument("--noisevar", default=0.02, type=float, help="variance of noise added to input")
    parser.add_argument("--depth", default=3, type=int, help="Number of core layers in Generator/Discriminator")
    parser.add_argument("--seed", default=20, type=float, help="Seed generator. Use -1 for random.")
    parser.add_argument("--append", default=0, type=int, help="Append discriminator input. 1 for true")
    parser.add_argument("--classes", default="", help="Name of training class. Keep blank for random")
    parser.add_argument("--latent", default=16, type=int,  help="Dimension of the latent space.")
    parser.add_argument("--ntype", default=4, type=int, help="Novelty detector: 1 - AE 2 - ALOCC 3 - latentD 4 - OCGAN")
    parser.add_argument("--protocol", default=1, type=int, help="1 : 80/20 split, 2 : Train / Test split")
   
    # Modify
    parser.add_argument('--one_class_idx', type=int, default= 0, 
                        help='choose of one class label number that wants to deal with. -1 is for multi-classification')
    
    parser.add_argument('--padding', type=str, 
                        default='mean', help='choose one of them : no, max, mean')
    parser.add_argument('--timespan', type=int, 
                            default=100, help='choose of the number of timespan between data points(1000 = 1sec, 60000 = 1min)')
    parser.add_argument('--min_seq', type=int, 
                            default=10, help='choose of the minimum number of data points in a example')
    parser.add_argument('--min_samples', type=int, default=20, 
                            help='choose of the minimum number of samples in each label')
    parser.add_argument('--selected_dataset', default='lapras', type=str,
                            help='Dataset of choice: lapras, casas, opportunity, aras_a, aras_b')
    parser.add_argument('--aug_method', type=str, default='AddNoise', help='choose the data augmentation method')
    parser.add_argument('--aug_wise', type=str, default='Temporal', help='choose the data augmentation wise')

    parser.add_argument('--test_ratio', type=float, default=0.2, help='choose the number of test ratio')
    parser.add_argument('--valid_ratio', type=float, default=0.2, help='choose the number of test ratio')

    parser.add_argument("--batch_size", default=64, type=int, help="Batch size per iteration")
    parser.add_argument('--device', default='cuda', type=str,
                    help='cpu or cuda')
    # Checkpoints
    parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    
    args = parser.parse_args()
    if args.use_gpu == 1:
        args.use_gpu = True
    else:
        args.use_gpu = False
    if args.append == 1:
        args.append = True
    else:
        args.append = False
    return args

def plotloss(loss_vec, fname):
    plt.gcf().clear()
    plt.plot(loss_vec[0], label="Dr", alpha = 0.7)
    plt.plot(loss_vec[4], label="Dl", alpha = 0.7)
    plt.plot(loss_vec[1], label="G", alpha=0.7)
    plt.plot(loss_vec[2], label="R", alpha= 0.7)
    plt.plot(loss_vec[3], label="Acc", alpha = 0.7)
    plt.legend()
    plt.savefig(fname)

class Load_Dataset(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, data_list, label_list):
        super(Load_Dataset, self).__init__()
        
        X_train = data_list
        y_train = label_list

        if len(X_train.shape) < 3:
            X_train = X_train.unsqueeze(2)

        # make sure the Channels in second dim
        X_train = np.transpose(X_train,(0, 2, 1))
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
    

def data_generator(args, datalist, labellist):
    test_ratio = args.test_ratio
    valid_ratio = args.valid_ratio
    seed =  args.seed 

    # Split train and valid dataset
    train_list, test_list, train_label_list, test_label_list = train_test_split(datalist, 
                                                                                labellist, test_size=test_ratio, stratify= labellist, random_state=seed) 
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

    print("test label", test_label_list, len(test_label_list))
 
    if(args.one_class_idx != -1): # one-class
        sup_class_idx = [x for x in exist_labels]
        

        known_class_idx = [args.one_class_idx]
        novel_class_idx = [item for item in sup_class_idx if item not in set(known_class_idx)]
        
        train_list = train_list[np.where(train_label_list == args.one_class_idx)]
        train_label_list = train_label_list[np.where(train_label_list == args.one_class_idx)]

        valid_list = test_list[np.where(test_label_list == args.one_class_idx)]
        valid_label_list = test_label_list[np.where(test_label_list == args.one_class_idx)]

        # only use for testing novelty
        test_list = test_list[np.where(test_label_list != args.one_class_idx)]
        test_label_list  = test_label_list[np.where(test_label_list != args.one_class_idx)]


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

        
    train_label_list[:] = 1
    
    # for testing
    valid_label_list[:] = 1
    test_label_list[:] = 0

    dataset = Load_Dataset(train_list, train_label_list)
    train_loader = DataLoader(dataset, batch_size = args.batch_size, shuffle=True)   


    # replace label : anomaly -> 0 : normal -> 1
    replace_list = np.concatenate((valid_list, test_list),axis=0)
    replace_label_list = np.concatenate((valid_label_list, test_label_list),axis=0)

    print("test label_Modified", replace_label_list, len(replace_label_list))
    print("novel_class:", novel_class_idx)
    
    dataset = Load_Dataset(replace_list , replace_label_list)
    test_loader = DataLoader(dataset, batch_size = args.batch_size, shuffle=True)
    

    return train_loader, test_loader, novel_class_idx  


def train_ae(args,trainloader, enc, dec, optimizer_en, optimizer_de, criterion, epoch, 
             use_cuda, channel, sequence_length):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    enc.train()
    dec.train()

    end = time.time()    


    for batch_idx, (inputs, targets) in enumerate(trainloader):

        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
        
        u = np.random.uniform(-1, 1, (len(inputs), channel, sequence_length))   
        l2 = torch.from_numpy(u).float()

        n = torch.randn(len(inputs), channel, sequence_length).cuda()
        l1 = enc(inputs + n)
        #print(inputs.shape)
        #print(l1.shape)
        del1 = dec(l1)
        #print(del1.shape)

        loss = criterion(del1,inputs)


        losses.update(loss.item(), inputs.size(0))


        enc.zero_grad()
        dec.zero_grad()

        loss.backward()

        optimizer_en.step()
        optimizer_de.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # # plot progress
        # bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f}  '.format(
        #             batch=batch_idx + 1,
        #             size=len(trainloader),
        #             data=data_time.avg,
        #             bt=batch_time.avg,
        #             total=bar.elapsed_td,
        #             eta=bar.eta_td,
        #             loss=losses.avg,
        #             )

  


    return losses.avg

def trainAE(opt, trainloader, testloader, device,  networks):

    netEn = networks[0].cuda()
    netDe = networks[1].cuda()
    
    epochs = opt.epochs
    batch_size = opt.batch_size

    L1_loss =  nn.MSELoss()

    
    loss_rec_G = []
    loss_rec_D = []
    loss_rec_R = []
    loss_rec_D2 = []


    optimizer_en = optim.Adam(netEn.parameters(), lr=args.lr, betas=(0.9, 0.99))
    optimizer_de = optim.Adam(netDe.parameters(), lr=args.lr, betas=(0.9, 0.99))

    for epoch in range(1, epochs):
        tic = time.time()
        btic = time.time()

        iter = 0

        for batch_idx, (inputs, targets)  in enumerate(trainloader):
            n = torch.randn(inputs.shape, device=inputs.device)

            real_in = inputs.cuda()
            real_out = inputs.cuda() + n.cuda()                      
            
           
            fake_out = netDe(netEn(real_in))
            errR = L1_loss(real_out, fake_out)

            netEn.zero_grad()
            netDe.zero_grad()
            
            errR.backward()
            
            optimizer_en.step()
            optimizer_de.step()
        
        loss_rec_R.append(np.mean(errR.item()))

                
        iter = iter + 1

        btic = time.time()

    # testing
    lbllist = []
    scorelist = []


    count = 0

    for batch_idx, (inputs, targets)  in enumerate(testloader):
        count = count+1
        output1=np.zeros(args.batch_size)

        real_in = inputs.cuda()
        n = torch.randn(inputs.shape, device=inputs.device)

        real_out =  inputs.cuda() + n.cuda()
        lbls_in = targets.cuda()


        outnn = (netDe(netEn(real_in)))         
        #print(outnn.shape)
        output1 = -1* torch.mean((outnn - real_out)**2,dim=[1,2])


        lbllist = lbllist+lbls_in.tolist()
        scorelist = scorelist + output1.tolist()

    fpr, tpr, _ = roc_curve(lbllist, scorelist, pos_label=1)
    roc_auc1 = auc(fpr, tpr)    

    print(roc_auc1)
    
    print(max([roc_auc1]))

    scores = scorelist
    labels = lbllist

    return auroc(scores, labels), aupr(scores, labels), fpr_at_95_tpr(scores, labels), detection_error(scores, labels), scores, labels



def trainadnov(opt, trainloader, testloader, device, networks):

    netEn = networks[0].cuda()
    netDe = networks[1].cuda()
    netD = networks[2].cuda()
    netD2 = networks[3].cuda()
    netDS = networks[4].cuda()
    # trainerEn = networks[5]
    # trainerDe = networks [6]
    # trainerD =networks[7]
    # trainerD2 = networks[8]
    # trainerSD = networks[9]
    

    epochs = opt.epochs
    lambda1 = opt.lambda1
    batch_size = opt.batch_size

    
    GAN_loss = nn.BCELoss()
    L1_loss = nn.MSELoss()



    loss_rec_G2 =[]
    acc2_rec = []
    loss_rec_G = []
    loss_rec_D = []
    loss_rec_R = []
    acc_rec = []
    loss_rec_D2 = []



    lr = args.lr = 2.0 * batch_size

    l2_int=torch.empty(size=(args.batch_size, 32), dtype=torch.float32) 
    optimizer_en = optim.Adam(netEn.parameters(), lr=args.lr, betas=(0.9, 0.99))
    optimizer_de = optim.Adam(netDe.parameters(), lr=args.lr, betas=(0.9, 0.99))
    optimizer_d = optim.Adam(netD.parameters(), lr=args.lr, betas=(0.9, 0.99))
    optimizer_d2 = optim.Adam(netD2.parameters(), lr=args.lr, betas=(0.9, 0.99))
    optimizer_ds  = optim.Adam(netDS.parameters(), lr=args.lr, betas=(0.9, 0.99))
    optimizer_l2 = optim.Adam([{'params':l2_int}], lr=args.lr, betas=(0.9, 0.99))



    for epoch in range(1, epochs):
        tic = time.time()
        btic = time.time()

        iter = 0
        for batch_idx, (inputs, targets)  in enumerate(trainloader):
            ############################
            # (1) Update D network: maximize log(D(x, y)) + log(1 - D(x, G(x, z)))
            ###########################
            real_in = inputs.cuda()
            n = torch.randn(inputs.shape, device=inputs.device)
        #print("input", inputs.shape)
        #l1 = enc(inputs + n)
            real_out =  inputs.cuda() +n.cuda()
            fake_latent = netEn(real_in)

            #print(fake_latent.shape)
            mu = np.random.uniform(-1, 1, fake_latent.shape)
            real_latent = torch.from_numpy(np.random.uniform(-1, 1, fake_latent.shape)).float().to(device)
            fake_out = netDe(fake_latent)
            fake_concat = fake_out

            eps2 = torch.from_numpy(np.tanh(mu)).float().to(device)


                # Train with fake image
            output = netD(fake_concat)
            output2 = netD2(fake_latent)

            fake_label = torch.from_numpy(np.zeros(output.shape)).float().to(device)
            fake_latent_label = torch.from_numpy(np.zeros(output2.shape)).float().to(device)
            
            eps = torch.from_numpy(np.random.uniform(-1, 1, fake_latent.shape)).float().to(device)

                        
            rec_output = netD(netDe(eps))
            errD_fake = GAN_loss(rec_output, fake_label)
            errD_fake2 = GAN_loss(output, fake_label)
            errD2_fake = GAN_loss(output2, fake_latent_label)


            real_concat = real_out
            output = netD(real_concat)
            output2 = netD2(real_latent)

            real_label = torch.from_numpy(np.ones(output.shape)).float().to(device)
            real_latent_label = torch.from_numpy(np.ones(output2.shape)).float().to(device)

            errD_real = GAN_loss(output, real_label)
            errD2_real = GAN_loss(output2, real_latent_label)
            errD = (errD_real + errD_fake) * 0.5
            errD2 = (errD2_real + errD2_fake) * 0.5
            totalerrD = errD + errD2

            netD2.zero_grad()
            netDe.zero_grad()
            netD.zero_grad()

            totalerrD.backward()

            optimizer_d.step()
            optimizer_d2.step()


            # Train classifier
            strong_output = netDS(netDe(eps))
            strong_real = netDS(fake_concat)
            errs1 = GAN_loss(strong_output, fake_label)
            errs2 = GAN_loss(strong_real, real_label)

            strongerr = 0.5 * (errs1 + errs2)
            
            netDS.zero_grad()
            strongerr.backward()

            optimizer_ds.step()
            ############################
            # (2) Update G network: maximize log(D(x, G(x, z))) - lambda1 * L1(y, G(x, z))
            ###########################
            
            rec_output = netD(netDe(eps2))
            fake_latent = netEn(real_in)
            output2 = netD2(fake_latent)
            fake_out = netDe(fake_latent)
            fake_concat = fake_out
            output = netD(fake_concat)
            
            real_label = torch.from_numpy(np.ones(output.shape)).float().to(device)
            real_latent_label = torch.from_numpy(np.ones(output2.shape)).float().to(device)
            
            errG2 = GAN_loss(rec_output, real_label)
            errR = L1_loss(real_out, fake_out) * lambda1
            errG = 10.0 * GAN_loss(output2, real_latent_label) + errG2 + errR
           
            netEn.zero_grad()
            netD2.zero_grad()
            netDe.zero_grad()
            netD.zero_grad()
            
            errG.backward()

            optimizer_en.step()
            optimizer_de.step()


            #print(errG2)
            loss_rec_G2.append(np.mean(errG2.item()))
            loss_rec_G.append(np.mean(np.mean(errG.item())) - np.mean(errG2.item()) - np.mean(errR.item()))
            loss_rec_D.append(np.mean(errD.item()))
            loss_rec_R.append(np.mean(errR.item()))
            loss_rec_D2.append(np.mean(errD2.item()))


            iter = iter + 1
            btic = time.time()

    # testing
    lbllist = []
    scorelist = []
    scorelist2 = []
    scorelist3 = []
    scorelist4 = []


    count = 0

    for batch_idx, (inputs, targets)  in enumerate(testloader):
        count = count+1
        output1=np.zeros(args.batch_size)
        output2=np.zeros(args.batch_size)
        output3=np.zeros(args.batch_size)
        output4=np.zeros(args.batch_size)


        real_in = inputs.cuda()
        n = torch.randn(inputs.shape, device=inputs.device)
        #print("input", inputs.shape)
        #l1 = enc(inputs + n)
        real_out =  inputs.cuda() +n.cuda()
        lbls_in = targets.cuda()


        outnn = (netDe(netEn(real_in)))         
        #print(outnn.shape)
        output1 = -1* torch.mean((outnn - real_out)**2,dim=[1,2])

        if opt.ntype >1:
            out_concat = outnn
            #print(netD(out_concat))
            output2 = torch.mean((netD(out_concat)), dim=[1,2])            

            
            output3 = torch.mean( netD(( real_in)),  dim=[1,2])     

            output4 = torch.mean(netD(netDe(netEn(real_out))),  dim=[1,2]) 
        #    print(output4)

        lbllist = lbllist+lbls_in.tolist()
        scorelist = scorelist + output1.tolist()
        scorelist2 = scorelist2 +output2.tolist()
        scorelist3 = scorelist3 +output3.tolist()
        scorelist4 = scorelist4 +output4.tolist()


    fpr, tpr, _ = roc_curve(lbllist, scorelist, pos_label=1)
    roc_auc1 = auc(fpr, tpr)
    # roc_auc1 = 0
    roc_auc2 = 0
    roc_auc3 = 0
    roc_auc4 = 0
    
    if int(opt.ntype) >1: #AE

        fpr, tpr, _ = roc_curve(lbllist, scorelist2, pos_label=1)
        roc_auc2 = auc(fpr, tpr)
        fpr, tpr, _ = roc_curve(lbllist, scorelist3, pos_label=1)
        roc_auc3 = auc(fpr, tpr)

        fpr, tpr, _ = roc_curve(lbllist, scorelist4, pos_label=1)
        roc_auc4 = auc(fpr, tpr)

    print(roc_auc1, roc_auc2,  roc_auc3, roc_auc4 )
    
    print(max([roc_auc1, roc_auc2,  roc_auc3, roc_auc4]))
    
    scores = scorelist
    labels = lbllist

    return auroc(scores, labels), aupr(scores, labels), fpr_at_95_tpr(scores, labels), detection_error(scores, labels), scores, labels



def train(args,trainloader,enc, dec,cl,disc_l,disc_v,
                    optimizer_en, optimizer_de,optimizer_c,optimizer_dl,optimizer_dv,optimizer_l2,
                    criterion_ae, criterion_ce, Tensor, epoch, use_cuda, channel, sequence_length):
    # switch to train mode
    enc.train()
    dec.train()
    cl.train()
    disc_l.train()
    disc_v.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()   


    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        # update class
        # same shape of encoder's output
        #l2 = torch.rand(len(inputs), 16, 1, device=inputs.device).uniform_(-1, 1)
        l2 = torch.rand(len(inputs), 32, device=inputs.device).uniform_(-1, 1)
    #    l2 = torch.tanh(l2)

        # print(l2.shape, targets.shape)
        
        n = torch.randn(len(inputs), channel, sequence_length, device=inputs.device)
        #print("input", inputs.shape)
        l1 = enc(inputs + n) # Use .clone() to create a new tensor

        logits_C_l1 = cl(dec(l1))
        logits_C_l2 = cl(dec(l2))

        #print(logits_C_l1.shape, dec(l1).shape)

        valid_logits_C_l1 = Tensor(logits_C_l1.shape[0], logits_C_l1.shape[1]).fill_(1.0).to(inputs.device)
        fake_logits_C_l2 = Tensor(logits_C_l2.shape[0], logits_C_l2.shape[1]).fill_(0.0).to(inputs.device)

        loss_cl_l1 = criterion_ce(logits_C_l1, valid_logits_C_l1)
        loss_cl_l2 = criterion_ce(logits_C_l2, fake_logits_C_l2)

        loss_cl = (loss_cl_l1 + loss_cl_l2) / 2

        cl.zero_grad()
        loss_cl.backward(retain_graph=True)
        optimizer_c.step()

        disc_l_l1 = l1.view(l1.size(0),-1)
        disc_l.zero_grad()
        logits_Dl_l1 = disc_l(disc_l_l1)
        logits_Dl_l2 = disc_l(l2)

        #logits_Dl_l1 = disc_l(l1)
        #logits_Dl_l2 = disc_l(l2)

        dl_logits_DL_l1 = Tensor(logits_Dl_l1.shape[0], logits_Dl_l1.shape[1]).fill_(0.0).to(inputs.device)
        dl_logits_DL_l2 = Tensor(logits_Dl_l2.shape[0], logits_Dl_l2.shape[1]).fill_(1.0).to(inputs.device)

        loss_dl_1 = criterion_ce(logits_Dl_l1, dl_logits_DL_l1)
        loss_dl_2 = criterion_ce(logits_Dl_l2, dl_logits_DL_l2)
        loss_dl = (loss_dl_1 + loss_dl_2) / 2

        loss_dl.backward(retain_graph=True)
        optimizer_dl.step()

        logits_Dv_X = disc_v(inputs)
        logits_Dv_l2 = disc_v(dec(l2))

        dv_logits_Dv_X = Tensor(logits_Dv_X.shape[0], logits_Dv_X.shape[1]).fill_(1.0).to(inputs.device)
        dv_logits_Dv_l2 = Tensor(logits_Dv_l2.shape[0], logits_Dv_l2.shape[1]).fill_(0.0).to(inputs.device)


        #optimizer_dl.zero_grad()
        #optimizer_dv.zero_grad()

        loss_dv_1 = criterion_ce(logits_Dv_X, dv_logits_Dv_X)
        loss_dv_2 = criterion_ce(logits_Dv_l2, dv_logits_Dv_l2)
        loss_dv = (loss_dv_1 + loss_dv_2) / 2
        
        #초기화, 역전파, 업데이트
        #loss_second=loss_dv+loss_dl     
        #loss_second.backward(retain_graph=True)

        disc_v.zero_grad()
        loss_dv.backward()
        optimizer_dv.step()
        #optimizer_dl.step()

        for i in range(5):
            logits_C_l2_mine = cl(dec(l2))
            zeros_logits_C_l2_mine = Tensor(logits_C_l2_mine.shape[0], logits_C_l2_mine.shape[1]).fill_(0.0).to(inputs.device)
            loss_C_l2_mine = criterion_ce(logits_C_l2_mine, zeros_logits_C_l2_mine)
            optimizer_l2.zero_grad()
            loss_C_l2_mine.backward()
            optimizer_l2.step()

        # update ae
        out_gv1 = disc_v(dec(l2))
        Xh = dec(l1).detach()  # Use .detach() to create a new tensor
        loss_mse = criterion_ae(Xh, inputs)

        ones_logits_Dl_l1 = Tensor(logits_Dl_l1.shape[0], logits_Dl_l1.shape[1]).fill_(1.0).to(inputs.device)

        loss_AE_l = criterion_ce(logits_Dl_l1, ones_logits_Dl_l1)



        logits_Dv_l2_mine = disc_v(dec(l2))
        ones_logits_Dv_l2_mine = Tensor(logits_Dv_l2_mine.shape[0], logits_Dv_l2_mine.shape[1]).fill_(1.0).to(inputs.device)
        loss_ae_v = criterion_ce(logits_Dv_l2_mine, ones_logits_Dv_l2_mine)

        #optimizer_en.zero_grad()
        #optimizer_de.zero_grad()

        loss_ae_all = 10 * loss_mse + loss_ae_v + loss_AE_l


        enc.zero_grad()
        dec.zero_grad()

        loss_ae_all.backward()

        optimizer_en.step()
        optimizer_de.step()

        losses.update(loss_ae_all.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    #     # plot progress
    #     bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f}  '.format(
    #                 batch=batch_idx + 1,
    #                 size=len(trainloader),
    #                 data=data_time.avg,
    #                 bt=batch_time.avg,
    #                 total=bar.elapsed_td,
    #                 eta=bar.eta_td,
    #                 loss=losses.avg,

    #                 )
    #     bar.next()
    # bar.finish()
    
    #save images during training time
    # if epoch % 5 == 0:
    #     recon = dec(enc(inputs))
    #     recon = recon.cpu().data
    #     inputs = inputs.cpu().data
    #     if not os.path.exists('./result/0000/train_dc_fake-1'):
    #         os.mkdir('./result/0000/train_dc_fake-1')
    #     if not os.path.exists('./result/0000/train_dc_real-1'):
    #         os.mkdir('./result/0000/train_dc_real-1')
    #     save_image(recon, './result/0000/train_dc_fake-1/fake_0{}.png'.format(epoch))
    #     save_image(inputs, './result/0000/train_dc_real-1/real_0{}.png'.format(epoch))  
    return losses.avg

def test(args, testloader, device, networks):
    global best_acc

    enc = networks[0].cuda()
    dec = networks[1].cuda()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    top1 = AverageMeter()


    # switch to evaluate mode
    enc.eval()
    dec.eval()
    # cl.eval()
    # disc_l.eval()
    # disc_v.eval()

    end = time.time()

    for batch_idx, (inputs, targets) in enumerate(testloader):
        # measure data loading time
        data_time.update(time.time() - end)

        #if use_cuda:
        inputs, targets = inputs.cuda(), targets.cuda()

        # compute output
        recon = dec(enc(inputs))       
        scores = torch.mean(torch.pow((inputs - recon), 2),dim=[1,2])

        #print(scores.cpu().detach().numpy(), targets.cpu().detach().numpy())
        
        #prec1 = roc_auc_score(targets.cpu().detach().numpy(), -scores.cpu().detach().numpy())
        prec1 = auroc( -scores.cpu().detach().numpy(), targets.cpu().detach().numpy())
        #print('\nBatch: {0:d} =====  auc:{1:.2f}' .format(batch_idx,prec1))
        top1.update(prec1, inputs.size(0))
 
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # # plot progress
        # bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | top1: {top1: .4f} '.format(
        #             batch=batch_idx + 1,
        #             size=len(testloader),
        #             data=data_time.avg,
        #             bt=batch_time.avg,
        #             total=bar.elapsed_td,
        #             eta=bar.eta_td,
        #             top1=top1.avg,
        #             )
    
    print(top1.avg)

    return top1.avg, aupr(-scores.cpu().detach().numpy(), targets.cpu().detach().numpy())

def save_checkpoint(state, is_best, checkpoint,filename):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']


if __name__ == "__main__":

    args = train_options()
    state = {k: v for k, v in args._get_kwargs()}
    
    # Setting for dataset
    data_type = args.selected_dataset
    if data_type == 'lapras': 
        args.timespan = 10000
        class_num = [0, 1, 2, 3, -1]
        layer_seq = [23, 13, 2]
        seq_length = 598
        channel = 7
    elif data_type == 'casas': 
        seq_length = 46
        args.aug_wise = 'Temporal2'
        class_num = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, -1]
        layer_seq = [23, 2, 1]
        channel = 37
    elif data_type == 'opportunity': 
        args.timespan = 1000
        class_num = [0, 1, 2, 3, 4, -1]        
        seq_length = 169
        layer_seq = [13, 13, 1]
        channel = 241
    elif data_type == 'aras_a': 
        args.timespan = 1000
        seq_length = 63
        channel = 19
        class_num = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, -1]
        layer_seq = [7, 3, 3]

    title_str =""
    if args.ntype == 1:
        title_str = "AE"
    elif args.ntype == 4:
        title_str = "OCGAN"

    store_path = 'result_files/' + str(title_str)+'_'+ data_type+'.xlsx'
    vis_path = 'figure/'+str(title_str)+'_ROC_'+data_type+'.png'
    vis_title ="ROC curves of "+str(title_str)+""

    final_auroc = []
    final_aupr  = []
    final_fpr   = []
    final_de    = []  

    y_onehot_test=[]
    y_score = []

    device = torch.device(args.device)
    num_classes, datalist, labellist = loading_data(data_type, args)

    seed_num = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]

    for args.one_class_idx in class_num:
        auroc_a = []
        aupr_a  = []
        fpr_a   = []
        de_a    = []

        testy_rs = []
        scores_rs = []



        for test_num in seed_num :
            # Set seed
            # ##### fix random seeds for reproducibility ########
            SEED = args.seed = test_num
            np.random.seed(SEED)
            random.seed(SEED)
            torch.manual_seed(SEED)
            #####################################################
    

            train_loader, test_loader, novel_class_idx = data_generator(args, datalist, labellist)

            print('Data loading done.')
            print("=" * 45)
            print("Dataset:",data_type)
            print("True Class:", args.one_class_idx)            
            print("Type:", args.ntype)
            print("Seed:", args.seed)
            print("=" * 45)
            print("creating model")
            title = 'Pytorch-OCGAN'

            # set model
            networks= set_network(args, device, False, channel, layer_seq)

            model = networks[0].cuda()

            print("==> training start")

            auroc_rs, aupr_rs, fpr_rs, de_re = 0, 0, 0, 0
        
            if args.ntype == 4:
                auroc_rs, aupr_rs, fpr_rs, de_re, scores, labels = trainadnov(args, train_loader, test_loader, device, networks)
            else:
                auroc_rs, aupr_rs, fpr_rs, de_re, scores, labels = trainAE(args, train_loader, test_loader, device,  networks)
            
            auroc_a.append(auroc_rs)     
            aupr_a.append(aupr_rs)   
            fpr_a.append(fpr_rs)
            de_a.append(de_re)

            testy_rs = testy_rs + labels
            scores_rs = scores_rs + scores
        
        final_auroc.append([np.mean(auroc_a), np.std(auroc_a)])
        final_aupr.append([np.mean(aupr_a), np.std(aupr_a)])
        final_fpr.append([np.mean(fpr_a), np.std(fpr_a)])
        final_de.append([np.mean(de_a), np.std(de_a)])

        print(testy_rs)  
        #print(scores_rs)
        # for visualization
        onehot_encoded = list()        
        label_binarizer = LabelBinarizer().fit(testy_rs)                    
        onehot_encoded = label_binarizer.transform(testy_rs)
        y_onehot_test.append(onehot_encoded)
        y_score.append(scores_rs)


    final_rs =[]
    for i in final_auroc:
        final_rs.append(i)
    for i in final_aupr:
        final_rs.append(i)                
    for i in final_fpr:
        final_rs.append(i)
    for i in final_de:
        final_rs.append(i)

    df = pd.DataFrame(final_rs, columns=['mean', 'std'])
    df.to_excel(store_path, sheet_name='the results')

    
    # visualization    
    fig, ax = plt.subplots(figsize=(6, 6))
    if(len(class_num)<=5):
        colors = cycle(["tomato", "darkorange", "gold", "darkseagreen","dodgerblue"])
    else:
        colors = cycle(["firebrick", "tomato", "sandybrown", "darkorange", "olive", "gold", 
                            "darkseagreen", "darkgreen", "dodgerblue", "royalblue","slategrey",
                            "slateblue", "mediumpurple","indigo", "orchid", "hotpink"])
        
    for class_id, color in zip(range(len(class_num)), colors):

        if class_num[class_id] != -1:
            RocCurveDisplay.from_predictions(
                y_onehot_test[class_id],
                y_score[class_id],
                pos_label=1,
                name=f"ROC curve for {(class_num[class_id]+1)}",
                color=color,
                ax=ax, 
            )
        else:
            RocCurveDisplay.from_predictions(
                y_onehot_test[class_id],
                y_score[class_id],
                pos_label=1,
                name=f"ROC curve for Multi",
                color="black",
                ax=ax, 

            )

                
    plt.axis("square")
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Chance Level (0.5)')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(vis_title)
    plt.legend()
    plt.show()
    plt.savefig(vis_path)

    print("Finished")
                    
    
    # print('  enc     Total params: %.2fM' % (sum(p.numel() for p in netEn.parameters())/1000000.0))
    # print('  dec     Total params: %.2fM' % (sum(p.numel() for p in netDe.parameters())/1000000.0))
    # print('  disc_v  Total params: %.2fM' % (sum(p.numel() for p in netDS.parameters())/1000000.0))
    # print('  disc_l  Total params: %.2fM' % (sum(p.numel() for p in netDL.parameters())/1000000.0))
    # print('  cl      Total params: %.2fM' % (sum(p.numel() for p in cl.parameters())/1000000.0))

    #Loss Loss Loss Loss Loss Loss Loss

    #test_acc, aupr_rs  = test(args, test_loader, device, networks)

    # criterion_ce = torch.nn.BCELoss(size_average=True).cuda()
    # criterion_ae = nn.MSELoss(size_average=True).cuda()

    # l2_int=torch.empty(size=(args.batch_size, 32), dtype=torch.float32) 
    # optimizer_en = optim.Adam(netEn.parameters(), lr=args.lr, betas=(0.9, 0.99))
    # optimizer_de = optim.Adam(netDe.parameters(), lr=args.lr, betas=(0.9, 0.99))
    # optimizer_dv = optim.Adam(netDS.parameters(), lr=args.lr, betas=(0.9, 0.99))
    # optimizer_dl = optim.Adam(netDL.parameters(), lr=args.lr, betas=(0.9, 0.99))
    # optimizer_c  = optim.Adam(cl.parameters(), lr=args.lr, betas=(0.9, 0.99))
    # optimizer_l2 = optim.Adam([{'params':l2_int}], lr=args.lr, betas=(0.9, 0.99))

    # best_acc = 0
    # # Train and val
    # for epoch in range(0, args.epochs):
    #     # adjust_learning_rate(optimizer, epoch)

    #     #print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))

    #     # # model = optimize_fore()
    #     #if epoch <= 1:
        
    #     if args.ntype == 1:

    #         train_loss_ae = train_ae(args, train_loader, netEn.cuda(), netDe.cuda(), optimizer_en, 
    #                                     optimizer_de,criterion_ae, epoch, device, channel, seq_length)
                                        
    #         test_acc, aupr_rs  = test(args, test_loader, netEn, netDe, cl, netDL, netDS, epoch, device)
    #         #test_acc = 0
    #         #print(f'Epoch:{epoch}, Loss:{train_loss_ae :.3f}, AUROC:{test_acc :.3f}, AUPR:{aupr_rs :.3f}')      
    #     else:
        
    #         train_loss = train(args, train_loader, netEn.cuda(), netDe.cuda(), cl.cuda(), netDL.cuda(), netDS.cuda(),
    #                                 optimizer_en, optimizer_de,optimizer_c,optimizer_dl,optimizer_dv,optimizer_l2,
    #                                 criterion_ae, criterion_ce, torch.cuda.FloatTensor, epoch, device, channel, seq_length
    #                             )
    #         test_acc, aupr_rs = test(args, test_loader, netEn, netDe, cl, netDL, netDS, epoch, device)    
    #         #print(f'Epoch:{epoch}, Loss:{train_loss :.3f}, AUROC:{test_acc :.3f}, AUPR:{aupr_rs :.3f}')      
        
    #     if test_acc > best_acc:
    #         best_acc = test_acc

    
    # print(f'Best Acc:{best_acc :.3f}') 
    
        
        

    # print('Training start')
    # # train networks based on opt.ntype(1 - AE 2 - ALOCC 3 - latentD  4 - adnov)
    # if args.ntype == 4:
    #     loss_vec = andgan.trainadnov(opt, train_data, val_data, ctx, networks)
    # elif opt.ntype == 2:
    #     loss_vec = andgan.traincvpr18(opt, train_data, val_data, ctx, networks)
    # elif opt.ntype == 1:
    #     loss_vec = andgan.trainAE(opt, train_data, val_data, ctx, networks)
    # plotloss(loss_vec, 'outputs/'+opt.expname+'_loss.png')






