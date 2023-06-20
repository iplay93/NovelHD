import os
import sys

sys.path.append("..")
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


from models.loss import NTXentLoss, SupConLoss, get_similarity_matrix, NT_xent
from sklearn.metrics import f1_score, roc_auc_score
from tsaug import *
import torch.fft as fft


def Trainer(model, model_optimizer, classifier, classifier_optimizer, train_dl, valid_dl, test_dl, device, logger, configs, experiment_log_dir, training_mode):
    # Start training
    logger.debug("Training started ....")

    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model_optimizer, 'min')

    for epoch in range(1, configs.num_epoch + 1):
        # Train and validate
        train_loss, train_acc = model_train(model, model_optimizer, classifier, 
                    classifier_optimizer, criterion, train_dl, configs, device, training_mode)
        valid_loss, valid_acc, valid_f1, _, _, _ = model_evaluate(model, classifier, 
                                                    valid_dl, device, training_mode)
        scheduler.step(valid_loss)
      #  logger.debug(f'\nEpoch : {epoch}\n'
      #               f'Train Loss     : {train_loss:.4f}\t | \tTrain Accuracy     : {train_acc:2.4f}\n'
      #               f'Valid Loss     : {valid_loss:.4f}\t | \tValid Accuracy     : {valid_acc:2.4f} | \tValid F1-score      : {valid_f1:0.4f}')

    os.makedirs(os.path.join(experiment_log_dir, "saved_models"), exist_ok=True)
    chkpoint = {'model_state_dict': model.state_dict(), 'classifier_state_dict': classifier.state_dict()}
    torch.save(chkpoint, os.path.join(experiment_log_dir, "saved_models", f'ckp_last.pt'))

    logger.debug("\n################## Training is Done! #########################")
    return model

def normalize(x, dim=1, eps=1e-8):
    return x / (x.norm(dim=dim, keepdim=True) + eps)


def model_train(model, model_optimizer, classifier, classifier_optimizer, criterion, train_loader, configs, device, training_mode):
    total_loss = []
    total_acc = []
    model.train()
    classifier.train()
    
    for batch_idx, (data, labels, aug1, data_f, aug1_f) in enumerate(train_loader):

        batch_size = data.shape[0]
        # send to device
        data, labels = data.float().to(device), labels.long().to(device) # data: [128, 1, 178], labels: [128]
        aug1 = aug1.float().to(device)  # aug1 = aug2 : [128, 1, 178]
        data_f, aug1_f = data_f.float().to(device), aug1_f.float().to(device)  # aug1 = aug2 : [128, 1, 178]
 
        # optimizer
        model_optimizer.zero_grad()
        classifier_optimizer.zero_grad()

        # do model
        sensor_pair = torch.cat([data, aug1], dim=0)             
        sensor_pair_f = torch.cat([data_f, aug1_f], dim=0)   
        # original data and augmented data 
        h_t, z_t, s_t, h_f, z_f, s_f  = model(sensor_pair, sensor_pair_f)            
        shift_labels = torch.cat([torch.ones_like(labels) * k for k in range(2)], 0)   

        # else:
        #     h_t, z_t, s_t, h_f, z_f, s_f  = model(data, data_f)
        #     h_t_aug, z_t_aug, s_t_aug, h_f_aug, z_f_aug, s_f_aug = model(aug1, aug1_f)
        #     fea_concat = torch.cat((z_t, z_f), dim=1)
        #     predictions = classifier(fea_concat)

        # compute loss

        loss = criterion(s_t, shift_labels)
        total_acc.append(shift_labels.eq(s_t.detach().argmax(dim=1)).float().mean())

        total_loss.append(loss.item())
        loss.backward()
        model_optimizer.step()
            
        # elif training_mode != "self_supervised" and training_mode != "novelty_detection":# supervised training or fine tuining
        #     loss = criterion(predictions, labels)
        #     total_acc.append(labels.eq(predictions.detach().argmax(dim=1)).float().mean())

        #     total_loss.append(loss.item())
        #     loss.backward()
        #     model_optimizer.step()
        #     classifier_optimizer.step()

    total_loss = torch.tensor(total_loss).mean()
    total_acc = torch.tensor(total_acc).mean()

    return total_loss, total_acc

def model_evaluate(model, classifier, test_dl, device, training_mode):
    model.eval()
    classifier.eval()

    total_loss = []
    total_acc = []
    total_f1 = []
    total_auroc = []

    criterion = nn.CrossEntropyLoss()
    outs = np.array([])
    trgs = np.array([])

    with torch.no_grad():
        for (data, labels, aug1, data_f, aug1_f) in test_dl:
        # send to device
            data, labels = data.float().to(device), labels.long().to(device) # data: [128, 1, 178], labels: [128]
            aug1 = aug1.float().to(device)  # aug1 = aug2 : [128, 1, 178]
            data_f, aug1_f = data_f.float().to(device), aug1_f.float().to(device)  # aug1 = aug2 : [128, 1, 178]

            sensor_pair = torch.cat([data, aug1], dim=0)             
            sensor_pair_f = torch.cat([data_f, aug1_f], dim=0)   
            # original data and augmented data 
            h_t, z_t, s_t, h_f, z_f, s_f  = model(sensor_pair, sensor_pair_f)            
            shift_labels = torch.cat([torch.ones_like(labels) * k for k in range(2)], 0)    

            # else:
            #     h_t, z_t, h_f, z_f = model(data, data_f)
            #     fea_concat = torch.cat((z_t, z_f), dim=1)                

            # # compute loss
            # if training_mode != "self_supervised" and training_mode != "novelty_detection" and training_mode != "ood_ness":
            #     predictions = classifier(fea_concat)
            #     loss = criterion(predictions, labels)
            #     total_acc.append(labels.eq(predictions.detach().argmax(dim=1)).float().mean())
            #     #print(labels, predictions.detach().argmax(dim=1))
            #     total_f1.append(f1_score(labels.cpu(), predictions.detach().argmax(dim=1).cpu(), average='macro'))
            #     total_loss.append(loss.item())
                # pred = predictions.max(1, keepdim=True)[1]  # get the index of the max log-probability
                # outs = np.append(outs, pred.cpu().numpy())
                # trgs = np.append(trgs, labels.data.cpu().numpy())

            loss = criterion(s_t , shift_labels)
            total_loss.append(loss.item())
            softmax = nn.Softmax(dim=1)
            s_t = softmax(s_t)

            total_acc.append(shift_labels.eq(s_t.detach().argmax(dim=1)).float().mean())
            total_f1.append(f1_score(shift_labels.cpu(), s_t.detach().argmax(dim=1).cpu(), average='macro'))
            s_t_p= s_t[:, 1]

            total_auroc.append(roc_auc_score(shift_labels.cpu(), s_t_p.detach().cpu()))
            #print(labels, predictions.detach().argmax(dim=1))
            

            pred = s_t.max(1, keepdim=True)[1]  # get the index of the max log-probability
            outs = np.append(outs, pred.cpu().numpy())
            trgs = np.append(trgs, shift_labels.data.cpu().numpy())



    total_loss = torch.tensor(total_loss).mean()  # average loss
    total_acc = torch.tensor(total_acc).mean()  # average acc
    total_f1  = torch.tensor(total_f1).mean() # average f1
    total_auroc = torch.tensor(total_auroc).mean() # average auroc

    return total_loss, total_acc, total_f1, total_auroc, outs, trgs