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


# data augmentation for negative pairs
my_aug = (Dropout( p=0.1,fill=0)) 
#my_aug = (TimeWarp(n_speed_change=5, max_speed_ratio=3))

def Trainer(model, model_optimizer, classifier, classifier_optimizer, train_dl, valid_dl, test_dl, device, logger, configs, experiment_log_dir, training_mode):
    # Start training
    logger.debug("Training started ....")

    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model_optimizer, 'min')

    for epoch in range(1, configs.num_epoch + 1):
        # Train and validate
        train_loss, train_acc = model_train(model, model_optimizer, classifier, classifier_optimizer, criterion, train_dl, configs, device, training_mode)
        valid_loss, valid_acc, valid_f1, _, _, _ = model_evaluate(model, classifier, valid_dl, device, training_mode)
        if training_mode != 'self_supervised' and training_mode!="novelty_detection":  # use scheduler in all other modes.
            scheduler.step(valid_loss)

        logger.debug(f'\nEpoch : {epoch}\n'
                     f'Train Loss     : {train_loss:.4f}\t | \tTrain Accuracy     : {train_acc:2.4f}\n'
                     f'Valid Loss     : {valid_loss:.4f}\t | \tValid Accuracy     : {valid_acc:2.4f} | \tValid F1-score      : {valid_f1:0.4f}')
    if training_mode == "novelty_detection":
        os.makedirs(os.path.join(experiment_log_dir, "saved_models"), exist_ok=True)
        chkpoint = {'model_state_dict': model.state_dict(), 'classifier_state_dict': classifier.state_dict()}
        torch.save(chkpoint, os.path.join(experiment_log_dir, "saved_models", f'ckp_last.pt'))
    else:
        os.makedirs(os.path.join(experiment_log_dir, "saved_models"), exist_ok=True)
        chkpoint = {'model_state_dict': model.state_dict(), 'classifier_state_dict': classifier.state_dict()}
        torch.save(chkpoint, os.path.join(experiment_log_dir, "saved_models", f'ckp_last.pt'))

    if training_mode != "self_supervised" and training_mode != "novelty_detection":  # no need to run the evaluation for self-supervised mode.
        # evaluate on the test set
        logger.debug('\nEvaluate on the Test set:')
        test_loss, test_acc, test_f1, _, _, _ = model_evaluate(model, classifier, test_dl, device, training_mode)
        logger.debug(f'Test loss      :{test_loss:0.4f}\t | Test Accuracy      : {test_acc:0.4f}\t | Test F1-score      : {test_f1:0.4f}')

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


        if training_mode == "self_supervised" and configs.batch_size == batch_size:
   
            h_t, z_t, s_t, h_f, z_f, s_f  = model(data, data_f)
            h_t_aug, z_t_aug, s_t_aug, h_f_aug, z_f_aug, s_f_aug = model(aug1, aug1_f)

            # normalize projection feature vectors
            #zis = temp_cont_lstm_feat1 
            #zjs = temp_cont_lstm_feat2 
        
        elif training_mode =="novelty_detection" and configs.batch_size == batch_size:

            # adding shifted transformation
            for k in range(data.size(0)):
                temp_data = torch.from_numpy(np.array([my_aug.augment(data[k].cpu().numpy())]))
                temp_aug1 = torch.from_numpy(np.array([my_aug.augment(aug1[k].cpu().numpy())]))

                data = torch.cat((data, temp_data.to(device)), 0)
                aug1 = torch.cat((aug1, temp_aug1.to(device)), 0)
                data_f = torch.cat((data_f, fft.fft(temp_data).abs().to(device)), 0)
                aug1_f = torch.cat((aug1_f, fft.fft(temp_aug1).abs().to(device)), 0)

            shift_labels = torch.cat([torch.ones_like(labels) * k for k in range(2)], 0)  # B -> 2B
            shift_labels = shift_labels.repeat(2)
            
            sensor_pair = torch.cat([data, aug1], dim=0) # B -> 4B       
            sensor_pair_f = torch.cat([data_f, aug1_f], dim=0) 
  
            # original data and augmented data 
            h_t, z_t, s_t, h_f, z_f, s_f  = model(sensor_pair, sensor_pair_f)
            
            #h_t_aug, z_t_aug, s_t_aug, h_f_aug, z_f_aug, s_f_aug = model(aug1, aug1_f)
        elif training_mode == "ood_ness" and configs.batch_size == batch_size:
            sensor_pair = torch.cat([data, aug1], dim=0)             
            sensor_pair_f = torch.cat([data_f, aug1_f], dim=0)   
            # original data and augmented data 
            h_t, z_t, s_t, h_f, z_f, s_f  = model(sensor_pair, sensor_pair_f)            
            shift_labels = torch.cat([torch.ones_like(labels) * k for k in range(2)], 0)   

        else:
            h_t, z_t, s_t, h_f, z_f, s_f  = model(data, data_f)
            h_t_aug, z_t_aug, s_t_aug, h_f_aug, z_f_aug, s_f_aug = model(aug1, aug1_f)
            fea_concat = torch.cat((z_t, z_f), dim=1)
            predictions = classifier(fea_concat)

        # compute loss
        if training_mode == "self_supervised" and configs.batch_size == batch_size:
            lambda1 = 0
            lambda2 = 0.7
            nt_xent_criterion = NTXentLoss(device, configs.batch_size, configs.Context_Cont.temperature,
                                           configs.Context_Cont.use_cosine_similarity)
            #loss = (temp_cont_loss1 + temp_cont_loss2) * lambda1 +  nt_xent_criterion(zis, zjs) * lambda2
            loss_t = nt_xent_criterion(h_t, h_t_aug)
            loss_f = nt_xent_criterion(h_f, h_f_aug)
            l_TF = nt_xent_criterion(z_t, z_f) # this is the initial version of TF loss
            
            l_1, l_2, l_3 = nt_xent_criterion(z_t, z_f_aug), nt_xent_criterion(z_t_aug, z_f), nt_xent_criterion(z_t_aug, z_f_aug)
            loss_c = (1 + l_TF - l_1) + (1 + l_TF - l_2) + (1 + l_TF - l_3)
            
            lam = 0.2
            loss = lam*(loss_t + loss_f) + l_TF

            # for supervised CL
            #representations = torch.cat([zis.unsqueeze(1), zjs.unsqueeze(1)], dim=1)
            #nt_xent_criterion = SupConLoss()
            #loss = (temp_cont_loss1 + temp_cont_loss2) * lambda1 +  nt_xent_criterion(representations, labels) * lambda2
            #loss = (temp_cont_loss1 + temp_cont_loss2) * lambda1 +  nt_xent_criterion(representations) * lambda2

            total_loss.append(loss.item())
            loss.backward()
            model_optimizer.step()

        elif training_mode =="novelty_detection" and configs.batch_size == batch_size:

            # For temporal contrastive
            sim_lambda = 0.1          

            simclr = normalize(z_t)  # normalize
            sim_matrix = get_similarity_matrix(simclr)            
            loss_sim = NT_xent(sim_matrix, temperature=0.5) * sim_lambda
            
            loss_shift = criterion(s_t, shift_labels)

            loss_t = loss_sim + loss_shift

            # For frequency contrastive
            sim_lambda_f = 0.1
            simclr_f = normalize(z_f)  # normalize
            sim_matrix_f = get_similarity_matrix(simclr_f)            
            loss_sim_f = NT_xent(sim_matrix_f, temperature=0.5) * sim_lambda_f 
            
            loss_shift_f = criterion(s_f, shift_labels)
            loss_f = loss_sim_f + loss_shift_f
            #loss_f = loss_shift_f
            nt_xent_criterion = NTXentLoss(device, configs.batch_size, configs.Context_Cont.temperature,
                                           configs.Context_Cont.use_cosine_similarity)
                        
            l_TF = nt_xent_criterion(z_t, z_f)

            print(f'Temporal: {loss_sim.item():.4f}, {loss_shift.item():.4f}, {loss_t.item():.4f}')
            print(f'Frequency: {loss_sim_f.item():.4f}, {loss_shift_f.item():.4f}, {loss_f.item():.4f}')
            print("TF", l_TF.item())

            lam = 0.01
            #loss
            loss = loss_t 
            #+ lam * loss_f            
            #loss = loss_t + loss_f
            
            #loss = (loss_t + loss_f) + l_TF
            total_loss.append(loss.item())
            loss.backward()
            model_optimizer.step()

            """Post-processing stuffs"""
            penul_1 = h_t[:batch_size]
            penul_2 = h_t[2*batch_size:3*batch_size]
            outputs_penul = torch.cat([penul_1, penul_2]) 

            ### Linear evaluation ###
            outputs_linear_eval = model.linear(outputs_penul.detach())
            loss_linear = criterion(outputs_linear_eval, labels.repeat(2)) 
            
            classifier_optimizer.zero_grad()
            loss_linear.backward()
            classifier_optimizer.step()

        elif training_mode == "ood_ness" and configs.batch_size == batch_size:
            loss = criterion(s_t, shift_labels)
            total_acc.append(shift_labels.eq(s_t.detach().argmax(dim=1)).float().mean())

            total_loss.append(loss.item())
            loss.backward()
            model_optimizer.step()
            
        elif training_mode != "self_supervised" and training_mode != "novelty_detection":# supervised training or fine tuining
            loss = criterion(predictions, labels)
            total_acc.append(labels.eq(predictions.detach().argmax(dim=1)).float().mean())

            total_loss.append(loss.item())
            loss.backward()
            model_optimizer.step()
            classifier_optimizer.step()

    total_loss = torch.tensor(total_loss).mean()

    if training_mode == "self_supervised" or  training_mode == "novelty_detection":
        total_acc = 0
    else:
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

            if training_mode == "self_supervised" or training_mode =="novelty_detection":
                pass
            elif training_mode == "ood_ness":
                sensor_pair = torch.cat([data, aug1], dim=0)             
                sensor_pair_f = torch.cat([data_f, aug1_f], dim=0)   
                # original data and augmented data 
                h_t, z_t, s_t, h_f, z_f, s_f  = model(sensor_pair, sensor_pair_f)            
                shift_labels = torch.cat([torch.ones_like(labels) * k for k in range(2)], 0)    

            else:
                h_t, z_t, h_f, z_f = model(data, data_f)
                fea_concat = torch.cat((z_t, z_f), dim=1)                

            # compute loss
            if training_mode != "self_supervised" and training_mode != "novelty_detection" and training_mode != "ood_ness":
                predictions = classifier(fea_concat)
                loss = criterion(predictions, labels)
                total_acc.append(labels.eq(predictions.detach().argmax(dim=1)).float().mean())
                #print(labels, predictions.detach().argmax(dim=1))
                total_f1.append(f1_score(labels.cpu(), predictions.detach().argmax(dim=1).cpu(), average='macro'))
                total_loss.append(loss.item())

            elif training_mode == "ood_ness":
                loss = criterion(s_t , shift_labels)
                total_acc.append(shift_labels.eq(s_t.detach().argmax(dim=1)).float().mean())
                total_auroc.append(roc_auc_score(shift_labels.cpu(), s_t.detach().argmax(dim=1).cpu()))
                #print(labels, predictions.detach().argmax(dim=1))
                total_f1.append(f1_score(shift_labels.cpu(), s_t.detach().argmax(dim=1).cpu(), average='macro'))
                total_loss.append(loss.item())
                pred = s_t.max(1, keepdim=True)[1]  # get the index of the max log-probability
                outs = np.append(outs, pred.cpu().numpy())
                trgs = np.append(trgs, shift_labels.data.cpu().numpy())

            if training_mode != "self_supervised" and training_mode != "novelty_detection" and training_mode != "ood_ness":
                pred = predictions.max(1, keepdim=True)[1]  # get the index of the max log-probability
                outs = np.append(outs, pred.cpu().numpy())
                trgs = np.append(trgs, labels.data.cpu().numpy())

    if training_mode != "self_supervised" and training_mode != "novelty_detection":
        total_loss = torch.tensor(total_loss).mean()  # average loss
    else:
        total_loss = 0

    if training_mode == "self_supervised" or training_mode == "novelty_detection":
        total_acc = 0
        total_f1  = 0
        return total_loss, total_acc, total_f1, total_auroc, [], []
    else:
        total_acc = torch.tensor(total_acc).mean()  # average acc
        total_f1  = torch.tensor(total_f1).mean() # average f1
        total_auroc = torch.tensor(total_auroc).mean() # average auroc

    return total_loss, total_acc, total_f1, total_auroc, outs, trgs