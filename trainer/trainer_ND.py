import os
import sys
sys.path.append("..")
import numpy as np
import torch
import torch.nn as nn
from models.loss import NTXentLoss, SupConLoss, get_similarity_matrix, NT_xent, get_similarity_two_matrix, NT_xent_TF
from sklearn.metrics import f1_score, roc_auc_score
from tsaug import *
import torch.fft as fft


# shifted data transformations for negative pairs
shifted_aug = (Drift(max_drift=0.7, n_drift_points=5))

def Trainer(model, model_optimizer, classifier, classifier_optimizer, 
            train_dl, device, logger, configs, experiment_log_dir, args):
    # Start training
    logger.debug("Training started ....")

    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, configs.num_epoch + 1):
        if epoch % 20 == 0 : 
            logger.debug(f'\nEpoch : {epoch}\n')
        # Train and validate
        train_loss, train_acc = model_train(epoch, logger, model, model_optimizer, classifier, 
                                            classifier_optimizer, criterion, train_dl, configs, device, args)
        if epoch % 20 == 0 : 
            logger.debug(f'Train Loss   : {train_loss:.4f}\t | \tTrain Accuracy     : {train_acc:2.4f}\n')
    
    # Save the model
    os.makedirs(os.path.join(experiment_log_dir, "saved_models"), exist_ok=True)
    chkpoint = {'model_state_dict': model.state_dict(), 'classifier_state_dict': classifier.state_dict()}
    torch.save(chkpoint, os.path.join(experiment_log_dir, "saved_models", f'ckp_last.pt'))

    logger.debug("\n################## Training is Done! #########################")
    return model

def normalize(x, dim=1, eps=1e-8):
    return x / (x.norm(dim=dim, keepdim=True) + eps)

def model_train(epoch, logger, model, model_optimizer, classifier, classifier_optimizer, 
                criterion, train_loader, configs, device, args):
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
        
        if configs.batch_size == batch_size:

            # adding shifted transformation
            for k in range(data.size(0)):
                #print("temp_before", data[k].shape)
                if args.aug_wise == 'Temporal':
                    temp_data = torch.from_numpy(shifted_aug.augment(np.reshape(data[k].cpu().numpy(),(1, data[k].shape[1],-1)))).permute(0, 2, 1)
                    temp_aug1 = torch.from_numpy(shifted_aug.augment(np.reshape(aug1[k].cpu().numpy(),(1, aug1[k].shape[1],-1)))).permute(0, 2, 1)
                elif args.aug_wise == 'Sensor':
                    temp_data = torch.from_numpy(shifted_aug.augment(np.reshape(data[k].cpu().numpy(),(1, data[k].shape[0],-1))))
                    temp_aug1 = torch.from_numpy(shifted_aug.augment(np.reshape(aug1[k].cpu().numpy(),(1, aug1[k].shape[0],-1))))


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
        
            
            # for constructing loss functions
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
            
            # combined two latent space
            sim_two_matrix = get_similarity_two_matrix(simclr, simclr_f)
            #nt_xent_criterion = NTXentLoss(device, configs.batch_size, configs.Context_Cont.temperature,
                                           #configs.Context_Cont.use_cosine_similarity)
                        
            #l_TF = nt_xent_criterion(z_t, z_f)
            l_TF = NT_xent_TF(sim_two_matrix, temperature=0.5)

            if epoch % 20 == 0 : 
                logger.debug(f'Temporal: {loss_sim.item():.4f}, {loss_shift.item():.4f}, {loss_t.item():.4f}')
                logger.debug(f'Frequency: {loss_sim_f.item():.4f}, {loss_shift_f.item():.4f}, {loss_f.item():.4f}')
                logger.debug(f'TF: {l_TF.item():.4f}')

            # Select loss according to ood_score
            #loss
            assert len(args.ood_score) == 1  # assume single ood_score for simplicity
            ood_score = args.ood_score[0]

            if ood_score == 'T':
                loss = loss_t   
            elif ood_score == 'TCON':
                loss = loss_sim
            elif ood_score == 'TCLS':
                loss = loss_shift
            elif ood_score == 'FCON':
                loss = loss_t + loss_sim_f
            elif ood_score == 'FCLS':
                loss = loss_t + loss_shift_f
            elif ood_score == 'NovelHD':
                loss = loss_t + loss_f
            elif ood_score == 'NovelHD_TF':
                loss = (loss_t + loss_f) + l_TF
            else:
                raise ValueError() 
            

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

    total_loss = torch.tensor(total_loss).mean()

    total_acc = 0

    return total_loss, total_acc