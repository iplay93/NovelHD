
import sys
sys.path.append("..")
import os
import numpy as np
import torch
import torch.nn as nn
from models.loss import NTXentLoss, NTXentLoss_poly, SupConLoss, get_similarity_matrix, NT_xent, get_similarity_two_matrix, NT_xent_TF
from sklearn.metrics import f1_score, roc_auc_score
from tsaug import *
import torch.fft as fft
from data_preprocessing.augmentations import select_transformation

# shifted data transformations for negative pairs
 
def Trainer(model, model_optimizer, classifier, classifier_optimizer, 
            train_dl, device, logger, configs, experiment_log_dir, args, negative_list, negative_list_f, positive_list, positive_list_f):
    # Start training
    logger.debug("Training started ....")

    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, configs.num_epoch + 1):
        if epoch % 50 == 0 : 
            logger.debug(f'\nEpoch : {epoch}\n')
        # Train and validate
        train_loss, train_acc = model_train(epoch, logger, model, model_optimizer, classifier, 
                    classifier_optimizer, criterion, train_dl, configs, device, args, negative_list, negative_list_f, positive_list, positive_list_f)
        if epoch % 50 == 0 : 
            logger.debug(f'Train Loss   : {train_loss:.4f}\t | \tTrain Accuracy     : {train_acc:2.4f}\n')
    
    # Save the model
    os.makedirs(os.path.join(experiment_log_dir, "saved_models"), exist_ok=True)
    chkpoint = {'model_state_dict': model.state_dict(), 'classifier_state_dict': classifier.state_dict()}
    torch.save(chkpoint, os.path.join(experiment_log_dir, "saved_models", f'ckp_last.pt'))

    logger.debug("\n################## Training is Done! #########################")
    return model

def normalize(x, dim=1, eps=1e-8):
    return x / (x.norm(dim=dim, keepdim=True) + eps)

def generate_augmentation(args, positive_list, data,  labels, device, negative_list, postive_num):
    original_data = data
    aug_list =[]
    postive_num = postive_num

    # 1부터 시작 : 이미 data loader에서 하나를 함
    for positive_num in range(0, postive_num):
        normal_aug = select_transformation(positive_list[positive_num], original_data.shape[2])
        temp_data = torch.from_numpy(np.array(normal_aug.augment(original_data.permute(0, 2, 1).cpu().numpy()))).permute(0, 2, 1) 
        aug_list.append(temp_data.to(device))

    original_aug_list = aug_list.copy()

    for shifted_num in range(len(negative_list)):
                        #print(shifted_num, negative_list[shifted_num], data.shape)
        shifted_aug = select_transformation(negative_list[shifted_num], original_data.shape[2])
                    # (N, C, T) -> (N, T, C) -> (N, C, T)
        temp_data = torch.from_numpy(np.array(shifted_aug.augment(original_data.permute(0, 2, 1).cpu().numpy()))).permute(0, 2, 1)                        
        data = torch.cat((data, temp_data.to(device)), 0)
                        
        # for each positive_num
        for positive_num in range(0, postive_num):
            temp_aug1 = torch.from_numpy(np.array(shifted_aug.augment(original_aug_list[positive_num].permute(0, 2, 1).cpu().numpy()))).permute(0, 2, 1)
            aug_list[positive_num] = torch.cat((aug_list[positive_num], temp_aug1.to(device)), 0)

    shift_labels= torch.cat([torch.ones_like(labels) * k for k in range(len(negative_list)+1)], 0)  # B -> 2B (+1 for original data)
    shift_labels = shift_labels.repeat(postive_num+1)


    for positive_num in range(0, postive_num):
        data = torch.cat([data, aug_list[positive_num]], dim=0)

    return data, shift_labels


def model_train(epoch, logger, model, model_optimizer, classifier, classifier_optimizer, 
                criterion, train_loader, configs, device, args, negative_list, negative_list_f, positive_list, positive_list_f):
    
    assert len(args.ood_score) == 1  # assume single ood_score for simplicity
    ood_score = args.ood_score[0]
    
    
    if ood_score == 'simclr':
        assert args.K_shift == 1
    else:
        assert args.K_shift > 1
        assert args.K_shift_f > 1
    total_loss = []
    total_acc = []
    model.train()
    classifier.train()
    
    for batch_idx, (data, labels, aug1, _, _) in enumerate(train_loader):

        batch_size = data.shape[0]
        # send to device
        data, labels = data.float().to(device), labels.long().to(device) # data: [128, 1, 178], labels: [128]
        aug1 = aug1.float().to(device)  # aug1 = aug2 : [128, 1, 178]
        #data_f, aug1_f = data_f.float().to(device), aug1_f.float().to(device)  # aug1 = aug2 : [128, 1, 178]
        # (N, C, T)
        
        # optimizer
        model_optimizer.zero_grad()
        classifier_optimizer.zero_grad()
        
        #if configs.batch_size == batch_size
        original_data = data


        data, shift_labels = generate_augmentation(args, positive_list, original_data,  labels, device, negative_list, args.K_pos)
        data_f, shift_labels_f = generate_augmentation(args, positive_list_f, original_data,  labels, device, negative_list_f, args.K_pos_f)
            
        data_fft = torch.fft.fftn(data_f[0], norm="forward").reshape(1, data_f[0].shape[0], data_f[0].shape[1])

        for i in range(1, len(data_f)):
            data_fft = torch.cat([data_fft, torch.fft.fftn(data_f[i], norm="forward").reshape(1, data_f[0].shape[0], data_f[0].shape[1])], 0)            

            #sensor_pair_f = torch.cat([normal_fft, aug_fft], dim=0)   
            
            #data_f = fft.fftn(data_f, norm="backward").abs().to(device)

        data_f = data_fft

        assert data.size(0) == shift_labels.size(0)
        assert data_f.size(0) == shift_labels_f.size(0)            

        # original data and augmented data 
        if args.vis:
            h_t, z_t, s_t, h_f, z_f, s_f, _, _  = model(data, data_f)
        else:
            h_t, z_t, s_t, h_f, z_f, s_f = model(data, data_f)        
            
            #Initialize loss
        loss_sim, loss_shift, loss_t = torch.tensor(0), torch.tensor(0), torch.tensor(0)
        loss_sim_f, loss_shift_f, loss_f = torch.tensor(0), torch.tensor(0), torch.tensor(0)
        l_TF = torch.tensor(0)

            # for constructing loss functions
            # For temporal contrastive          
        if ood_score in ['T', 'TCON' ,'TCLS' ,'CON' ,'CLS' ,'NovelHD_TF', 'NovelHD']:
            simclr = normalize(z_t)  # normalize
            sim_matrix = get_similarity_matrix(simclr)            
            loss_sim = NT_xent(sim_matrix, temperature = args.temp, chunk = args.K_pos+1) #* sim_lambda
                
            loss_shift = criterion(s_t, shift_labels)
            #print(data.shape, z_t.shape, simclr.shape)
            loss_t = loss_sim + loss_shift

            
        # combined two latent space
        #sim_two_matrix = get_similarity_two_matrix(simclr, simclr_f)

            # B = simclr.size(0) // (args.K_pos+1)
            # B_f = simclr_f.size(0) // (args.K_pos_f+1)
            # for mul in range(0, (args.K_pos+1)):
            #     l_TF = l_TF + nt_xent_criterion(simclr[mul*B:(mul)*B+batch_size], simclr_f[mul* B_f:(mul)* B_f+batch_size])
            #l_TF = NT_xent_TF(sim_two_matrix, temperature=0.5)


            # Select loss according to ood_score
            #loss
            
            B_t = simclr.size(0) // (args.K_pos+1)
            B_f = s_f.size(0) // (args.K_pos_f+1)

             # For frequency contrastive
            
        if ood_score in ['F', 'FCON' ,'FCLS' ,'CON' ,'CLS' ,'NovelHD_TF', 'NovelHD']:

            simclr_f = normalize(z_f)  # normalize
            sim_matrix_f = get_similarity_matrix(simclr_f)            
            loss_sim_f = NT_xent(sim_matrix_f, temperature = args.temp, chunk = args.K_pos_f+1) #* sim_lambda_f 
                
            loss_shift_f = criterion(s_f, shift_labels_f)
                
            loss_f = loss_sim_f + loss_shift_f
            

        if ood_score == 'T':
            loss = loss_t   
        elif ood_score == 'F':
            loss = loss_f                
        elif ood_score == 'TCON' or ood_score == 'simclr' :
            loss = loss_sim
        elif ood_score == 'TCLS':
            loss = loss_shift
        elif ood_score == 'FCON':
            loss =  loss_sim_f 
        elif ood_score == 'FCLS':
            loss = loss_shift_f
        elif ood_score == 'NovelHD':
            loss = args.lam_a *(loss_sim + loss_sim_f) + loss_shift + loss_shift_f
        elif ood_score == 'CON':
            loss = loss_sim + loss_sim_f
        elif ood_score == 'CLS':
            loss = loss_shift + loss_shift_f
        elif ood_score == 'CLAN':
            loss = loss_t + loss_sim_f # + l_TF
        else:
            raise ValueError() 
            
        if epoch % 20 == 0 : 
            logger.debug(f'Temporal: {loss_sim.item():.4f}, {loss_shift.item():.4f}, {loss_t.item():.4f}')
            logger.debug(f'Frequency: {loss_sim_f.item():.4f},{loss_shift_f.item():.4f}, {loss_f.item():.4f}')
            logger.debug(f'TF: {loss.item():.4f}')


        total_loss.append(loss.item())
        loss.backward()
        model_optimizer.step()


        # if ood_score != 'simclr' :
        #     """Post-processing stuffs"""
        #         penul_1 = h_t[:batch_size]
        #         penul_2 = h_t[2*batch_size:3*batch_size]
        #         outputs_penul = torch.cat([penul_1, penul_2]) 

        #         ### Linear evaluation ###
        #         outputs_linear_eval = model.linear(outputs_penul.detach())
        #         loss_linear = criterion(outputs_linear_eval, labels.repeat(2)) 
                
        #         classifier_optimizer.zero_grad()
        #         loss_linear.backward()
        #         classifier_optimizer.step()

    total_loss = torch.tensor(total_loss).mean()

    total_acc = 0

    return total_loss, total_acc