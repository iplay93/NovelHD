
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
from utils.utils import plot_tsne_with_shift_labels
# shifted data transformations for negative pairs
import time

def Trainer(model, model_optimizer, classifier, classifier_optimizer, 
            train_dl, device, logger, configs, experiment_log_dir, args, negative_list, negative_list_f, positive_list, positive_list_f):
    # Start training
    logger.debug("Training started ....")

    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, configs.num_epoch + 1):
        if epoch % 50 == 0 : 
            logger.debug(f'\nEpoch : {epoch}\n')
        # Train and validate
        total_loss, average_time_per_batch, average_memory_per_batch = model_train(epoch, logger, model, model_optimizer, classifier, 
                    classifier_optimizer, criterion, train_dl, configs, device, args, negative_list, negative_list_f, positive_list, positive_list_f)
        # Store results for this epoch
        results.append({
            'epoch': epoch,
            'total_loss': total_loss,
            'average_time_per_batch': average_time_per_batch,
            'average_memory_per_batch': average_memory_per_batch
        })

        # if epoch % 50 == 0 : 
        #     logger.debug(f'Train Loss   : {train_loss:.4f}\t | \tTrain Accuracy     : {train_acc:2.4f}\n')
    
    # Convert results to a pandas DataFrame
    df_results = pd.DataFrame(results)

    # Save to an Excel file
    df_results.to_excel('CLAN_efficiency.xlsx', index=False)

    print("Results saved to 'CLAN_efficiency.xlsx'")

    all_data = []
    all_label = []

    # Iterate through the DataLoader and collect all data
    for batch_idx, (data, labels, aug1, _, _) in enumerate(train_dl):
        all_data.append(data.float().to(device))
        all_label.append(labels.float().to(device))  

    all_data = torch.cat(all_data, dim=0).to(device)  # Shape: [N, 1, 178], where N is the total number of samples
    all_label = torch.cat(all_label, dim=0).to(device)  # Shape: [N]

    data, shift_labels = generate_augmentation(args, all_data,  all_label, device,  positive_list, negative_list)
    h_t, z_t, s_t, _, _, _  = model(data, data)

    combined_features = torch.cat((s_t, z_t), dim=1)

        # h_t와 h_f를 결합한 특성을 사용하여 t-SNE 시각화 생성
    plot_tsne_with_shift_labels(combined_features, shift_labels, 'tsne_combined.png', negative_list = negative_list, title='t-SNE of combined', perplexity=30)

    plot_tsne_with_shift_labels(z_t, shift_labels, 'tsne_z_t.png', title='t-SNE of s_t', negative_list = negative_list, perplexity=30)

        # z_t와 z_f를 사용한 시각화 생성 예시
    plot_tsne_with_shift_labels(s_t, shift_labels, 'tsne_s_t.png', title='t-SNE of s_t', negative_list = negative_list, perplexity=30)

    # Save the model
    os.makedirs(os.path.join(experiment_log_dir, "saved_models"), exist_ok=True)
    chkpoint = {'model_state_dict': model.state_dict(), 'classifier_state_dict': classifier.state_dict()}
    torch.save(chkpoint, os.path.join(experiment_log_dir, "saved_models", f'ckp_last.pt'))

    logger.debug("\n################## Training is Done! #########################")
    return model

def normalize(x, dim=1, eps=1e-8):
    return x / (x.norm(dim=dim, keepdim=True) + eps)

def generate_augmentation(args, data,  labels, device,  positive_list, negative_list):
    original_data = data
    aug_list =[]

    for positive_num in range(0, len(positive_list)):
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
        for positive_num in range(0, len(positive_list)):
            temp_aug1 = torch.from_numpy(np.array(shifted_aug.augment(original_aug_list[positive_num].permute(0, 2, 1).cpu().numpy()))).permute(0, 2, 1)
            aug_list[positive_num] = torch.cat((aug_list[positive_num], temp_aug1.to(device)), 0)
    
    shift_labels= torch.cat([torch.ones_like(labels) * k for k in range(len(negative_list)+1)], 0)  # B -> 2B (+1 for original data)

    shift_labels = shift_labels.repeat(len(positive_list)+1)


    for positive_num in range(0, len(positive_list)):
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

    total_time = 0.0
    total_memory = 0.0
    num_batches = len(train_loader)

    
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
        

        data, shift_labels = generate_augmentation(args, original_data, labels, device, positive_list, negative_list)
        data_f, shift_labels_f = generate_augmentation(args, original_data, labels, device, positive_list_f, negative_list_f)
        if args.ood_score[0] == 'NovelHD_TF':            
            data_f, shift_labels_f = data.clone().detach(), shift_labels.clone().detach()
            
        data_fft = torch.fft.fftn(data_f[0], norm="forward").reshape(1, data_f[0].shape[0], data_f[0].shape[1])

        for i in range(1, len(data_f)):
            data_fft = torch.cat([data_fft, torch.fft.fftn(data_f[i], norm="forward").reshape(1, data_f[0].shape[0], data_f[0].shape[1])], 0)            

            #sensor_pair_f = torch.cat([normal_fft, aug_fft], dim=0)   
            
            #data_f = fft.fftn(data_f, norm="backward").abs().to(device)

        data_f = data_fft

        assert data.size(0) == shift_labels.size(0)
        assert data_f.size(0) == shift_labels_f.size(0)            

        #Initialize loss
        loss_sim, loss_shift, loss_t = torch.tensor(0), torch.tensor(0), torch.tensor(0)
        loss_sim_f, loss_shift_f, loss_f = torch.tensor(0), torch.tensor(0), torch.tensor(0)
        l_TF = torch.tensor(0)

        
         # 시간 및 메모리 측정 시작
        torch.cuda.synchronize(device)  # GPU에서 동기화 (정확한 시간 측정을 위해)
        start_time = time.time()
        start_memory = torch.cuda.memory_allocated(device)  # 현재 할당된 GPU 메모리 측정

        
        # if ood_score in ['NovelHD_CB']:
        #     con_data = torch.cat((data, data_f), axis=1)
        #     h_t, z_t, s_t, h_f, z_f, s_f = model(con_data, con_data)    
        # else:
        # # original data and augmented data 
        #     if args.vis:
        #         h_t, z_t, s_t, h_f, z_f, s_f, _, _  = model(data, data_f)
        #     else:
        h_t, z_t, s_t, h_f, z_f, s_f = model(data, data_f)                  
               


            # for constructing loss functions
            # For temporal contrastive          
        #if ood_score in ['T', 'TCON' ,'TCLS' ,'CON' ,'CLS' ,'NovelHD_TF', 'NovelHD','simclr', 'NovelHD_CB']:
        simclr = normalize(z_t)  # normalize
        sim_matrix = get_similarity_matrix(simclr)            
        loss_sim = NT_xent(sim_matrix, data.size(0), temperature = args.temp, chunk = args.K_pos+1) #* sim_lambda
                
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
            
            # B_t = simclr.size(0) // (args.K_pos+1)
            # B_f = s_f.size(0) // (args.K_pos_f+1)

             # For frequency contrastive
            
        #if ood_score in ['F', 'FCON' ,'FCLS' ,'CON' ,'CLS' ,'NovelHD_TF', 'NovelHD']:

        simclr_f = normalize(z_f)  # normalize
        sim_matrix_f = get_similarity_matrix(simclr_f)            
        loss_sim_f = NT_xent(sim_matrix_f, data.size(0), temperature = args.temp, chunk = args.K_pos_f+1) #* sim_lambda_f 
        loss_shift_f = criterion(s_f, shift_labels_f)
        loss_f = loss_sim_f + loss_shift_f
            
        # if ood_score in ['NovelHD_TF']:
        #     nt_xent_criterion = NTXentLoss_poly(device, simclr.size(0), 0.5,
        #                                True) # device, 128, 0.2, True
        #     l_TF = nt_xent_criterion(z_t, z_f)
        
        
        # # 해당 코드 부분의 GPU 작업 완료를 기다림 (메모리 측정 후)
        # torch.cuda.synchronize()

        # # 메모리 사용량 측정 후 상태 기록
        # after_allocated = torch.cuda.memory_allocated()
        # after_reserved = torch.cuda.memory_reserved()

        # allocated_diff = after_allocated - before_allocated
        # reserved_diff = after_reserved - before_reserved
        # memory = []
        # memory =memory.append([allocated_diff, reserved_diff])
        # import pandas as pd

        # df = pd.DataFrame(memory, columns=['mean', 'std'])
        # df.to_excel('memory_results_CB.xlsx', sheet_name='the results')


        # if ood_score in ['NovelHD_CB', 'T']:
        #     loss = loss_t   
        # elif ood_score == 'F':
        #     loss = loss_f                
        # elif ood_score == 'TCON' or ood_score == 'simclr' :
        #     loss = loss_sim
        # elif ood_score == 'TCLS':
        #     loss = loss_shift
        # elif ood_score == 'FCON':
        #     loss =  loss_sim_f 
        # elif ood_score == 'FCLS':
        #     loss = loss_shift_f
        # elif ood_score == 'NovelHD':
        loss = args.lam_a *(loss_sim + loss_sim_f) + loss_shift + loss_shift_f
        # elif ood_score == 'CON':
        #     loss = loss_sim + loss_sim_f
        # elif ood_score == 'CLS':
        #     loss = loss_shift + loss_shift_f
        # elif ood_score == 'NovelHD_TF':
        #     loss = loss_sim + loss_sim_f + loss_shift + loss_shift_f + l_TF
        # else:
        #     raise ValueError() 
            
        # if epoch % 20 == 0 : 
        #     logger.debug(f'Temporal: {loss_sim.item():.4f}, {loss_shift.item():.4f}, {loss_t.item():.4f}')
        #     logger.debug(f'Frequency: {loss_sim_f.item():.4f},{loss_shift_f.item():.4f}, {loss_f.item():.4f}')
        #     logger.debug(f'TF: {loss.item():.4f}')


        total_loss.append(loss.item())
        loss.backward()
        model_optimizer.step()

        # 배치 처리 시간 계산 및 누적
        torch.cuda.synchronize(device)  # GPU 동기화 (시간 측정 종료 전)
        end_time = time.time()
        batch_time = (end_time - start_time) * 1000  # 초를 밀리초로 변환
        total_time += batch_time

        # 메모리 사용량 측정 및 누적
        end_memory = torch.cuda.memory_allocated(device)  # 현재 할당된 GPU 메모리 측정
        batch_memory = end_memory - start_memory  # 배치 처리에 사용된 메모리 차이
        total_memory += batch_memory

        print(f"Batch {batch_idx + 1}/{num_batches}: {batch_time:.2f} ms, {batch_memory / (1024 ** 2):.2f} MB")



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

    # 평균 배치 처리 시간 및 메모리 사용량 계산
    average_time_per_batch = total_time / num_batches
    average_memory_per_batch = total_memory / num_batches / (1024 ** 2)  # Bytes를 MB로 변환

    print(f"Average training time per batch: {average_time_per_batch:.2f} ms")
    print(f"Average memory usage per batch: {average_memory_per_batch:.2f} MB")

    return total_loss, average_time_per_batch, average_memory_per_batch
