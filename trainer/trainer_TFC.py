import os
import sys

sys.path.append("..")
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.loss import NTXentLoss, SupConLoss, get_similarity_matrix, NT_xent
from sklearn.metrics import f1_score, roc_auc_score

def Trainer(model, model_optimizer, classifier, classifier_optimizer, train_dl, valid_dl, test_dl, device, logger, configs, experiment_log_dir, training_mode):
    # Start training
    logger.debug("Training started ....")

    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model_optimizer, 'min')

    for epoch in range(1, configs.num_epoch + 1):
        # Train and validate
        train_loss, train_acc = model_train(model, model_optimizer, classifier, classifier_optimizer, criterion, train_dl, configs, device, training_mode)
        valid_loss, valid_acc, valid_f1, _, _ = model_evaluate(model, classifier, test_dl, device, training_mode)
        if training_mode != 'self_supervised' and training_mode!="novelty_detection":  # use scheduler in all other modes.
            scheduler.step(valid_loss)

        logger.debug(f'\nEpoch : {epoch}\n'
                     f'Train Loss     : {train_loss:.4f}\t | \tTrain Accuracy     : {train_acc:2.4f}\n'
                     f'Valid Loss     : {valid_loss:.4f}\t | \tValid Accuracy     : {valid_acc:2.4f} | \tValid F1-score      : {valid_f1:0.4f}')

    os.makedirs(os.path.join(experiment_log_dir, "saved_models"), exist_ok=True)
    chkpoint = {'model_state_dict': model.state_dict(), 'classifier_state_dict': classifier.state_dict()}
    torch.save(chkpoint, os.path.join(experiment_log_dir, "saved_models", f'ckp_last.pt'))

    if training_mode != "self_supervised" and training_mode != "novelty_detection":  # no need to run the evaluation for self-supervised mode.
        # evaluate on the test set
        logger.debug('\nEvaluate on the Test set:')
        test_loss, test_acc, test_f1, _, _ = model_evaluate(model, classifier, test_dl, device, training_mode)
        logger.debug(f'Test loss      :{test_loss:0.4f}\t | Test Accuracy      : {test_acc:0.4f}\t | Test F1-score      : {test_f1:0.4f}')

    logger.debug("\n################## Training is Done! #########################")

def normalize(x, dim=1, eps=1e-8):
    return x / (x.norm(dim=dim, keepdim=True) + eps)

def model_train(model, model_optimizer, classifier, classifier_optimizer, criterion, train_loader, configs, device, training_mode):
    total_loss = []
    total_acc = []
    model.train()
    classifier.train()
    
    for batch_idx, (data, labels, aug1, data_f, aug1_f) in enumerate(train_loader):
        # send to device
        data, labels = data.float().to(device), labels.long().to(device) # data: [128, 1, 178], labels: [128]
        aug1 = aug1.float().to(device)  # aug1 = aug2 : [128, 1, 178]
        data_f, aug1_f = data_f.float().to(device), aug1_f.float().to(device)  # aug1 = aug2 : [128, 1, 178]

        # optimizer
        model_optimizer.zero_grad()
        classifier_optimizer.zero_grad()


        if training_mode == "self_supervised" and configs.batch_size == data.shape[0]:
   
            h_t, z_t, h_f, z_f  = model(data, data_f)
            h_t_aug, z_t_aug, h_f_aug, z_f_aug = model(aug1, aug1_f)

            # normalize projection feature vectors
            #zis = temp_cont_lstm_feat1 
            #zjs = temp_cont_lstm_feat2 

        else:
            h_t, z_t, h_f, z_f  = model(data, data_f)
            h_t_aug, z_t_aug, h_f_aug, z_f_aug = model(aug1, aug1_f)
            fea_concat = torch.cat((z_t, z_f), dim=1)
            predictions = classifier(fea_concat)



        # compute loss
        if training_mode == "self_supervised" and configs.batch_size == data.shape[0]:
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

                
        elif training_mode != "self_supervised" and training_mode != "novelty_detection": # supervised training or fine tuining
            
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

    criterion = nn.CrossEntropyLoss()
    outs = np.array([])
    trgs = np.array([])

    with torch.no_grad():
        for data, labels, _,data_f, _ in test_dl:
            data, labels = data.float().to(device), labels.long().to(device)
            data_f = data_f.float().to(device)

            if training_mode == "self_supervised":
                pass
            else:
                h_t, z_t, h_f, z_f = model(data, data_f)
                fea_concat = torch.cat((z_t, z_f), dim=1)                

            # compute loss
            if training_mode != "self_supervised":
                predictions = classifier(fea_concat)
                loss = criterion(predictions, labels)
                total_acc.append(labels.eq(predictions.detach().argmax(dim=1)).float().mean())
                #print(labels, predictions.detach().argmax(dim=1))
                total_f1.append(f1_score(labels.cpu(), predictions.detach().argmax(dim=1).cpu(), average='macro'))
                total_loss.append(loss.item())

            if training_mode != "self_supervised":
                pred = predictions.max(1, keepdim=True)[1]  # get the index of the max log-probability
                outs = np.append(outs, pred.cpu().numpy())
                trgs = np.append(trgs, labels.data.cpu().numpy())

    if training_mode != "self_supervised":
        total_loss = torch.tensor(total_loss).mean()  # average loss
    else:
        total_loss = 0

    if training_mode == "self_supervised":
        total_acc = 0
        total_f1  = 0
        return total_loss, total_acc, total_f1, [], []
    else:
        total_acc = torch.tensor(total_acc).mean()  # average acc
        total_f1  = torch.tensor(total_f1).mean() # average f1

    return total_loss, total_acc, total_f1, outs, trgs

def eval_ood_detection(P, model, id_loader, ood_loaders, ood_scores, train_loader=None, simclr_aug=None):
    model.eval()
    auroc_dict = dict()
    for ood in ood_loaders.keys():
        auroc_dict[ood] = dict()

    assert len(ood_scores) == 1  # assume single ood_score for simplicity
    ood_score = ood_scores[0]

    base_path = os.path.split(P.load_path)[0]  # checkpoint directory

    prefix = f'{P.ood_samples}'
    if P.resize_fix:
        prefix += f'_resize_fix_{P.resize_factor}'
    else:
        prefix += f'_resize_range_{P.resize_factor}'

    prefix = os.path.join(base_path, f'feats_{prefix}')

    kwargs = {
        'simclr_aug': simclr_aug,
        'sample_num': P.ood_samples,
        'layers': P.ood_layer,
    }

    print('Pre-compute global statistics...')
    feats_train = get_features(P, f'{P.dataset}_train', model, train_loader, prefix=prefix, **kwargs)  # (M, T, d)

    P.axis = []
    for f in feats_train['simclr'].chunk(P.K_shift, dim=1):
        axis = f.mean(dim=1)  # (M, d)
        P.axis.append(normalize(axis, dim=1).to(device))
    print('axis size: ' + ' '.join(map(lambda x: str(len(x)), P.axis)))

    f_sim = [f.mean(dim=1) for f in feats_train['simclr'].chunk(P.K_shift, dim=1)]  # list of (M, d)
    f_shi = [f.mean(dim=1) for f in feats_train['shift'].chunk(P.K_shift, dim=1)]  # list of (M, 4)

    weight_sim = []
    weight_shi = []
    for shi in range(P.K_shift):
        sim_norm = f_sim[shi].norm(dim=1)  # (M)
        shi_mean = f_shi[shi][:, shi]  # (M)
        weight_sim.append(1 / sim_norm.mean().item())
        weight_shi.append(1 / shi_mean.mean().item())

    if ood_score == 'simclr':
        P.weight_sim = [1]
        P.weight_shi = [0]
    elif ood_score == 'CSI':
        P.weight_sim = weight_sim
        P.weight_shi = weight_shi
    else:
        raise ValueError()

    print(f'weight_sim:\t' + '\t'.join(map('{:.4f}'.format, P.weight_sim)))
    print(f'weight_shi:\t' + '\t'.join(map('{:.4f}'.format, P.weight_shi)))

    print('Pre-compute features...')
    feats_id = get_features(P, P.dataset, model, id_loader, prefix=prefix, **kwargs)  # (N, T, d)
    feats_ood = dict()
    for ood, ood_loader in ood_loaders.items():
        if ood == 'interp':
            feats_ood[ood] = get_features(P, ood, model, id_loader, interp=True, prefix=prefix, **kwargs)
        else:
            feats_ood[ood] = get_features(P, ood, model, ood_loader, prefix=prefix, **kwargs)

    print(f'Compute OOD scores... (score: {ood_score})')
    scores_id = get_scores(P, feats_id, ood_score).numpy()
    scores_ood = dict()
    if P.one_class_idx is not None:
        one_class_score = []

    for ood, feats in feats_ood.items():
        scores_ood[ood] = get_scores(P, feats, ood_score).numpy()
        auroc_dict[ood][ood_score] = get_auroc(scores_id, scores_ood[ood])
        if P.one_class_idx is not None:
            one_class_score.append(scores_ood[ood])

    if P.one_class_idx is not None:
        one_class_score = np.concatenate(one_class_score)
        one_class_total = get_auroc(scores_id, one_class_score)
        print(f'One_class_real_mean: {one_class_total}')

    if P.print_score:
        print_score(P.dataset, scores_id)
        for ood, scores in scores_ood.items():
            print_score(ood, scores)

    return auroc_dict


def get_scores(P, feats_dict, ood_score):
    # convert to gpu tensor
    feats_sim = feats_dict['simclr'].to(device)
    feats_shi = feats_dict['shift'].to(device)
    N = feats_sim.size(0)

    # compute scores
    scores = []
    for f_sim, f_shi in zip(feats_sim, feats_shi):
        f_sim = [f.mean(dim=0, keepdim=True) for f in f_sim.chunk(P.K_shift)]  # list of (1, d)
        f_shi = [f.mean(dim=0, keepdim=True) for f in f_shi.chunk(P.K_shift)]  # list of (1, 4)
        score = 0
        for shi in range(P.K_shift):
            score += (f_sim[shi] * P.axis[shi]).sum(dim=1).max().item() * P.weight_sim[shi]
            score += f_shi[shi][:, shi].item() * P.weight_shi[shi]
        score = score / P.K_shift
        scores.append(score)
    scores = torch.tensor(scores)

    assert scores.dim() == 1 and scores.size(0) == N  # (N)
    return scores.cpu()


def get_auroc(scores_id, scores_ood):
    scores = np.concatenate([scores_id, scores_ood])
    labels = np.concatenate([np.ones_like(scores_id), np.zeros_like(scores_ood)])
    return roc_auc_score(labels, scores)


def get_features(P, data_name, model, loader, interp=False, prefix='',
                 simclr_aug=None, sample_num=1, layers=('simclr', 'shift')):

    if not isinstance(layers, (list, tuple)):
        layers = [layers]

    # load pre-computed features if exists
    feats_dict = dict()
    # for layer in layers:
    #     path = prefix + f'_{data_name}_{layer}.pth'
    #     if os.path.exists(path):
    #         feats_dict[layer] = torch.load(path)

    # pre-compute features and save to the path
    left = [layer for layer in layers if layer not in feats_dict.keys()]
    if len(left) > 0:
        _feats_dict = _get_features(P, model, loader, interp, P.dataset == 'imagenet',
                                    simclr_aug, sample_num, layers=left)

        for layer, feats in _feats_dict.items():
            path = prefix + f'_{data_name}_{layer}.pth'
            torch.save(_feats_dict[layer], path)
            feats_dict[layer] = feats  # update value

    return feats_dict


def _get_features(P, model, loader, interp=False, imagenet=False, simclr_aug=None,
                  sample_num=1, layers=('simclr', 'shift')):

    if not isinstance(layers, (list, tuple)):
        layers = [layers]

    # check if arguments are valid
    assert simclr_aug is not None

    if imagenet is True:  # assume batch_size = 1 for ImageNet
        sample_num = 1

    # compute features in full dataset
    model.eval()
    feats_all = {layer: [] for layer in layers}  # initialize: empty list
    for i, (x, _) in enumerate(loader):
        if interp:
            x_interp = (x + last) / 2 if i > 0 else x  # omit the first batch, assume batch sizes are equal
            last = x  # save the last batch
            x = x_interp  # use interp as current batch

        if imagenet is True:
            x = torch.cat(x[0], dim=0)  # augmented list of x

        x = x.to(device)  # gpu tensor

        # compute features in one batch
        feats_batch = {layer: [] for layer in layers}  # initialize: empty list
        for seed in range(sample_num):
            set_random_seed(seed)

            if P.K_shift > 1:
                x_t = torch.cat([P.shift_trans(hflip(x), k) for k in range(P.K_shift)])
            else:
                x_t = x # No shifting: SimCLR
            x_t = simclr_aug(x_t)

            # compute augmented features
            with torch.no_grad():
                kwargs = {layer: True for layer in layers}  # only forward selected layers
                _, output_aux = model(x_t, **kwargs)

            # add features in one batch
            for layer in layers:
                feats = output_aux[layer].cpu()
                if imagenet is False:
                    feats_batch[layer] += feats.chunk(P.K_shift)
                else:
                    feats_batch[layer] += [feats]  # (B, d) cpu tensor

        # concatenate features in one batch
        for key, val in feats_batch.items():
            if imagenet:
                feats_batch[key] = torch.stack(val, dim=0)  # (B, T, d)
            else:
                feats_batch[key] = torch.stack(val, dim=1)  # (B, T, d)

        # add features in full dataset
        for layer in layers:
            feats_all[layer] += [feats_batch[layer]]

    # concatenate features in full dataset
    for key, val in feats_all.items():
        feats_all[key] = torch.cat(val, dim=0)  # (N, T, d)

    # reshape order
    if imagenet is False:
        # Convert [1,2,3,4, 1,2,3,4] -> [1,1, 2,2, 3,3, 4,4]
        for key, val in feats_all.items():
            N, T, d = val.size()  # T = K * T'
            val = val.view(N, -1, P.K_shift, d)  # (N, T', K, d)
            val = val.transpose(2, 1)  # (N, 4, T', d)
            val = val.reshape(N, T, d)  # (N, T, d)
            feats_all[key] = val

    return feats_all


def print_score(data_name, scores):
    quantile = np.quantile(scores, np.arange(0, 1.1, 0.1))
    print('{:18s} '.format(data_name) +
          '{:.4f} +- {:.4f}    '.format(np.mean(scores), np.std(scores)) +
          '    '.join(['q{:d}: {:.4f}'.format(i * 10, quantile[i]) for i in range(11)]))
