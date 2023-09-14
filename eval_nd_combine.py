import os
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import random
from util import calculate_acc_rv

import torch.fft as fft
from ood_metrics import auroc, aupr, fpr_at_95_tpr, detection_error
from data_preprocessing.augmentations import select_transformation
from sklearn.metrics import f1_score, roc_auc_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def normalize(x, dim=1, eps=1e-8):
    return x / (x.norm(dim=dim, keepdim=True) + eps)

def softmax(x):
    x = np.asarray(x)
    e_x = np.exp(x - np.max(x))
    return (e_x / e_x.sum()).tolist()

def eval_ood_detection(args, path, model, id_loader, ood_loaders, ood_scores, train_loader, negative_list, negative_list_f):

    auroc_dict  = dict()
    aupr_dict   = dict()
    fpr_dict    = dict()
    de_dict     = dict()

    for ood in ood_loaders.keys():
        auroc_dict[ood] = dict()
        aupr_dict[ood]  = dict()
        fpr_dict[ood]   = dict()
        de_dict[ood]    = dict()
    assert len(ood_scores) == 1  # assume single ood_score for simplicity
    ood_score = ood_scores[0]

    base_path = path#os.path.split(P.load_path)[0]  # checkpoint directory

    prefix = f'{args.ood_samples}'

    prefix = os.path.join(base_path, f'feats_{prefix}')

    kwargs = {
        'sample_num': args.ood_samples,
        'layers': ['simclr', 'shift'],
    }

    print('Pre-compute global statistics...')
    feats_train = get_features(args, negative_list, negative_list_f, f'{args.selected_dataset}_train', 
                               model, train_loader, prefix=prefix, **kwargs)  # (M, T, d)

    args.axis = []
    
    for f in feats_train['simclr'].chunk((args.K_shift+args.K_shift_f), dim=1):
        axis = f.mean(dim=1)  # (M, d)
        args.axis.append(normalize(axis, dim=1).to(device))
    print('axis size: ' + ' '.join(map(lambda x: str(len(x)), args.axis)))



    f_sim = [f.mean(dim=1) for f in feats_train['simclr'].chunk((args.K_shift+args.K_shift_f), dim=1)]  # list of (M, d)
    f_shi = [f.mean(dim=1) for f in feats_train['shift'].chunk((args.K_shift+args.K_shift_f), dim=1)]  # list of (M, 4)



    # weight
    weight_sim = []
    weight_shi = []

    print(len(f_sim),len(f_shi),len(f_shi[0]))
    for shi in range((args.K_shift)):
        sim_norm_t = f_sim[shi].norm(dim=1)  # (M)
        shi_mean_t = f_shi[shi][:, shi]  # (M)
        weight_sim.append(1 / sim_norm_t.mean().item())
        weight_shi.append(1 / shi_mean_t.mean().item())
    

    args.weight_sim = weight_sim # weight_sim_t or [0,0]
    args.weight_shi = weight_shi # weight_shi_t or [0,0]


    print(f'weight_sim_t:\t' + '\t'.join(map('{:.4f}'.format, args.weight_sim)))
    print(f'weight_shi_t:\t' + '\t'.join(map('{:.4f}'.format, args.weight_shi)))


    print('Compute known class features...')
    feats_id = get_features(args, negative_list, negative_list_f, args.selected_dataset, 
                            model, id_loader, prefix=prefix, **kwargs)  # (N, T, d)
    feats_ood = dict()
    for ood, ood_loader in ood_loaders.items():
        feats_ood[ood] = get_features(args, negative_list, negative_list_f, ood, model, ood_loader, prefix=prefix, **kwargs)


    scores_id = get_scores(args, feats_id).numpy()
    scores_ood = dict()
    #if args.one_class_idx != -1:
    one_class_score = []

    print(f'Compute OOD scores... (score: {ood_score})')
    for ood, feats in feats_ood.items():
        scores_ood[ood]             = get_scores(args, feats).numpy()
        auroc_dict[ood][ood_score]  = get_auroc(scores_id, scores_ood[ood])
        aupr_dict[ood][ood_score]   = get_aupr(scores_id, scores_ood[ood])
        fpr_dict[ood][ood_score]    = get_fpr(scores_id, scores_ood[ood])
        de_dict[ood][ood_score]     = get_de(scores_id, scores_ood[ood])
        #if args.one_class_idx       != -1:
        one_class_score.append(scores_ood[ood])

    #if args.one_class_idx != -1:
    one_class_score = np.concatenate(one_class_score)
    one_class_total = get_auroc(scores_id, one_class_score)
    one_class_aupr  = get_aupr(scores_id, one_class_score)
    one_class_fpr   = get_fpr(scores_id, one_class_score)
    one_class_de    = get_de(scores_id, one_class_score)
        #print(f'One_class_real_mean: {one_class_total:.3f}')
        #print(f'One_class_aupr_mean: {one_class_aupr:.3f}')
        #print(f'One_class_fpr_mean: {one_class_fpr:.3f}')
        #print(f'One_class_f1_mean: {one_class_f1}')
    print(f'{one_class_total:.3f}')
    print(f'{one_class_aupr:.3f}')
    print(f'{one_class_fpr:.3f}')
    print(f'{one_class_de:.3f}')


    if args.print_score:
        print_score(args.selected_dataset, scores_id)
        for ood, scores in scores_ood.items():
            print_score(ood, scores)

    scores = np.concatenate([scores_id, one_class_score]).tolist()
    labels = np.concatenate([np.ones_like(scores_id, dtype=int), np.zeros_like(one_class_score, dtype=int)]).tolist()
    auroc, fpr, f1, acc = calculate_acc_rv(labels, scores)


    return auroc_dict, aupr_dict, fpr_dict, de_dict, auroc, fpr, f1, acc, scores, labels


def get_scores(args, feats_dict):
    # convert to gpu tensor
    feats_sim = feats_dict['simclr'].to(device)
    feats_shi = feats_dict['shift'].to(device)

    N = feats_sim.size(0)

    # compute scores
    scores = []
   

    for f_sim_t, f_shi_t  in zip(feats_sim, feats_shi):
        f_sim_t = [f.mean(dim=0, keepdim=True) for f in f_sim_t.chunk((args.K_shift+args.K_shift_f))]  # list of (1, d)
        f_shi_t = [f.mean(dim=0, keepdim=True) for f in f_shi_t.chunk((args.K_shift+args.K_shift_f))]  # list of (1, 4)


        score = 0


        for shi in range(args.K_shift):
            score += (f_sim_t[shi] * args.axis[shi]).sum(dim=1).max().item() * args.weight_sim[shi]
            score += f_shi_t[shi][:, shi].item() * args.weight_shi[shi]
        score = score / (args.K_shift+args.K_shift_f)

        scores.append(score)
    scores = torch.tensor(scores)

    assert scores.dim() == 1 and scores.size(0) == N  # (N)
    return scores.cpu()


def get_features(args, negative_list, negative_list_f, data_name, model, loader, prefix='',
                sample_num=1, layers=('simclr', 'shift')):

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
        _feats_dict = _get_features(args, negative_list, negative_list_f, model, loader, sample_num, layers=left)

        for layer, feats in _feats_dict.items():
            path = prefix + f'_{data_name}_{layer}.pth'
            torch.save(_feats_dict[layer], path)
            feats_dict[layer] = feats  # update value

    return feats_dict


def _get_features(args, negative_list, negative_list_f, model, loader, sample_num=10, layers=('simclr', 'shift')):
    output_aux = dict()
    if not isinstance(layers, (list, tuple)):
        layers = [layers]

    # compute features in full dataset
    model.eval()
    feats_all = {layer: [] for layer in layers}  # initialize: empty list

    for i, (x, labels, _, _, _) in enumerate(loader):

        x    = x.to(device) 
        x_f  = x.to(device) 
        # compute features in one batch
        feats_batch = {layer: [] for layer in layers}  # initialize: empty list
        
        for seed in range(sample_num):            

            original_data = x

            for shifted_num in range(len(negative_list)):
                shifted_aug = select_transformation(negative_list[shifted_num], original_data.shape[2])
                # adding shifted transformation
                temp_data = torch.from_numpy(np.array(shifted_aug.augment(original_data.permute(0, 2, 1).cpu().numpy()))).permute(0, 2, 1)
                x = torch.cat([x, temp_data.to(device)], 0)

            for shifted_num in range(len(negative_list_f)): 
                shifted_aug = select_transformation(negative_list_f[shifted_num], original_data.shape[2])
                temp_data = torch.from_numpy(np.array(shifted_aug.augment(original_data.permute(0, 2, 1).cpu().numpy()))).permute(0, 2, 1)
                x_f = torch.cat([x_f, temp_data.to(device)], 0)

            data_fft = torch.fft.fftn(x_f[0], norm="forward").abs().reshape(1, x_f[0].shape[0], x_f[0].shape[1])

            for i in range(1, len(x_f)):
                data_fft = torch.cat([data_fft, torch.fft.fftn(x_f[i], norm="forward").abs().reshape(1, x_f[0].shape[0], x_f[0].shape[1])], 0)            


            x_f = data_fft
            
            #x_f      = fft.fftn(x_f, norm="backward").abs()                   

            # compute augmented features
            with torch.no_grad():
                kwargs = {layer: True for layer in layers}  # only forward selected layers
                _, z_t, s_t, _, z_f, s_f  = model(x, x_f)
                output_aux['simclr'] = torch.cat([z_t, z_f])
                output_aux['shift'] = torch.cat([s_t, s_f])
                #output_aux['simclr_f'] = z_f
                #output_aux['shift_f'] = s_f                     

            # add features in one batch
            for layer in ['simclr', 'shift']:
                feats = output_aux[layer].cpu()
                feats_batch[layer] += feats.chunk(args.K_shift+args.K_shift_f)

        # concatenate features in one batch
        for key, val in feats_batch.items():
            feats_batch[key] = torch.stack(val, dim=1)  # (B, T, d)

        # add features in full dataset
        for layer in layers:
            feats_all[layer] += [feats_batch[layer]]

    # concatenate features in full dataset
    for key, val in feats_all.items():
        feats_all[key] = torch.cat(val, dim=0)  # (N, T, d)

    # reshape order
        # Convert [1,2,3,4, 1,2,3,4] -> [1,1, 2,2, 3,3, 4,4]
    for key, val in feats_all.items():
        N, T, d = val.size()  # T = K * T'
        
        val = val.view(N, -1, (args.K_shift+args.K_shift_f), d)  # (N, T', K, d)
        val = val.transpose(2, 1)  # (N, 4, T', d)
        val = val.reshape(N, T, d)  # (N, T, d)

        feats_all[key] = val

    return feats_all

def get_auroc(scores_id, scores_ood):
    scores = np.concatenate([scores_id, scores_ood])
    labels = np.concatenate([np.ones_like(scores_id), np.zeros_like(scores_ood)])
    return auroc(scores, labels)

def get_aupr(scores_id, scores_ood):
    scores = np.concatenate([scores_id, scores_ood])
    labels = np.concatenate([np.ones_like(scores_id), np.zeros_like(scores_ood)])
    return aupr(scores, labels)

def get_fpr(scores_id, scores_ood):
    scores = np.concatenate([scores_id, scores_ood])
    labels = np.concatenate([np.ones_like(scores_id), np.zeros_like(scores_ood)])
    return fpr_at_95_tpr(scores, labels)

def get_de(scores_id, scores_ood):
    scores = np.concatenate([scores_id, scores_ood])
    labels = np.concatenate([np.ones_like(scores_id), np.zeros_like(scores_ood)])
    return detection_error(scores, labels)

def print_score(data_name, scores):
    quantile = np.quantile(scores, np.arange(0, 1.1, 0.1))
    print('{:18s} '.format(data_name) +
          '{:.4f} +- {:.4f}    '.format(np.mean(scores), np.std(scores)) +
          '    '.join(['q{:d}: {:.4f}'.format(i * 10, quantile[i]) for i in range(11)]))