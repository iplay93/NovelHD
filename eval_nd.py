import os
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import random
from sklearn.metrics import roc_auc_score,  f1_score
from trainer.trainer_TFC import shifted_aug
import torch.fft as fft
from ood_metrics import auroc, aupr, fpr_at_95_tpr, detection_error
import kmeans1d

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def normalize(x, dim=1, eps=1e-8):
    return x / (x.norm(dim=dim, keepdim=True) + eps)

def eval_ood_detection(args, path, model, id_loader, ood_loaders, ood_scores, train_loader=None):
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
        'layers': ['simclr_t', 'shift_t','simclr_f', 'shift_f'],
    }

    print('Pre-compute global statistics...')
    feats_train = get_features(args, f'{args.selected_dataset}_train', 
                               model, train_loader, prefix=prefix, **kwargs)  # (M, T, d)

    args.axis = []
    for f in feats_train['simclr_t'].chunk(2, dim=1):
        axis = f.mean(dim=1)  # (M, d)
        args.axis.append(normalize(axis, dim=1).to(device))
    print('axis size: ' + ' '.join(map(lambda x: str(len(x)), args.axis)))


    f_sim_t = [f.mean(dim=1) for f in feats_train['simclr_t'].chunk(2, dim=1)]  # list of (M, d)
    f_shi_t = [f.mean(dim=1) for f in feats_train['shift_t'].chunk(2, dim=1)]  # list of (M, 4)
    f_sim_f = [f.mean(dim=1) for f in feats_train['simclr_f'].chunk(2, dim=1)]  # list of (M, d)
    f_shi_f = [f.mean(dim=1) for f in feats_train['shift_f'].chunk(2, dim=1)]  # list of (M, 4)

    weight_sim_t = []
    weight_shi_t = []
    
    weight_sim_f = []
    weight_shi_f = []
    for shi in range(2):
        sim_norm_t = f_sim_t[shi].norm(dim=1)  # (M)
        shi_mean_t = f_shi_t[shi][:, shi]  # (M)
        weight_sim_t.append(1 / sim_norm_t.mean().item())
        weight_shi_t.append(1 / shi_mean_t.mean().item())
        
        sim_norm_f = f_sim_f[shi].norm(dim=1)  # (M)
        shi_mean_f = f_shi_f[shi][:, shi]  # (M)
        weight_sim_f.append(1 / sim_norm_f.mean().item())
        weight_shi_f.append(1 / shi_mean_f.mean().item())

    if ood_score == 'T':
        args.weight_sim_t = weight_sim_t # weight_sim_t or [0,0]
        args.weight_shi_t = weight_shi_t # weight_shi_t or [0,0]
        args.weight_sim_f = [0, 0]   # weight_sim_f or [0,0] 
        args.weight_shi_f = [0, 0] # weight_shi_f or [0,0]        
    elif ood_score == 'TCON':
        args.weight_sim_t = [1 ,1]
        args.weight_shi_t = [0, 0]
        args.weight_sim_f = [0, 0]
        args.weight_shi_f = [0, 0]
    elif ood_score == 'TCLS':
        args.weight_sim_t = [0, 0]
        args.weight_shi_t = [1, 1]
        args.weight_sim_f = [0, 0]
        args.weight_shi_f = [0, 0]
    elif ood_score == 'FCON':
        args.weight_sim_t = weight_sim_t # weight_sim_t or [0,0]
        args.weight_shi_t = weight_shi_t # weight_shi_t or [0,0]
        args.weight_sim_f = weight_sim_f   # weight_sim_f or [0,0] 
        args.weight_shi_f = [0, 0] # weight_shi_f or [0,0]  
    elif ood_score == 'FCLS':
        args.weight_sim_t = weight_sim_t # weight_sim_t or [0,0]
        args.weight_shi_t = weight_shi_t # weight_shi_t or [0,0]
        args.weight_sim_f = [0, 0]   # weight_sim_f or [0,0] 
        args.weight_shi_f = weight_shi_f # weight_shi_f or [0,0]       
    elif ood_score == 'NovelHD' or ood_score == 'NovelHD_TF' :
        args.weight_sim_t = weight_sim_t # weight_sim_t or [0,0]
        args.weight_shi_t = weight_shi_t # weight_shi_t or [0,0]
        args.weight_sim_f = weight_sim_f # weight_sim_f or [0,0] 
        args.weight_shi_f = weight_shi_f # weight_shi_f or [0,0]
    else:
        raise ValueError()

    print(f'weight_sim_t:\t' + '\t'.join(map('{:.4f}'.format, args.weight_sim_t)))
    print(f'weight_shi_t:\t' + '\t'.join(map('{:.4f}'.format, args.weight_shi_t)))
    print(f'weight_sim_f:\t' + '\t'.join(map('{:.4f}'.format, args.weight_sim_f)))
    print(f'weight_shi_f:\t' + '\t'.join(map('{:.4f}'.format, args.weight_shi_f)))

    print('Pre-compute features...')
    feats_id = get_features(args, args.selected_dataset, 
                            model, id_loader, prefix=prefix, **kwargs)  # (N, T, d)
    feats_ood = dict()
    for ood, ood_loader in ood_loaders.items():
        feats_ood[ood] = get_features(args, ood, model, ood_loader, prefix=prefix, **kwargs)

    print(f'Compute OOD scores... (score: {ood_score})')
    scores_id = get_scores(args, feats_id, ood_score).numpy()
    scores_ood = dict()
    #if args.one_class_idx != -1:
    one_class_score = []

    for ood, feats in feats_ood.items():
        scores_ood[ood]             = get_scores(args, feats, ood_score).numpy()
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

    return auroc_dict, aupr_dict, fpr_dict, de_dict, one_class_total, one_class_aupr, one_class_fpr, one_class_de


def get_scores(args, feats_dict, ood_score):
    # convert to gpu tensor
    feats_sim_t = feats_dict['simclr_t'].to(device)
    feats_shi_t = feats_dict['shift_t'].to(device)
    feats_sim_f = feats_dict['simclr_f'].to(device)
    feats_shi_f = feats_dict['shift_f'].to(device)
    N = feats_sim_t.size(0)

    # compute scores
    scores = []
    for f_sim_t, f_shi_t, f_sim_f, f_shi_f  in zip(feats_sim_t, feats_shi_t, feats_sim_f, feats_shi_f):
        f_sim_t = [f.mean(dim=0, keepdim=True) for f in f_sim_t.chunk(2)]  # list of (1, d)
        f_shi_t = [f.mean(dim=0, keepdim=True) for f in f_shi_t.chunk(2)]  # list of (1, 4)
        f_sim_f = [f.mean(dim=0, keepdim=True) for f in f_sim_f.chunk(2)]  # list of (1, d)
        f_shi_f = [f.mean(dim=0, keepdim=True) for f in f_shi_f.chunk(2)]  # list of (1, 4)
        score = 0
        lam = 1
        for shi in range(2):
            score += (f_sim_t[shi] * args.axis[shi]).sum(dim=1).max().item() * args.weight_sim_t[shi]
            score += f_shi_t[shi][:, shi].item() * args.weight_shi_t[shi]
            score += (f_sim_f[shi] * args.axis[shi]).sum(dim=1).max().item() * args.weight_sim_f[shi] * lam
            score += f_shi_f[shi][:, shi].item() * args.weight_shi_f[shi] * lam
        score = score / 2
        scores.append(score)
    scores = torch.tensor(scores)

    assert scores.dim() == 1 and scores.size(0) == N  # (N)
    return scores.cpu()


def get_features(args, data_name, model, loader, prefix='',
                sample_num=1, layers=('simclr_t', 'shift_t','simclr_f', 'shift_f')):

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
        _feats_dict = _get_features(args, model, loader, sample_num, layers=left)

        for layer, feats in _feats_dict.items():
            path = prefix + f'_{data_name}_{layer}.pth'
            torch.save(_feats_dict[layer], path)
            feats_dict[layer] = feats  # update value

    return feats_dict


def _get_features(args, model, loader, sample_num=1, layers=('simclr_t', 'shift_t','simclr_f', 'shift_f')):
    output_aux = dict()
    if not isinstance(layers, (list, tuple)):
        layers = [layers]

    # compute features in full dataset
    model.eval()
    feats_all = {layer: [] for layer in layers}  # initialize: empty list
    for i, (x, labels, _, x_f, _) in enumerate(loader):

        x       = x.to(device) 
        x_f     = x_f.to(device)

        # compute features in one batch
        feats_batch = {layer: [] for layer in layers}  # initialize: empty list
        for seed in range(sample_num):
            set_random_seed(seed)

            # adding shifted transformation
            for k in range(x.size(0)):
                temp_data = torch.from_numpy(shifted_aug.augment
                    (np.reshape(x[k].cpu().numpy(),(1, x[k].shape[1],-1)))).permute(0, 2, 1)

                x        = torch.cat((x, temp_data.to(device)), 0)
                
            x_f      = fft.fft(x).abs()       

            # compute augmented features
            with torch.no_grad():
                kwargs = {layer: True for layer in layers}  # only forward selected layers
                _, z_t, s_t, _, z_f, s_f  = model(x, x_f)
                output_aux['simclr_t'] = z_t
                output_aux['shift_t'] = s_t
                output_aux['simclr_f'] = z_f
                output_aux['shift_f'] = s_f                 

            # add features in one batch
            for layer in layers:
                feats = output_aux[layer].cpu()

                feats_batch[layer] += feats.chunk(2)


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
        val = val.view(N, -1, 2, d)  # (N, T', K, d)
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