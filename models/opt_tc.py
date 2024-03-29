import torch.utils.data
import numpy as np
import torch
import torch.utils.data
from torch.backends import cudnn
#from wideresnet import WideResNet
from sklearn.metrics import roc_auc_score
from models.TFC import TFC_GOAD, target_classifier
from ood_metrics import auroc, aupr, fpr_at_95_tpr, detection_error


cudnn.benchmark = True

def tc_loss(zs, m):
    means = zs.mean(0).unsqueeze(0)
    res = ((zs.unsqueeze(2) - means.unsqueeze(1)) ** 2).sum(-1)
    pos = torch.diagonal(res, dim1=1, dim2=2)
    offset = torch.diagflat(torch.ones(zs.size(1))).unsqueeze(0).cuda() * 1e6
    neg = (res + offset).min(-1)[0]
    loss = torch.clamp(pos + m - neg, min=0).mean()
    return loss

def select_transformation(aug_method, target_len):
    if(aug_method == 'AddNoise'):
        my_aug = (AddNoise(scale=0.01))
    elif(aug_method == 'Convolve'):
        my_aug = (Convolve(window="flattop", size=11))
    elif(aug_method == 'Crop'):
        my_aug = (Crop(size = target_len))
    elif(aug_method == 'Drift'):
        my_aug = (Drift(max_drift=0.7, n_drift_points=5))
    elif(aug_method == 'Dropout'):
        my_aug = (Dropout( p=0.1,fill=0))        
    elif(aug_method == 'Pool'):
        my_aug = (Pool(size=2))
    elif(aug_method == 'Quantize'):
        my_aug = (Quantize(n_levels=20))
    elif(aug_method == 'Resize'):
        my_aug = (Resize(size = target_len))
    elif(aug_method == 'Reverse'):
        my_aug = (Reverse())
    elif(aug_method == 'TimeWarp'):
        my_aug = (TimeWarp(n_speed_change=5, max_speed_ratio=3))
    
    return my_aug

class TransClassifier():
    def __init__(self, num_trans, args, configs):
        self.n_trans = num_trans
        self.args = args
        self.configs = configs
        self.model = TFC_GOAD(configs, num_trans).cuda()
        #self.netWRN = WideResNet(self.args.depth, num_trans, self.args.widen_factor).cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), 
                                            lr=configs.lr, betas=(configs.beta1, configs.beta2), weight_decay=3e-4)
        #self.optimizer = torch.optim.Adam(self.netWRN.parameters())


    def fit_trans_classifier(self, x_train, x_test, y_test):
        print("Training")
        # self.netWRN.train()
        self.model.train()
        bs = self.args.batch_size # 64
        N, sh, sw, = x_train.shape
        n_rots = self.n_trans #16
        m = self.args.m
        celoss = torch.nn.CrossEntropyLoss()
        ndf = 256

        for epoch in range(self.configs.num_epoch):
            rp = np.random.permutation(N//n_rots)
            rp = np.concatenate([np.arange(n_rots) + rp[i]*n_rots for i in range(len(rp))])
            assert len(rp) == N
            all_zs = torch.zeros((len(x_train), ndf)).cuda()
            diffs_all = []

            for i in range(0, len(x_train), bs):
                batch_range = min(bs, len(x_train) - i)
                idx = np.arange(batch_range) + i
                xs = torch.from_numpy(x_train[rp[idx]]).float().cuda()
                _, zs_tc, zs_ce = self.model(xs)

                all_zs[idx] = zs_tc
                train_labels = torch.from_numpy(np.tile(np.arange(n_rots), batch_range//n_rots)).long().cuda()
                zs = torch.reshape(zs_tc, (batch_range//n_rots, n_rots, ndf))

                means = zs.mean(0).unsqueeze(0)
                diffs = -((zs.unsqueeze(2).detach().cpu().numpy() - means.unsqueeze(1).detach().cpu().numpy()) ** 2).sum(-1)
                diffs_all.append(torch.diagonal(torch.tensor(diffs), dim1=1, dim2=2))

                tc = tc_loss(zs, m)
                ce = celoss(zs_ce, train_labels)
                if self.args.reg:
                    loss = ce + self.args.lmbda * tc + 10 *(zs*zs).mean()
                else:
                    loss = ce + self.args.lmbda * tc

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            self.model.eval()
            all_zs = torch.reshape(all_zs, (N//n_rots, n_rots, ndf))
            means = all_zs.mean(0, keepdim=True)


        with torch.no_grad():
            batch_size = bs
            val_probs_rots = np.zeros((len(y_test), self.n_trans))
            for i in range(0, len(x_test), batch_size):
                batch_range = min(batch_size, len(x_test) - i)
                idx = np.arange(batch_range) + i
                xs = torch.from_numpy(x_test[idx]).float().cuda()

                _, zs, fs = self.model(xs)
                zs = torch.reshape(zs, (batch_range // n_rots, n_rots, ndf))

                diffs = ((zs.unsqueeze(2) - means) ** 2).sum(-1)
                diffs_eps = self.args.eps * torch.ones_like(diffs)
                diffs = torch.max(diffs, diffs_eps)
                logp_sz = torch.nn.functional.log_softmax(-diffs, dim=2)

                zs_reidx = np.arange(batch_range // n_rots) + i // n_rots
                val_probs_rots[zs_reidx] = -torch.diagonal(logp_sz, 0, 1, 2).cpu().data.numpy()

            val_probs_rots = val_probs_rots.sum(1)
            print("Epoch:", epoch, ", AUC: ", roc_auc_score(y_test, -val_probs_rots))

            scores = (-val_probs_rots).tolist()
            labels = y_test.tolist()
                       

        return scores, labels

