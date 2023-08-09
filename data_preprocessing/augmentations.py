import numpy as np
import torch
from tsaug import *
import random

def select_transformation(aug_method):
    if(aug_method == 'AddNoise'):
        my_aug = (AddNoise(scale=0.01))
    elif(aug_method == 'Convolve'):
        my_aug = Convolve(window="flattop", size=11)
    elif(aug_method == 'Crop'):
        my_aug = PERMUTE(min_segments=10, max_segments=15, seg_mode="random")
    #     my_aug = (Crop(size = target_len))
    elif(aug_method == 'Drift'):
        my_aug = (Drift(max_drift=0.8, n_drift_points=10))
    elif(aug_method == 'Dropout'):
        my_aug = (Dropout(p=0.1,fill=0))        
    elif(aug_method == 'Pool'):
        #my_aug = (Pool(size=2))
        my_aug = (Pool(size=10))
    elif(aug_method == 'Quantize'):
        #my_aug = (Quantize(n_levels=20))
        my_aug = (Quantize(n_levels=3))
    elif(aug_method == 'Resize'):
        #my_aug = SCALE(sigma=1.1, loc = 1.3)
        my_aug = SCALE(sigma=0.8, loc = 1.5)
        #my_aug = (Resize(size = target_len))
    elif(aug_method == 'Reverse'):
        my_aug = (Reverse())
    elif(aug_method == 'TimeWarp'):
        my_aug = (TimeWarp(n_speed_change=5, max_speed_ratio=3))
    elif(aug_method == 'AddNoise2'):
        my_aug = (AddNoise(scale=0.05))
    else:
        return ValueError
        
    return my_aug

def one_hot_encoding(X):
    X = [int(x) for x in X]
    n_values = np.max(X) + 1
    b = np.eye(n_values)[X]
    return b

def DataTransform(sample, config):
    """Weak and strong augmentations"""
    weak_aug = scaling(sample, config.augmentation.jitter_scale_ratio)
    # weak_aug = permutation(sample, max_segments=config.augmentation.max_seg)
    strong_aug = jitter(permutation(sample, max_segments=config.augmentation.max_seg), config.augmentation.jitter_ratio)

    return weak_aug, strong_aug

# def DataTransform_TD(sample, config):
#     """Weak and strong augmentations"""
#     weak_aug = sample
#     strong_aug = jitter(permutation(sample, max_segments=config.augmentation.max_seg), config.augmentation.jitter_ratio) #masking(sample)
#     return weak_aug, strong_aug
#
# def DataTransform_FD(sample, config):
#     """Weak and strong augmentations in Frequency domain """
#     # weak_aug =  remove_frequency(sample, 0.1)
#     strong_aug = add_frequency(sample, 0.1)
#     return weak_aug, strong_aug
def DataTransform_TD(sample, config):
    """Simplely use the jittering augmentation. Feel free to add more autmentations you want,
    but we noticed that in TF-C framework, the augmentation has litter impact on the final tranfering performance."""
    aug = jitter(sample, config.augmentation.jitter_ratio)
    return aug



def DataTransform_TD_bank(sample, config):
    """Augmentation bank that includes four augmentations and randomly select one as the positive sample.
    You may use this one the replace the above DataTransform_TD function."""
    aug_1 = jitter(sample, config.augmentation.jitter_ratio)
    aug_2 = scaling(sample, config.augmentation.jitter_scale_ratio)
    aug_3 = permutation(sample, max_segments=config.augmentation.max_seg)
    aug_4 = masking(sample, keepratio=0.9)

    li = np.random.randint(0, 4, size=[sample.shape[0]])
    li_onehot = one_hot_encoding(li)
    aug_1 = aug_1 * li_onehot[:, 0][:, None, None]  # the rows that are not selected are set as zero.
    aug_2 = aug_2 * li_onehot[:, 0][:, None, None]
    aug_3 = aug_3 * li_onehot[:, 0][:, None, None]
    aug_4 = aug_4 * li_onehot[:, 0][:, None, None]
    aug_T = aug_1 + aug_2 + aug_3 + aug_4
    return aug_T

def DataTransform_FD(sample, config):
    """Weak and strong augmentations in Frequency domain """
    aug_1 = remove_frequency(sample, pertub_ratio=0.1)
    aug_2 = add_frequency(sample, pertub_ratio=0.1)
    aug_F = aug_1 + aug_2
    return aug_F

def remove_frequency(x, pertub_ratio=0.0):
    mask = torch.cuda.FloatTensor(x.shape).uniform_() > pertub_ratio # maskout_ratio are False
    mask = mask.to(x.device)
    return x*mask

def add_frequency(x, pertub_ratio=0.0):

    mask = torch.cuda.FloatTensor(x.shape).uniform_() > (1-pertub_ratio) # only pertub_ratio of all values are True
    mask = mask.to(x.device)
    max_amplitude = x.max()
    random_am = torch.rand(mask.shape)*(max_amplitude*0.1)
    pertub_matrix = mask*random_am
    return x+pertub_matrix


def generate_binomial_mask(B, T, D, p=0.5): # p is the ratio of not zero
    return torch.from_numpy(np.random.binomial(1, p, size=(B, T, D))).to(torch.bool)

def masking(x, keepratio=0.9, mask= 'binomial'):
    global mask_id
    nan_mask = ~x.isnan().any(axis=-1)
    x[~nan_mask] = 0
    # x = self.input_fc(x)  # B x T x Ch

    if mask == 'binomial':
        mask_id = generate_binomial_mask(x.size(0), x.size(1), x.size(2), p=keepratio).to(x.device)
    # elif mask == 'continuous':
    #     mask = generate_continuous_mask(x.size(0), x.size(1)).to(x.device)
    # elif mask == 'all_true':
    #     mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
    # elif mask == 'all_false':
    #     mask = x.new_full((x.size(0), x.size(1)), False, dtype=torch.bool)
    # elif mask == 'mask_last':
    #     mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
    #     mask[:, -1] = False

    # mask &= nan_mask
    x[~mask_id] = 0
    return x

def jitter(x, sigma=0.8):
    # https://arxiv.org/pdf/1706.00527.pdf
    return x + np.random.normal(loc=0., scale=sigma, size=x.shape)

class SCALE():
    def __init__(self, sigma=1.1, loc = 1.3):
        self.sigma = sigma
        self.loc = loc
    def augment(self, x):
        # https://arxiv.org/pdf/1706.00527.pdf
        # loc -> multification #
        factor = np.random.normal(loc=self.loc, scale=self.sigma, size=(x.shape[0], x.shape[2]))
        ai = []
        for i in range(x.shape[1]):
            xi = x[:, i, :]
            ai.append(np.multiply(xi, factor[:, :])[:, np.newaxis, :])
        return np.concatenate((ai), axis=1)

class PERMUTE():   
    def __init__(self, min_segments=2, max_segments=15, seg_mode="random"):
        self.min = min_segments
        self.max = max_segments
        self.seg_mode = seg_mode

    def augment(self, x):
        # input : (N, T, C)
        # reshape과 swapaxes/transpose 유의
        orig_steps = np.arange(x.shape[1])

        num_segs = np.random.randint(self.min, self.max , size=(x.shape[0]))

        ret = np.zeros_like(x)
        
        # for each sample
        for i, pat in enumerate(x):
            if num_segs[i] > 1:
                if self.seg_mode == "random":
                    split_points = np.random.choice(x.shape[1] - 2, num_segs[i] - 1, replace=False)
                    split_points.sort()                
                    splits = np.split(orig_steps, split_points)
                else:
                    splits = np.array_split(orig_steps, num_segs[i])
                warp = np.concatenate(np.random.permutation(splits)).ravel()
                ret[i] = pat[warp, : ]
            else:
                ret[i] = pat

        #return torch.from_numpy(ret)     
        return ret   


    

# from GOAD

import abc
import itertools
import numpy as np
#from keras.preprocessing.image import apply_affine_transform
# The code is adapted from https://github.com/izikgo/AnomalyDetectionTransformations/blob/master/transformations.py

def get_transformer(args, config):
    if args.type_trans == 'complicated':
        #tr_x, tr_y = 8, 8
        transformer = Transformer(args, config)
        return transformer
    elif args.type_trans == 'simple':
        transformer = SimpleTransformer()
        return transformer


class AffineTransformation(object):
    def __init__(self, j, s, p, m, config, trans_list):
        self.j = j
        self.s = s
        self.p = p
        self.m = m
        self.config = config
        self.trans_list = trans_list
        print(self.trans_list)                    

    def __call__(self, x):
        res_x = x
        # shape of x (T,C)
        
        if self.j:
            #res_x = np.fliplr(res_x)
            #trans = select_transformation(self.trans_list[0])
            trans = PERMUTE(min_segments=2, max_segments=5, seg_mode="random")
            res_x = trans.augment(np.reshape(res_x,(1, res_x .shape[0], -1)))[0]
            #res_x = jitter(res_x, self.config.augmentation.jitter_ratio)
        if self.s:
            #trans = select_transformation(self.trans_list[1])
            trans = (AddNoise(scale=0.01))
            res_x = trans.augment(np.reshape(res_x,(1, res_x .shape[0], -1)))[0]
            trans  = SCALE(sigma=1.1, loc = 1.3)
            res_x = trans.augment(np.reshape(res_x,(1, res_x .shape[0], -1)))[0]
            #jitter(res_x, self.config.augmentation.jitter_ratio)
            #res_x = scaling(res_x, self.config.augmentation.jitter_scale_ratio)
        if self.p:
            #trans = select_transformation(self.trans_list[2])
            trans = (AddNoise(scale=0.01))
            res_x = trans.augment(np.reshape(res_x,(1, res_x .shape[0], -1)))[0]
            trans = PERMUTE(min_segments=2, max_segments=5, seg_mode="random")
            res_x = trans.augment(np.reshape(res_x,(1, res_x .shape[0], -1)))[0]
            #res_x = jitter(res_x, self.config.augmentation.jitter_ratio)
            #res_x = permutation(res_x, max_segments= self.config.augmentation.max_seg)
            #res_x = apply_affine_transform(res_x,
            #tx=self.tx, ty=self.ty, channel_axis=2, fill_mode='reflect')
        if self.m:
            trans = (AddNoise(scale=0.01))
            res_x = trans.augment(np.reshape(res_x,(1, res_x .shape[0], -1)))[0]
            trans = PERMUTE(min_segments=2, max_segments=5, seg_mode="random")
            res_x = trans.augment(np.reshape(res_x,(1, res_x .shape[0], -1)))[0]
            trans = SCALE(sigma=1.1, loc = 1.3)
            res_x = trans.augment(np.reshape(res_x,(1, res_x .shape[0], -1)))[0]
            #res_x = jitter(res_x, self.config.augmentation.jitter_ratio)
            #res_x = permutation(res_x, max_segments= self.config.augmentation.max_seg)
            #res_x = masking(res_x, keepratio=0.9)
            #res_x = np.rot90(res_x, self.k_90_rotate)
            #print("res_x", res_x.shape)
        return res_x


class AbstractTransformer(abc.ABC):
    def __init__(self):
        self._transformation_list = None
        self._create_transformation_list()

    @property
    def n_transforms(self):
        return len(self._transformation_list)

    @abc.abstractmethod
    def _create_transformation_list(self):
        return

    def transform_batch(self, x_batch, t_inds):
        assert len(x_batch) == len(t_inds)

        transformed_batch = x_batch.copy()
        
        for i, t_ind in enumerate(t_inds):
            #print(i, t_ind, transformed_batch[i].shape )
            transformed_batch[i] = self._transformation_list[t_ind](transformed_batch[i])
        return transformed_batch


class Transformer(AbstractTransformer):
    def __init__(self, args, config):
        #self.max_tx = translation_x
        #self.max_ty = translation_y
        self.config = config
        random.seed(args.seed)
        self.trans_list = random.sample(['AddNoise' ,'Convolve', 'Crop', 'Drift', 'Dropout', 'Pool', 
                    'Quantize', 'Resize', 'Reverse', 'TimeWarp'], 4)
        super().__init__()

    def _create_transformation_list(self):
        transformation_list = []
        for j, s, p, m in itertools.product((False, True), (False, True),
                                            (False, True), (False, True)):
            transformation = AffineTransformation(j, s, p, m, self.config, self.trans_list)
            transformation_list.append(transformation)
        self._transformation_list = transformation_list
        return transformation_list


class SimpleTransformer(AbstractTransformer):
    def _create_transformation_list(self):
        transformation_list = []
        for is_flip, k_rotate in itertools.product((False, True),
                                                    range(4)):
            transformation = AffineTransformation(is_flip, 0, 0, k_rotate)
            transformation_list.append(transformation)
        self._transformation_list = transformation_list
        return transformation_list
