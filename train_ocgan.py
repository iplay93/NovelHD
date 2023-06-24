#https://github.com/PramuPerera/OCGAN/tree/master
from __future__ import print_function
import os
import matplotlib as mpl
import tarfile
import matplotlib.image as mpimg
from matplotlib import pyplot as plt
import mxnet as mx
from mxnet import gluon
from mxnet import ndarray as nd
from mxnet.gluon import nn, utils
from mxnet.gluon.nn import Dense, Activation, Conv2D, Conv2DTranspose, \
    BatchNorm, LeakyReLU, Flatten, HybridSequential, HybridBlock, Dropout
from mxnet import autograd
import numpy as np
import random
from random import shuffle
import dataloaderiter as dload
import load_image
import visual
import OCGAN.models as models
from datetime import datetime
import time
import logging
import argparse
import models

import OCGAN.andgan as andgan

import argparse
def train_options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expname", default="expce", help="Name of the experiment")
    parser.add_argument("--batch_size", default=512, type=int, help="Batch size per iteration")
    parser.add_argument("--epochs", default=201, type=int,
                        help="Number of epochs for training")
    parser.add_argument("--use_gpu", default=1, type=int,  help="1 to use GPU  ")
    parser.add_argument("--dataset", default="Caltech256",
                        help="Specify the training dataset  ")
    parser.add_argument("--lr", default="0.0002", type=float, help="Base learning rate")
    parser.add_argument("--ngf", default=64, type=int, help="Number of base filters in Generator")
    parser.add_argument("--ndf", default=8, type=int, help="Number of base filters in Discriminator")
    parser.add_argument("--beta1", default=0.5, type=float, help="Parameter for Adam")
    parser.add_argument("--lambda1", default=500, type=float, help="Weight of reconstruction loss")
    parser.add_argument("--datapath", default='/users/pramudi/Documents/data/', help="Data path")
    parser.add_argument("--img_wd", default=61, type=int, help="Image width")
    parser.add_argument("--img_ht", default=61, type=int, help="Image height")
    parser.add_argument("--continueEpochFrom", default=-1,
                        help="Continue training from specified epoch")
    parser.add_argument("--noisevar", default=0.02, type=float, help="variance of noise added to input")
    parser.add_argument("--depth", default=3, type=int, help="Number of core layers in Generator/Discriminator")
    parser.add_argument("--seed", default=-1, type=float, help="Seed generator. Use -1 for random.")
    parser.add_argument("--append", default=0, type=int, help="Append discriminator input. 1 for true")
    parser.add_argument("--classes", default="", help="Name of training class. Keep blank for random")
    parser.add_argument("--latent", default=16, type=int,  help="Dimension of the latent space.")
    parser.add_argument("--ntype", default=4, type=int, help="Novelty detector: 1 - AE 2 - ALOCC 3 - latentD 4 - OCGAN")
    parser.add_argument("--protocol", default=1, type=int, help="1 : 80/20 split, 2 : Train / Test split")
    args = parser.parse_args()
    if args.use_gpu == 1:
        args.use_gpu = True
    else:
        args.use_gpu = False
    if args.append == 1:
        args.append = True
    else:
        args.append = False
    return args


def plotloss(loss_vec, fname):
    plt.gcf().clear()
    plt.plot(loss_vec[0], label="Dr", alpha = 0.7)
    plt.plot(loss_vec[4], label="Dl", alpha = 0.7)
    plt.plot(loss_vec[1], label="G", alpha=0.7)
    plt.plot(loss_vec[2], label="R", alpha= 0.7)
    plt.plot(loss_vec[3], label="Acc", alpha = 0.7)
    plt.legend()
    plt.savefig(fname)


def main(opt):
    if opt.seed != -1:
        random.seed(opt.seed)
    ctx = mx.gpu() if opt.use_gpu else mx.cpu()
    inclasspaths , inclasses = dload.loadPaths(opt)
    train_data, val_data = load_image.load_image(inclasspaths, opt)
    print('Data loading done.')
    networks = models.set_network(opt, ctx, False)
    print('training')
    # train networks based on opt.ntype(1 - AE 2 - ALOCC  4 - adnov)
    if opt.ntype == 4:
        loss_vec = andgan.trainadnov(opt, train_data, val_data, ctx, networks)
    elif opt.ntype == 2:
        loss_vec = andgan.traincvpr18(opt, train_data, val_data, ctx, networks)
    elif opt.ntype == 1:
        loss_vec = andgan.trainAE(opt, train_data, val_data, ctx, networks)
    plotloss(loss_vec, 'outputs/'+opt.expname+'_loss.png')
    return inclasses


if __name__ == "__main__":
    opt = train_options()
    inclasses = main(opt)
