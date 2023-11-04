from torch import nn
import torch
from .Transformer import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F
import numpy as np
import seaborn
import matplotlib.pyplot as plt

def draw(numpy_arrays):
    #https://nlp.seas.harvard.edu/2018/04/03/attention.html
    
    fig, axs = plt.subplots(1,2, figsize=(20, 10))
    for idx, arr in enumerate(numpy_arrays):
        #print(arr)
        
        seaborn.heatmap(arr[0], 
                    xticklabels=list(range(1, arr[0].shape[0]+1)), square=True, yticklabels=list(range(1, arr[0].shape[1]+1)), 
                    cbar=False, ax=axs[idx])
        
    plt.show()
    plt.savefig("example.png")

"""Two contrastive encoders"""
class TFC(nn.Module):
    def __init__(self, configs, args):
        super(TFC, self).__init__()

        encoder_layers_t = TransformerEncoderLayer(configs.input_channels, dim_feedforward=2*configs.input_channels, nhead=1, batch_first = True)
        self.transformer_encoder_t = TransformerEncoder(encoder_layers_t, 2)

        self.projector_t = nn.Sequential(
            nn.Linear(configs.TSlength_aligned * configs.input_channels, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        self.shift_cls_layer_t = nn.Linear(configs.TSlength_aligned * configs.input_channels, args.K_shift)

        encoder_layers_f = TransformerEncoderLayer(configs.input_channels, dim_feedforward=2*configs.input_channels, nhead=1, batch_first = True)
        self.transformer_encoder_f = TransformerEncoder(encoder_layers_f, 2)

        self.projector_f = nn.Sequential(
            nn.Linear(configs.TSlength_aligned * configs.input_channels, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )            
        self.shift_cls_layer_f = nn.Linear(configs.TSlength_aligned * configs.input_channels, args.K_shift_f)

        self.linear = nn.Linear(configs.TSlength_aligned * configs.input_channels, configs.num_classes)    


    def forward(self, x_in_t, x_in_f):

        x_in_t = x_in_t.permute(0,2,1)
        """Use Transformer"""
        x, attention_maps_t = self.transformer_encoder_t(x_in_t.float())
        h_time = x.reshape(x.shape[0], -1)
        # print(attention_maps)
        # attention_maps_cpu = [tensor.cpu() for tensor in attention_maps]
        # numpy_arrays = [tensor.detach().numpy() for tensor in attention_maps_cpu]
        # draw(numpy_arrays)

        """Cross-space projector"""
        z_time = self.projector_t(h_time)

        """Shifted transformation classifier"""
        s_time = self.shift_cls_layer_t(h_time)

        x_in_f = x_in_f.permute(0,2,1)
        """Frequency-based contrastive encoder"""
        f, attention_maps_f = self.transformer_encoder_f(x_in_f.float())
        h_freq = f.reshape(f.shape[0], -1)
        # attention_maps_cpu = [tensor.cpu() for tensor in attention_maps]
        # numpy_arrays = [tensor.detach().numpy() for tensor in attention_maps_cpu]
        # for arr in numpy_arrays:
        #     print(arr.shape)

        """Cross-space projector"""
        z_freq = self.projector_f(h_freq)

        """Shifted transformation classifier"""
        s_freq = self.shift_cls_layer_f(h_freq)

        return h_time, z_time, s_time, h_freq, z_freq, s_freq, attention_maps_t, attention_maps_f
