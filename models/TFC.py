from torch import nn
import torch
from torch.nn import TransformerEncoder, TransformerEncoderLayer

"""Two contrastive encoders"""
class TFC(nn.Module):
    def __init__(self, configs, args):
        super(TFC, self).__init__()

        encoder_layers_t = TransformerEncoderLayer(configs.TSlength_aligned, dim_feedforward=2*configs.TSlength_aligned, nhead=1, )
        self.transformer_encoder_t = TransformerEncoder(encoder_layers_t, 2)

        self.projector_t = nn.Sequential(
            nn.Linear(configs.TSlength_aligned * configs.input_channels, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        self.shift_cls_layer_t = nn.Linear(configs.TSlength_aligned * configs.input_channels, args.K_shift)

        encoder_layers_f = TransformerEncoderLayer(configs.TSlength_aligned, dim_feedforward=2*configs.TSlength_aligned,nhead=1,)
        self.transformer_encoder_f = TransformerEncoder(encoder_layers_f, 2)

        self.projector_f = nn.Sequential(
            nn.Linear(configs.TSlength_aligned * configs.input_channels_2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )                   
        self.shift_cls_layer_f = nn.Linear(configs.TSlength_aligned * configs.input_channels_2, args.K_shift)

        self.linear = nn.Linear(configs.TSlength_aligned * configs.input_channels, configs.num_classes)    
        # Positional Encoding
        self.positional_encoding = self.generate_positional_encoding(configs.TSlength_aligned, configs.input_channels)


    def forward(self, x_in_t, x_in_f):
        #x_in_t = x_in_t + self.positional_encoding.T.cuda()
        #x_in_f = x_in_f + self.positional_encoding.T.cuda()

        """Use Transformer"""
        x = self.transformer_encoder_t(x_in_t.float())
        h_time = x.reshape(x.shape[0], -1)

        """Cross-space projector"""
        z_time = self.projector_t(h_time)

        """Shifted transformation classifier"""
        s_time = self.shift_cls_layer_t(h_time)

        """Frequency-based contrastive encoder"""
        f = self.transformer_encoder_f(x_in_f.float())
        h_freq = f.reshape(f.shape[0], -1)

        """Cross-space projector"""
        z_freq = self.projector_f(h_freq)

        """Shifted transformation classifier"""
        s_freq = self.shift_cls_layer_f(h_freq)

        return h_time, z_time, s_time, h_freq, z_freq, s_freq

    def generate_positional_encoding(self, seq_length, input_channels):
        positional_encoding = torch.zeros(seq_length, input_channels)
        positions = torch.arange(0, seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, input_channels, 2).float() * (-torch.log(torch.tensor(10000.0)) / input_channels))
        positional_encoding[:, :input_channels:2] = torch.sin(positions * div_term)
        positional_encoding[:, 1:input_channels:2] = torch.cos(positions[:, :input_channels//2] * div_term[:input_channels//2])
        return positional_encoding




"""Downstream classifier only used in finetuning"""
class target_classifier(nn.Module):
    def __init__(self, configs):
        super(target_classifier, self).__init__()
        self.logits = nn.Linear(2*128, 64)
        #self.logits_simple = nn.Linear(64, configs.num_classes)
        self.logits_simple = nn.Linear(64, 2)

    def forward(self, emb):
        emb_flat = emb.reshape(emb.shape[0], -1)
        emb = torch.sigmoid(self.logits(emb_flat))
        pred = self.logits_simple(emb)
        return pred

"""One contrastive encoders"""
class TFC_GOAD(nn.Module):
    def __init__(self, configs, num_classes):
        super(TFC_GOAD, self).__init__()

        encoder_layers = TransformerEncoderLayer(configs.TSlength_aligned, dim_feedforward=2*configs.TSlength_aligned, nhead=1, )
        self.transformer_encoder = TransformerEncoder(encoder_layers, 2)

        self.projector = nn.Sequential(
            nn.Linear(configs.TSlength_aligned * configs.input_channels, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 256)
        )

        self.linear = nn.Linear(256, num_classes)    


    def forward(self, x_in):

        """Use Transformer"""
        x = self.transformer_encoder(x_in.float())
        h = x.reshape(x.shape[0], -1)

        """Cross-space projector"""
        z = self.projector(h)

        """Shifted transformation classifier"""
        s = self.linear(z)

        return h, z, s
