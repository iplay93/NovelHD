
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

import math

class TimeSeriesNet(nn.Module):
    def __init__(self, seq_length=100, input_channels = 1):
        super(TimeSeriesNet, self).__init__()
        self.rep_dim = 32

        encoder_layers = TransformerEncoderLayer(seq_length, dim_feedforward=2*seq_length, nhead=1, )
        self.transformer_encoder = TransformerEncoder(encoder_layers, 2)

        self.projector = nn.Sequential(
            nn.Linear(seq_length * input_channels, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, self.rep_dim)
        )

        self.positional_encoding = self.generate_positional_encoding(
            seq_length, input_channels
        )

    def generate_positional_encoding(self, seq_len, d_model):
        
        position = torch.arange(0, seq_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / d_model))

        pe = torch.sin(position * div_term[0])
        
        for i in range(1, d_model):

            if i % 2 == 0:
                pe = torch.cat([pe, torch.sin(position * div_term[(int)(i % 2)])], dim = 1)
            else:
                pe = torch.cat([pe, torch.cos(position * div_term[(int)(i % 2)])], dim = 1)
        
        #print("PE", pe.shape)
        return pe.permute(1,0)

    def forward(self, x_in):

        x_in = x_in + self.positional_encoding.unsqueeze(0).expand(x_in.shape[0], -1, -1).cuda()
        """Use Transformer"""
        x = self.transformer_encoder(x_in.float())
        h = x.reshape(x.shape[0], -1)

        """Cross-space projector"""
        z = self.projector(h)


        return z

# class TimeSeriesNet(nn.Module):
#     def __init__(self, seq_length=100, input_channels = 1):
#         super(TimeSeriesNet, self).__init__()
#         self.seq_length = seq_length
#         self.input_dim = input_channels
#         self.rep_dim = 32

#         # Encoder layers
#         self.encoder = nn.Sequential(
#             nn.Conv1d(self.input_dim, 16, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool1d(2, stride=2),
#             nn.Conv1d(16, 8, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool1d(2, stride=2),
#             nn.Conv1d(8, 4, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool1d(2, stride=2)
#         )
#         # Linear layer in the encoder
#         self.encoder_linear = nn.Linear(4 * (self.seq_length // 8), self.rep_dim)

#     def forward(self, x):
#         encoded = self.encoder(x)
#         encoded = encoded.view(encoded.size(0), -1)
#         encoded = self.encoder_linear(encoded)

#         return encoded


class TimeSeriesAutoencoder(nn.Module):
    def __init__(self, seq_length=100, input_channels = 1):
        super(TimeSeriesAutoencoder, self).__init__()
        self.seq_length = seq_length
        self.input_dim = input_channels

        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Conv1d(self.input_dim, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),
            nn.Conv1d(16, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),
            nn.Conv1d(8, 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2)
        )

        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='linear'),
            nn.Conv1d(4, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='linear'),
            nn.Conv1d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(size = seq_length, mode='linear'),
            nn.Conv1d(16, input_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


