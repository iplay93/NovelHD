
import torch
import torch.nn as nn
import torch.nn.functional as F

class TimeSeriesNet(nn.Module):
    def __init__(self, seq_length=100, input_channels = 1):
        super(TimeSeriesNet, self).__init__()
        self.seq_length = seq_length
        self.input_dim = input_channels
        self.rep_dim = 32

        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),
            nn.Conv1d(16, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),
            nn.Conv1d(8, 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2)
        )
        # Linear layer in the encoder
        self.encoder_linear = nn.Linear(4 * (seq_length // 8), self.rep_dim)

    def forward(self, x):
        encoded = self.encoder(x)
        encoded = encoded.view(encoded.size(0), -1)
        encoded = self.encoder_linear(encoded)

        return encoded


class TimeSeriesAutoencoder(nn.Module):
    def __init__(self, seq_length=100, input_channels = 1):
        super(TimeSeriesAutoencoder, self).__init__()
        self.seq_length = seq_length
        self.input_dim = input_channels

        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, 16, kernel_size=3, padding=1),
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


