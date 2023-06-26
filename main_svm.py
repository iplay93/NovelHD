import argparse
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from data_preprocessing.dataloader import loading_data
from torch.utils.data import DataLoader, Dataset


import torch
import torch.nn as nn
import torch.nn.functional as F


class CAE(nn.Module):
    def __init__(self, seq_length=100, input_dim=1, d_model=64, nhead=4, num_encoder_layers=2, num_decoder_layers=2):
        super(CAE, self).__init__()
        self.seq_length = seq_length
        self.input_dim = input_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)

        # Positional Encoding
        self.positional_encoding = self.generate_positional_encoding(seq_length, d_model)

        # Linear layer to map the decoder output to the original input dimension
        self.linear = nn.Linear(d_model, input_dim)

    def forward(self, x):
        x = x.unsqueeze(0)  # Add a batch dimension

        # Adjust positional encoding shape
        positional_encoding = self.positional_encoding[:, :x.size(1)].unsqueeze(0)

        # Apply positional encoding to the input
        x = x + positional_encoding

        # Transformer Encoder
        encoded = self.encoder(x)

        # Transformer Decoder
        decoded = self.decoder(encoded)

        # Linear layer to map the decoder output to the original input dimension
        decoded = self.linear(decoded.squeeze(0))

        return decoded

    def generate_positional_encoding(self, seq_length, d_model):
        positional_encoding = torch.zeros(seq_length, d_model)
        positions = torch.arange(0, seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        positional_encoding[:, 0::2] = torch.sin(positions * div_term)
        positional_encoding[:, 1::2] = torch.cos(positions * div_term)
        return positional_encoding


def parse_args():
    parser = argparse.ArgumentParser(description='Train Convolutional AutoEncoder and inference')
    parser.add_argument('--data_path', default='./data/cifar10.npz', type=str, help='path to dataset')
    parser.add_argument('--height', default=1, type=int, help='height of images')
    parser.add_argument('--width', default=32, type=int, help='width of images')
    parser.add_argument('--channel', default=3, type=int, help='channel of images')
    parser.add_argument('--num_epoch', default=100, type=int, help='the number of epochs')
    parser.add_argument('--batch_size', default=64, type=int, help='mini batch size')
    parser.add_argument('--output_path', default='./data/cifar10_cae.npz', type=str, help='path to directory to output')
    
    #for 
    parser.add_argument('--padding', type=str, 
                    default='mean', help='choose one of them : no, max, mean')
    parser.add_argument('--timespan', type=int, 
                        default=10000, help='choose of the number of timespan between data points(1000 = 1sec, 60000 = 1min)')
    parser.add_argument('--min_seq', type=int, 
                        default=10, help='choose of the minimum number of data points in a example')
    parser.add_argument('--min_samples', type=int, default=20, 
                        help='choose of the minimum number of samples in each label')
    parser.add_argument('--dataset', default='lapras', type=str,
                        help='Dataset of choice: lapras, casas, opportunity, aras_a, aras_b')
    parser.add_argument('--aug_method', type=str, default='AddNoise', help='choose the data augmentation method')
    parser.add_argument('--aug_wise', type=str, default='Temporal', help='choose the data augmentation wise')

    parser.add_argument('--test_ratio', type=float, default=0.2, help='choose the number of test ratio')

    args = parser.parse_args()

    return args

def flat_feature(enc_out):
    """flat feature of CAE features
    """
    enc_out_flat = []

    s1, s2 = enc_out[0].shape
    s = s1 * s2 
    for con in enc_out:
        enc_out_flat.append(con.reshape((s,)))

    return np.array(enc_out_flat)


def main_svm():
    """main function"""
    args = parse_args()
    seq_length = 598
    channel = 7
    num_epoch = 100
    batch_size = args.batch_size
    data_type = args.dataset
    output_path = './data/' + data_type + '_cae.npz'   

    if data_type == 'lapras': 
        args.timespan = 10000
        seq_length = 598
        channel = 7
    elif data_type == 'casas': 
        seq_length = 46
        channel = 37
    elif data_type == 'opportunity': 
        args.timespan = 1000
        seq_length = 169
        channel = 241
    elif data_type == 'aras_a': 
        args.timespan = 10000
        seq_length = 24
        channel = 19
    
    # load CIFAR-10 data from data directory
    #all_image, all_label = load_data(data_path)
    num_classes, datalist, labellist = loading_data(data_type, args)
    num_classes, datalist, labellist = torch.from_numpy(np.array(num_classes)), torch.from_numpy(np.array(datalist.cpu())),torch.from_numpy(np.array(labellist)).long()
    datalist = datalist.permute(0, 2, 1)
    # Make datalist shape 
    # Create an instance of the CAE model
    autoencoder = CAE(seq_length, channel)

    # Define the loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(autoencoder.parameters())

    # Create DataLoader
    dataloader = DataLoader(datalist, batch_size=batch_size, shuffle=True)    

    for epoch in range(num_epoch):
        running_loss = 0.0
        for inputs in dataloader:
            optimizer.zero_grad()
            outputs = autoencoder(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()


        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epoch}], Loss: {epoch_loss:.4f}")
 
    encoded_data = autoencoder.encoder(datalist)
    encoded_data = encoded_data.detach().numpy()


    # flat features for classification input
    enc_out = flat_feature(encoded_data)
    print(enc_out.shape)

    # save CAE output
    np.savez(output_path, ae_out=enc_out, labels=labellist)

#if __name__ == '__main__':
    #main()
    #start_test()