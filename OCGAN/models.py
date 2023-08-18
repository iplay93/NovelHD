import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import TransformerDecoder, TransformerDecoderLayer

def weights_init(mod):

    classname = mod.__class__.__name__
    if classname.find('Conv') != -1:
        mod.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        mod.weight.data.normal_(1.0, 0.02)
        mod.bias.data.fill_(0)

def set_network(args, device, istest, in_channel, layer_seq):
    
    netEn, netDe, netD, netD2, netDS = None, None, None, None, None

    # Make encoder and decoder
    netEn = Encoder(in_channel, layer_seq)
    netDe = Decoder(in_channel, layer_seq)
    netEn.apply(weights_init)
    netDe.apply(weights_init)
    
    # load networks based on opt.ntype (1 - AE 2 - ALOCC 3 - latentD 4 - adnov)
    
    if args.ntype > 1:
        netD = Discriminator(in_channel,layer_seq)
        netD.apply(weights_init)

    if args.ntype > 2:
        netD2 = LatentDiscriminator(in_channel)
        netD2.apply(weights_init)

    if args.ntype > 3:
        netDS = Discriminator(in_channel, layer_seq)
        netDS.apply(weights_init)

    #cl = Classifier(in_channel)
    #cl.apply(weights_init)

    return netEn, netDe, netD, netD2, netDS

def factorization(x):
    div_list=[]
    d = 2

    while d <= x:
        if x % d == 0:
            div_list.append(d)
            x = x / d
        else:
            d = d + 1

    return max(d)


class LatentDiscriminator(nn.Module):
    """
    Latent Discriminator Network for 7-channel data
    """
    # Input shape : [batch_size, channel, seq_len]
    # Output shape : [batch_size, 1]

    def __init__(self, channel):
        super(LatentDiscriminator, self).__init__()

        self.dense_1 = nn.Linear(16, 128)

        self.dense_2 = nn.Linear(128, 64)

        self.dense_3 = nn.Linear(64, 32)

        self.dense_4 = nn.Linear(32, 16)

        #self.dense_5 = nn.Linear(16, 1)
    

    def forward(self, input):

        #print("LatentDiscriminator input", input.shape)

        output = input.view(input.size(0),-1)
        output = F.relu(self.dense_1(output))
        output = F.relu(self.dense_2(output))
        output = F.relu(self.dense_3(output))
        #output = F.relu(self.dense_4(output))
        output = self.dense_4(output.detach())
        output = torch.sigmoid(output)

        #print("LatentDiscriminator output", input.shape)

        return output
    

# Define the PatchGAN discriminator
class Discriminator(nn.Module):
    # Input shape : [batch_size, channel, seq_len]
    # Output shape : [batch_size, 1]
    def __init__(self, channel, layer_seq):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv1d(channel, 32, layer_seq[0], stride=layer_seq[0])
        # self.activation = nn.Tanh()
        self.conv2 = nn.Conv1d(32, 32, layer_seq[1], stride=layer_seq[1]) 
        self.batch_norm_1 = nn.BatchNorm1d(32)
        #self.maxpooling_1 = nn.MaxPool1d(23, stride=23,padding=3//2)

        self.conv3 = nn.Conv1d(32, 64, layer_seq[2], stride=layer_seq[2])
        self.batch_norm_2 = nn.BatchNorm1d(64)

        self.conv4 = nn.Conv1d(64, 128, 1)
        self.batch_norm_3 = nn.BatchNorm1d(128)

        # latent =16
        self.conv5 = nn.Conv1d(128, 32, 1)
        self.batch_norm_4 = nn.BatchNorm1d(32)
        #self.maxpooling_2 = nn.MaxPool1d(13, stride=13, padding=3//2)

        #self.conv5 = nn.Conv1d(64, 32, 2)
        self.conv6 = nn.Conv1d(32, 16, 1)
        self.pooling = nn.AdaptiveAvgPool1d(output_size=1)
        
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, input):

        #print("Discriminator input", input.shape)
        output = self.conv1(input) 
        output = self.leaky_relu(output) 
        #print('output1:',output.shape)

        output =self.conv2(output)
        output = self.batch_norm_1(output)
        output = self.leaky_relu(output) 
        #print('output2:',output.shape)

        output = self.conv3(output)
        output = self.batch_norm_2(output)
        output = self.leaky_relu(output) 
        #print('output3:',output.shape)

        output = self.conv4(output)
        output = self.batch_norm_3(output)
        output = self.leaky_relu(output) 
        #print('output4:',output.shape)

        output = self.conv5(output)
        output = self.batch_norm_4(output)
        output = self.leaky_relu(output) 
        #print('output5:',output.shape)
        

        output = torch.sigmoid(self.conv6(output))
        #print("Discriminator output", output.shape)

        #print('output5:',output.shape)
        #output = output.view(output.size(0), -1)

        return output



# Input shape : [batch_size, channel, seq_len]
# Output shape : [batch_size, 1]
class Encoder(nn.Module):
    """
    DCGAN ENCODER NETWORK
    """
    def __init__(self, channel, layer_seq):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv1d(channel, 32, layer_seq[0], stride=layer_seq[0])
        # self.activation = nn.Tanh()
        self.conv2 = nn.Conv1d(32, 32, layer_seq[1], stride=layer_seq[1])
        self.batch_norm_1 = nn.BatchNorm1d(32)
        #self.maxpooling_1 = nn.MaxPool1d(23, stride=23,padding=3//2)

        self.conv3 = nn.Conv1d(32, 64, layer_seq[2], stride=layer_seq[2])
        self.batch_norm_2 = nn.BatchNorm1d(64)

        self.conv4 = nn.Conv1d(64, 128, 1)
        self.batch_norm_3 = nn.BatchNorm1d(128)

        # latent =16
        self.conv5 = nn.Conv1d(128, 16, 1)
        
        #self.maxpooling_2 = nn.MaxPool1d(13, stride=13, padding=3//2)

        #self.conv5 = nn.Conv1d(64, 32, 2)
        #self.conv6 = nn.Conv1d(32, 32, 2, padding = 1)
        
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, input): 
        # print('input:',input.shape)  

        #print("Encoder input", input.shape)
        output = self.conv1(input) 
        output = self.leaky_relu(output) 
        #print('output1:',output.shape)

        output =self.conv2(output)
        output = self.batch_norm_1(output)
        output = self.leaky_relu(output) 
        #print('output2:',output.shape)

        output = self.conv3(output)
        output = self.batch_norm_2(output)
        output = self.leaky_relu(output) 
        #print('output3:',output.shape)

        output = self.conv4(output)
        output = self.batch_norm_3(output)
        output = self.leaky_relu(output) 
        #print('output4:',output.shape)
        

        output = torch.tanh(self.conv5(output))
        #print("encoder output", output.shape)


        #output = output.view(output.size(0),-1)
        #print('wwwwwwwwwwwwwwwwwwwwwwwww:',output.shape)

        return output

# Eecoder의 output을 input으로 
# Input shape : [batch_size, 1]
# Output shape : [batch_size, channel, seq_len]
    
class Decoder(nn.Module):

    def __init__(self, channel, layer_seq):
        super(Decoder, self).__init__()
        self.conv1 = nn.ConvTranspose1d(16, 128, 1)
        self.batch_norm_1 = nn.BatchNorm1d(128)

        self.conv2 = nn.ConvTranspose1d(128, 64, 1)
        self.batch_norm_2 = nn.BatchNorm1d(64) 

        self.conv3 = nn.ConvTranspose1d(64, 32, layer_seq[2], stride=layer_seq[2])
        self.batch_norm_3 = nn.BatchNorm1d(32)
        self.dropout = nn.Dropout(p=0.2)

        # self.activation = nn.Tanh()
        self.conv4 = nn.ConvTranspose1d(32, 32,  layer_seq[1], stride=layer_seq[1])
        self.batch_norm_4 = nn.BatchNorm1d(32)

        self.conv5 = nn.ConvTranspose1d(32, channel, layer_seq[0], stride=layer_seq[0])
        

    def forward(self, input):
        
        #print("Decoder input", input.shape)
        #print('l2_input:',input.size(0))
        #output = input.view(input.size(0),16,-1)
        #print('l2_reshape:',output.shape)
        output = self.conv1(input)
        output = self.batch_norm_1(output)
        output = F.relu(output)        
        #print('l2_output1:',output.shape)

        output = self.conv2(output) 
        output = self.batch_norm_2(output)
        output = F.relu(output)     
        #print('l2_output2:',output.shape)
        
        output = self.conv3(output)
        output = self.batch_norm_3(output)
        output = self.dropout(output)
        output = F.relu(output) 
        #print('l2_output3:',output.shape)   

        output = self.conv4(output)
        output = self.batch_norm_4(output)
        output = F.relu(output) 
        #print('l2_output4:',output.shape)       

       
        output = torch.tanh(self.conv5(output))
        
        #print("Decoder output", output.shape)

        return output


    
# Decoder의 output을 input으로 
# Input shape : [batch_size, channel, seq_len]
# Output shape : [batch_size, 1]
class Classifier(nn.Module):
    def __init__(self, channel):
        super(Classifier, self).__init__()
        self.conv1 = nn.Conv1d(channel, 32, 23, stride=23)
        # self.activation = nn.Tanh()
        self.conv2 = nn.Conv1d(32, 32, 13, stride=13)
        self.batch_norm_1 = nn.BatchNorm1d(32)
        #self.maxpooling_1 = nn.MaxPool1d(23, stride=23,padding=3//2)

        self.conv3 = nn.Conv1d(32, 64, 2, stride=2)
        self.batch_norm_2 = nn.BatchNorm1d(64)

        self.conv4 = nn.Conv1d(64, 128, 1)
        self.batch_norm_3 = nn.BatchNorm1d(128)

        # latent =16
        self.conv5 = nn.Conv1d(128, 32, 1)
        self.batch_norm_4 = nn.BatchNorm1d(32)
        #self.maxpooling_2 = nn.MaxPool1d(13, stride=13, padding=3//2)

        #self.conv5 = nn.Conv1d(64, 32, 2)
        self.conv6 = nn.Conv1d(32, 1, 1)
        self.pooling = nn.AdaptiveAvgPool1d(output_size=1)
        
        self.leaky_relu = nn.LeakyReLU(0.2)


    def forward(self, input):

        output = self.conv1(input) 
        output = self.leaky_relu(output) 
        #print('output1:',output.shape)

        output =self.conv2(output)
        output = self.batch_norm_1(output)
        output = self.leaky_relu(output) 
        #print('output2:',output.shape)

        output = self.conv3(output)
        output = self.batch_norm_2(output)
        output = self.leaky_relu(output) 
        #print('output3:',output.shape)

        output = self.conv4(output)
        output = self.batch_norm_3(output)
        output = self.leaky_relu(output) 
        #print('output4:',output.shape)

        output = self.conv5(output)
        output = self.batch_norm_4(output)
        output = self.leaky_relu(output) 
        #print('output5:',output.shape)
        

        output = torch.sigmoid(self.conv6(output))
        #print('output5:',output.shape)
        output = output.view(output.size(0), -1)

        return output



class TimeSeriesEncoder(nn.Module):
    def __init__(self, input_channels, seq_length):
        super(TimeSeriesEncoder, self).__init__()
        self.rep_dim = 32

        encoder_layers = TransformerEncoderLayer(seq_length, dim_feedforward=2*seq_length, nhead=1, )
        self.transformer_encoder = TransformerEncoder(encoder_layers, 1)

        self.projector = nn.Sequential(
            nn.Linear(seq_length * input_channels, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, self.rep_dim)
        )


    def forward(self, x_in):

        #print("input", x_in.shape)
        """Use Transformer"""
        x = self.transformer_encoder(x_in.float())
        h = x.reshape(x.shape[0], -1)
        #print("h", h.shape)
        
        """Cross-space projector"""
        z = self.projector(h)
        print("encoder output", z.shape)

        return z
    
    
class TimeSeriesDecoder(nn.Module):
    def __init__(self, input_channels, seq_length):
        super(TimeSeriesDecoder, self).__init__()
        self.rep_dim = 32
        
        self.projector = nn.Sequential(
            nn.Linear(self.rep_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, seq_length * input_channels)
        )
        
        decoder_layers = TransformerDecoderLayer(seq_length, nhead=1)
        self.transformer_decoder = TransformerDecoder(decoder_layers, 1)
        
        self.seq_length = seq_length
        self.input_channels = input_channels

    def forward(self, z):

        #print("decoder")

        #print("input", z.shape)
        #output = z.view(z.size(0), self.rep_dim)
        """Cross-space projector inverse"""
        h = self.projector(z)
        #print("h", h.shape)
        x = h.reshape(-1, self.input_channels, self.seq_length)
        
        """Use Transformer decoder"""
        x_out = self.transformer_decoder(x, x)
       # print("decoder output", x_out.shape)

        return x_out
    
class TimeSeriesEncoderClassifier(nn.Module):
    def __init__(self, input_channels, seq_length):
        super(TimeSeriesEncoderClassifier, self).__init__()
        self.rep_dim = 32

        encoder_layers = nn.TransformerEncoderLayer(seq_length, dim_feedforward=2 * seq_length, nhead=1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, 1)

        self.projector = nn.Sequential(
            nn.Linear(seq_length * input_channels, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.rep_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x_in):

        """Use Transformer"""
        x = self.transformer_encoder(x_in.float())
        h = x.reshape(x.shape[0], -1)

        """Cross-space projector"""
        z = self.projector(h)

        """Classifier"""
        logits = torch.sigmoid(z)

        return logits
    

class TSLatentDiscriminator(nn.Module):
    """
    Latent Discriminator Network for 7-channel data
    """
    # Input shape : [batch_size, channel, seq_len]
    # Output shape : [batch_size, 1]

    def __init__(self, in_channel, seq_len):
        super(TSLatentDiscriminator, self).__init__()

        self.dense_1 = nn.Linear(32, 128)

        self.dense_2 = nn.Linear(128, 64)

        self.dense_3 = nn.Linear(64, 32)

        self.dense_4 = nn.Linear(32, 16)

        self.dense_5 = nn.Linear(16, 1)
    

    def forward(self, input):

        output = input.view(input.size(0),-1)
        output = F.relu(self.dense_1(output))
        output = F.relu(self.dense_2(output))
        output = F.relu(self.dense_3(output))
        output = F.relu(self.dense_4(output))
        output = self.dense_5(output.detach())
        output = torch.sigmoid(output)

        print("latent discriminator output", output.shape)
    
        return output
    


class TimeSeriesDiscriminator(nn.Module):
    def __init__(self, input_channels, seq_length):
        super(TimeSeriesDiscriminator, self).__init__()
        
        self.rep_dim = 32

        encoder_layers = nn.TransformerEncoderLayer(seq_length, dim_feedforward=2 * seq_length, nhead=1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, 1)

        self.projector = nn.Sequential(
            nn.Linear(seq_length * input_channels, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.rep_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x_in):

        """Use Transformer"""
        x = self.transformer_encoder(x_in.float())
        h = x.reshape(x.shape[0], -1)

        """Cross-space projector"""
        z = self.projector(h)

        """Classifier"""
        logits = torch.sigmoid(z)

        print("discriminator output", logits.shape)

        return logits