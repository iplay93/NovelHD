from torch import nn

class base_Model(nn.Module):
    def __init__(self, configs):
        super(base_Model, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(configs.input_channels, 32, kernel_size=configs.kernel_size,
                      stride=configs.stride, bias=False, padding=(configs.kernel_size//2)),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(configs.dropout)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(64, configs.final_out_channels, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(configs.final_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

        model_output_dim = configs.features_len
        last_dim = model_output_dim * configs.final_out_channels
        self.logits = nn.Linear(last_dim, configs.num_classes)
        self.simclr_layer = nn.Sequential(
            nn.Linear(last_dim, last_dim),
            nn.ReLU(),
            nn.Linear(last_dim, 128),
        )
        self.shift_cls_layer = nn.Linear(last_dim, 2)

    def forward(self, x_in):
        _aux = {}
        _return_aux = False
        x = self.conv_block1(x_in)
        #print(x.shape)
        x = self.conv_block2(x)
        #print(x.shape)
        x = self.conv_block3(x)
        #print(x.shape)

        x_flat = x.reshape(x.shape[0], -1)
        #(x_flat.shape)
        logits = self.logits(x_flat)
        return logits, x

# Previous experiments
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import math

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x




class TransformerEncoder(nn.Module):
    def __init__(self, *, num_classes, feature_dim, dim, depth=12, kernel_size= 10, stride = 5,
    heads=2, mlp_dim =64, pool = 'cls', dim_head = 64, dropout = 0., emb_dropout = 0.2):
        super().__init__()

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(nn.Conv1d(feature_dim, dim, kernel_size=kernel_size, stride =stride))        
        #self.pos_embedding = nn.Parameter(torch.randn(1, time_length+ 1, dim))
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        #self.pos_embedding = nn.Parameter(torch.randn(dim,dim))  
        
        self.div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
        
        self.dim = dim
        self.pool = pool
        self.to_latent = nn.Identity()
        

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, data):
        
        x = torch.transpose(data,1,2)
        x = self.to_patch_embedding(x)
        x = torch.transpose(x,1,2)
        #self.pos_embedding = nn.Parameter(torch.randn(1, x.shape[1]+1, self.dim))    

        # using sin and cos encoding (from original encoding)
        position = torch.arange(x.shape[1]).unsqueeze(1)
        self.pos_embedding = torch.zeros(1, x.shape[1], self.dim)        
        self.pos_embedding [0, :, 0::2] = torch.sin(position * self.div_term)
        self.pos_embedding [0, :, 1::2] = torch.cos(position * self.div_term)

        #b, n, _ = x.shape
        #cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)     
        #x = torch.cat((cls_tokens, x), dim=1)
        
        #
        x += self.pos_embedding.cuda()
        x = self.dropout(x)
        x = self.transformer(x)
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]        
        x = self.to_latent(x)
        
        return self.mlp_head(x)