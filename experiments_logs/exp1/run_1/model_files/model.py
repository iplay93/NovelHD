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
