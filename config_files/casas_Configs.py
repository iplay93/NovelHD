class Config(object):
    def __init__(self):
        # model configs
        self.input_channels = 37
        self.input_channels_2 = 37
        self.kernel_size = 6
        self.stride = 1
        self.final_out_channels = 64

        self.num_classes = 15
        self.dropout = 0.35
        self.features_len = 8

        # training configs
        self.num_epoch = 100

        # optimizer parameters
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.lr = 3e-4

        # data parameters
        self.drop_last = True
        self.batch_size = 64

        self.Context_Cont = Context_Cont_configs()
        self.TC = TC()
        self.augmentation = augmentations()
        
        """New hyperparameters"""
        self.TSlength_aligned = 46 #45(BA) #46
        self.TSlength_aligned_2 = 46
        self.lr_f = self.lr
        self.target_batch_size = 64#  84
        self.increased_dim = 1
        self.final_out_channels = 64
        self.num_classes_target = 15
        self.features_len_f = self.features_len
        self.CNNoutput_channel = 28#  104
        
        self.ST = [
            ['Convolve', 'Crop','Drift','Pool','Quantize','Resize','Reverse'],
            [ 'Crop',  'Dropout', 'Pool', 'Quantize', 'Reverse'],
            [ 'Convolve', 'Crop', 'Drift', 'Dropout', 'Pool', 'Quantize', 'Resize', 'Reverse', 'TimeWarp'],
            ['Convolve', 'Crop', 'Drift','Pool', 'Quantize', 'Reverse'],
            ['Convolve', 'Crop', 'Drift','Pool', 'Quantize', 'Reverse', 'TimeWarp'],
            ['Convolve', 'Crop', 'Drift','Pool', 'Quantize', 'Reverse']
        ]
        self.ST_f  = [
            ['Crop', 'Drift','Pool','Quantize', 'Reverse'],
            ['Convolve', 'Crop', 'Drift','Pool', 'Quantize', 'Reverse'],
            ['Convolve', 'Crop', 'Drift','Pool', 'Quantize', 'Reverse'],
            ['Convolve', 'Crop', 'Drift','Pool', 'Quantize', 'Reverse'],
            ['Convolve', 'Crop', 'Drift','Pool', 'Quantize', 'Reverse', 'TimeWarp'],
            ['Convolve', 'Crop', 'Drift','Pool', 'Quantize', 'Reverse']
        ]

class augmentations(object):
    def __init__(self):
        self.jitter_scale_ratio = 1.1
        self.jitter_ratio = 0.8
        self.max_seg = 8


class Context_Cont_configs(object):
    def __init__(self):
        self.temperature = 0.2
        self.use_cosine_similarity = True


class TC(object):
    def __init__(self):
        self.hidden_dim = 100
        self.timesteps = 6
