class Config(object):
    def __init__(self):
        # model configs
        self.input_channels = 19
        self.input_channels_2 = 19
        self.kernel_size = 6
        self.stride = 1
        self.final_out_channels = 32

        self.num_classes = 27
        self.dropout = 0.35
        self.features_len = 5

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

                #for ND
        self.hidden_size = 64
        self.num_layers = 3
        self.project_channels = 20

        self.freeze_length_epoch = 2
        self.change_center_epoch = 1

        self.center_eps = 0.1
        self.omega1 = 1
        self.omega2 = 0.1

        # optimizer parameters
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.lr = 1e-4

        # Anomaly Detection parameters
        self.nu = 0.001
        # Anomaly quantile of fixed threshold
        self.detect_nu = 0.0015
        # Methods for determining thresholds ("fix","floating","one-anomaly")
        self.threshold_determine = 'floating'
        # Specify COCA objective ("one-class" or "soft-boundary")
        self.objective = 'one-class'
        # Specify loss objective ("arc1","arc2","mix","no_reconstruction", or "distance")
        self.loss_type = 'distance'

        """New hyperparameters"""
        self.TSlength_aligned = 33
        self.lr_f = self.lr
        self.target_batch_size = 64#  84
        self.increased_dim = 1
        self.final_out_channels = 128
        self.num_classes_target = 27
        self.features_len_f = self.features_len
        self.CNNoutput_channel = 28#  104


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
        self.timesteps = 2
