class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/root/autodl-fs/VSPT'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = '/root/autodl-fs/VSPT/tensorboard'    # Directory for tensorboard files.
        self.pretrained_networks = '/root/autodl-fs/VSPT/pretrained_models'
        self.lsotb_dir = '/root/autodl-tmp/data/train_data/LSOTB_TIR_train'
        self.lvis_dir = ''
        self.sbd_dir = ''
        self.imagenetdet_dir = ''
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''
        self.davis_dir = ''
        self.youtubevos_dir = ''
