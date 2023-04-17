import os
from yacs.config import CfgNode as CN
import yaml

_C = CN()
_C.BASE = ['']

_C.DATA = CN()
_C.DATA.name = 'MNIST'
_C.DATA.shape_in_c = 1
_C.DATA.shape_in_h = 28
_C.DATA.shape_in_w = 28
_C.DATA.num_classes = 10
_C.DATA.data_path = './dataset/'
_C.DATA.dataset_list_train = './dataset/train_{}.txt'
_C.DATA.dataset_list_test = './dataset/test_{}.txt'

_C.MODEL = CN()
_C.MODEL.initial_filters = 8
_C.MODEL.num_fc = 10
_C.MODEL.dropout_rate = 0.2
_C.MODEL.weight_path = "weights/weights.pt"

_C.TRAIN = CN()
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = 'Adam'
_C.TRAIN.epoch_num = 10
_C.TRAIN.batch_size = 512
_C.TRAIN.lr = 0.01

_C.TEST = CN()
_C.TEST.batch_size = 1500

_C.IosForest = CN()
# _C.IosForest.processing = 'ori'
# _C.IosForest.processing = 'PCA'
_C.IosForest.processing = 'PCA'
_C.IosForest.dataset_path = './dataset/{}/raw/test'
_C.IosForest.n_estimators = 100
_C.IosForest.contamination = 0.08
_C.IosForest.sensitive_points_save_path = './SensitivePoints/sp'

_C.attack = CN()
_C.attack.pixel_num = 5
_C.attack.beta = 1
_C.attack.neighbor_size = 5
_C.attack.save_fig_max_num = 0
_C.attack.save_fig_path = './output_img/{}/'

_C.DE = CN()
_C.DE.step = 10
_C.DE.popsize = 10

_C.auto = CN()
_C.auto.auto = False
_C.auto.csv_path = 'result/{}/auto_'
_C.auto.feature_list = ['tsfel', 'PCA']
# _C.auto.beta_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# _C.auto.pixel_num_list = [1, 3, 5, 10, 20]

_C.auto.beta_list = [0.1]
_C.auto.pixel_num_list = [1]


def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as infile:
        yaml_cfg = yaml.load(infile, Loader=yaml.FullLoader)
    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    print('merging config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)
    # config.freeze()


def get_config(cfg_file=None):
    """Return a clone of config or load from yaml file"""
    config = _C.clone()
    if cfg_file:
        _update_config_from_file(config, cfg_file)
    return config
