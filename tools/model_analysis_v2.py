import random
import torch.backends.cudnn as cudnn
import torch
import torch.nn as nn
import copy
from argparse import ArgumentParser
try:
    from thirdparty.mtransformer.DSQ.DSQConv import DSQConv
    from thirdparty.mtransformer.APOT.APOTLayers import APOTQuantConv2d
    from thirdparty.mtransformer.LSQ.LSQConv import LSQConv2d
except ImportError:
    raise ImportError("Please import ALL Qunat layer!!!!")

from collections import OrderedDict
import mmcv
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist
from mmcv.utils import get_git_hash
import numpy as np

from mmdet import __version__
from mmdet.apis import set_random_seed, train_detector
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.utils import collect_env, get_root_logger
from thirdparty.mtransformer import build_mtransformer

from mmcv.runner import (HOOKS, DistSamplerSeedHook, EpochBasedRunner,
                         Fp16OptimizerHook, OptimizerHook, build_optimizer)
from mmcv.utils import build_from_cfg
from mmdet.utils import get_root_logger
from mmdet.apis import inference_detector, init_detector, show_result_pyplot
from functools import partial
from thirdparty.model_analysis_tool.ModelAnalyticalToolV2 import QModelAnalysis


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('--config-float', help='Float config file')
    parser.add_argument('--config-int', help='Int config file')
    parser.add_argument('--checkpoint-float', help='Float checkpoint file')
    parser.add_argument('--checkpoint-int', help = 'Int checkpoint file')
    parser.add_argument('--save-path', type = str, default= "./model_analysis.html", help = "html save path")
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model_float = init_detector(args.config_float, args.checkpoint_float, device=args.device)
    model_int = init_detector(args.config_int, args.checkpoint_int, device=args.device) 
    model_float.eval()
    model_int.eval()
    
    r""" 1. Float32 Model analysis """
    model_analysis_tool = QModelAnalysis(model_float, model_int, 
                                         smaple_num = 15, 
                                         max_data_length = 2e4, 
                                         bin_size = 0.01, 
                                         save_path = args.save_path,
                                         use_torch_plot = False)
    model_analysis_tool(inference_detector, img = args.img)
    model_analysis_tool.down_html()
