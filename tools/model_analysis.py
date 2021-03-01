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
from thirdparty.model_analysis_tool.ModelAnalyticalTool import ModelAnalyticalTool


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    args = parser.parse_args()
    cfg = Config.fromfile(args.config)

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    model.eval()
    model_float32 = copy.deepcopy(model)
    if hasattr(cfg, "quant_transformer"):
        model_transformer = build_mtransformer(cfg.quant_transformer)
        model = model_transformer(model)
    
    r""" 1. Float32 Model analysis """
    model_analysis_tool = ModelAnalyticalTool(model, is_quant = True, save_path = './model_analysis.html')
    model_analysis_tool.weight_dist_analysis(smaple_num = 15)
    
    infer_func = partial(inference_detector)
    model_analysis_tool.activation_dist_analysis(infer_func, smaple_num = 15, img = args.img)

    model_analysis_tool.down_html()