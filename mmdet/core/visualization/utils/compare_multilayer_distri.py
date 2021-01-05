from abc import ABCMeta
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import random
import mmcv
from mmcv.utils import TORCH_VERSION

if TORCH_VERSION < '1.1' or TORCH_VERSION == 'parrots':
        try:
            from tensorboardX import SummaryWriter
        except ImportError:
            raise ImportError('Please install tensorboardX to use '
                                'TensorboardLoggerHook.')
else:
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        raise ImportError(
            'Please run "pip install future tensorboard" to install '
            'the dependencies to use torch.utils.tensorboard '
            '(applicable to PyTorch 1.1 or higher)')


sns.set_style("whitegrid", {
    "ytick.major.size": 0.1,
    "ytick.minor.size": 0.05,
    'grid.linestyle': '--'
 })


class CompareMultiLayerDist(metaclass=ABCMeta):
    def __init__(self, 
                 figsize=(16,11), 
                 max_layer_num = 4, 
                 bins = 30,
                 **kwargs):
        super(CompareMultiLayerDist, self).__init__()
        assert(max_layer_num < 5)
        self.figsize = figsize
        self.max_layer_num = max_layer_num
        self.bins = bins
        self.colors = ['g', 'deeppink', 'dodgerblue', 'orange']
    
    def get_global_iter(self, runner):
            return runner.epoch * len(runner.data_loader) + runner.inner_iter
        
    def every_n_global_iter(self, runner, n):
        return (get_global_iter(runner) + 1 ) % n == 0 if n > 0 else False
    
    def get_epoch(self, runner):
        if runner.mode == 'train':
            epoch = runner.epoch + 1
        elif runner.mode == 'val':
            # normal val mode
            # runner.epoch += 1 has been done before val workflow
            epoch = runner.epoch
        else:
            raise ValueError(f"runner mode should be 'train' or 'val', "
                             f'but got {runner.mode}')
        return epoch
    
    def __call__(self, writer:SummaryWriter, runner: mmcv.runner.BaseRunner):

        fig, axes = plt.subplots(self.max_layer_num + 1, figsize = self.figsize)
        plt.switch_backend('agg')
        valid_weight_name = [name for name, para in runner.model.named_parameters() if name.endswith('weight') and 'conv' in name]
        valid_weight_name = random.sample(valid_weight_name, self.max_layer_num)
        
        idx = 0
        for name, para in runner.model.named_parameters():
            if name not in valid_weight_name:
                continue
            x = pd.Series(para.cpu().detach().numpy().flatten(), name = 'bin val')
            sns.distplot(x, 
                         bins = self.bins,
                         hist_kws = {'density' : True},
                         kde = True,
                         rug = False,
                         label = name,  
                         ax = axes[idx]
                         )
            sns.kdeplot(x,
                        shade = True,
                        color = self.colors[idx],
                        label = name, 
                        alpha = .7,
                        ax = axes[-1])
            axes[idx].set_ylabel('Density')
            axes[idx].legend()
            idx += 1
        axes[-1].legend()
        axes[-1].set_ylabel('Density')

        writer.add_figure('MultiLayer', fig,  self.get_epoch(runner))