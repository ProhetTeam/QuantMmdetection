import os.path as osp
from mmcv.utils import TORCH_VERSION
from mmcv.runner.dist_utils import master_only
from mmcv.runner import HOOKS
from torch.utils.data import DataLoader
from mmcv.runner.hooks.logger import LoggerHook
import torch
import numpy as np
from .utils import CompareMultiLayerDist

@HOOKS.register_module()
class TensorboardLoggerHookV2(LoggerHook):
    def __init__(self,
                 log_dir=None,
                 interval=10,
                 ignore_last=True,
                 reset_flag=True,
                 by_epoch=True,
                 weight_vis_interval = 1,
                 by_iter=False,
                 cmp_multilayer_dist = False):
        super(TensorboardLoggerHookV2, self).__init__(interval, ignore_last,
                                                    reset_flag, by_epoch)
        self.log_dir = log_dir
        self.weight_vis_interval = weight_vis_interval
        self.by_iter = by_iter
        self.cmp_multilayer_dist = cmp_multilayer_dist
        
        self.drawers = []
        if self.cmp_multilayer_dist:
            self.drawers.append(CompareMultiLayerDist())

    @master_only
    def before_run(self, runner):
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

        if self.log_dir is None:
            self.log_dir = osp.join(runner.work_dir, 'tf_logs')
        self.writer = SummaryWriter(self.log_dir)

        ## TODO : Show Network Structure
        #self.writer.add_graph(runner.model, 
        #                      {'img': torch.rand((1,3,224,224)), 
        #                       'gt_label':torch.tensor([1], dtype = torch.int)})

    @master_only
    def log(self, runner):
        r"""
        This can be used by: after_train_iter, after_train_epoch
        This just for record lr, losss, top1, top5, and etc.
        """
        
        tags = self.get_loggable_tags(runner, allow_text=True)
        for tag, val in tags.items():
            if isinstance(val, str):
                self.writer.add_text(tag, val, self.get_iter(runner))
            else:
                self.writer.add_scalar(tag, val, self.get_iter(runner))

    @master_only
    def after_train_epoch(self, runner):
        r"""
        Fisrt Part:
            record logger info, such as: lr, losss, top1, top5, and etc.
        Second Part:
            staticstic weights distribution or gradient distribution. u can implement 
            anything u want. JUST DO IT!
        """
        if runner.log_buffer.ready:
            self.log(runner)
            if self.reset_flag:
                runner.log_buffer.clear_output()

            ## Default drawers
            if runner.mode == 'train' and self.every_n_epochs(runner, self.weight_vis_interval):
                for name, param in runner.model.named_parameters():
                    if 'bn' not in name:
                        if not param.numel() == 1: # Tensor
                            self.writer.add_histogram('model_by_epoch/' +  name, param, self.get_epoch(runner))
                            if hasattr(param, "grad") and param.grad is not None:
                                self.writer.add_histogram('model_by_epoch/' + name + "_grad", param.grad, self.get_epoch(runner))
                        else:
                            self.writer.add_scalar('model_by_epoch/' +  name, param, self.get_epoch(runner))
                            if hasattr(param, "grad") and param.grad is not None:
                                self.writer.add_scalar('model_by_epoch/' + name + "_grad", param.grad, self.get_epoch(runner))

                ## Customer Drawers list
                for draw in self.drawers:
                    draw(self.writer, runner)

    @master_only
    def after_run(self, runner):
        self.writer.close()
    
    def after_train_iter(self, runner):

        def get_global_iter(runner):
            return runner.epoch * len(runner.data_loader) + runner.inner_iter
        
        def every_n_global_iter(runner, n):
            return (get_global_iter(runner) + 1 ) % n == 0 if n > 0 else False

        super(TensorboardLoggerHookV2, self).after_train_iter(runner)
        ### 2. Draw Model Paras: such as weights, bias, and etc parameters distribution.
        if runner.mode == 'train' and \
            self.by_iter and \
            hasattr(self, 'writer') and \
            every_n_global_iter(runner, self.interval):
                for name, param in runner.model.named_parameters():
                    if 'bn' not in name:
                        if not param.numel() == 1: # Tensor
                            self.writer.add_histogram('model_by_iter/' + name, param, get_global_iter(runner) + 1)
                            if hasattr(param, "grad") and param.grad is not None:
                                self.writer.add_histogram('model_by_iter/' + name + "_grad", param.grad, get_global_iter(runner) + 1)
                        else:                   # Scalar
                            self.writer.add_scalar('model_by_iter/' + name, param.flatten(), get_global_iter(runner) + 1)
                            if hasattr(param, "grad") and param.grad is not None:
                                self.writer.add_scalar('model_by_iter/' + name  + "_grad", param.grad.flatten(), get_global_iter(runner) + 1)