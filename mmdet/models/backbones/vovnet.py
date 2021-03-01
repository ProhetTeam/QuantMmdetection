from collections import OrderedDict
from typing import Optional, Sequence

import torch
import torch.nn as nn

from ..builder import BACKBONES
from mmdet.models.backbones.base_backbone import BaseBackbone, filter_by_out_idices

__all__ = ['VoVNet27Slim', 'VoVNet39', 'VoVNet57', 'VoVNet18Tiny']

model_urls = {
    'vovnet39': 'https://dl.dropbox.com/s/1lnzsgnixd8gjra/vovnet39_torchvision.pth?dl=1',
    'vovnet57': 'https://dl.dropbox.com/s/6bfu9gstbwfw31m/vovnet57_torchvision.pth?dl=1'
}

def load_pretrained_weights(model, model_url, load_fc=True):
    """ Loads pretrained weights, and downloads if loading for the first time. """
    from torch.hub import load_state_dict_from_url
    state_dict = load_state_dict_from_url(model_url, map_location=torch.device('cpu'))

    # strip prefix of state_dict
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    if load_fc:
        model.load_state_dict(state_dict)
    else:
        state_dict.pop('classifier.weight')
        state_dict.pop('classifier.bias')
        model.load_state_dict(state_dict, strict=False)
    print('Loaded pretrained weights for {}:{}'.format(model.arch, model_url))

def conv3x3(in_channels, out_channels, module_name, postfix,
            stride=1, groups=1, kernel_size=3, padding=1):
    """3x3 convolution with padding"""
    return [
        ('{}_{}/conv'.format(module_name, postfix),
         nn.Conv2d(in_channels, out_channels,
                   kernel_size=kernel_size,
                   stride=stride,
                   padding=padding,
                   groups=groups,
                   bias=False)),
        ('{}_{}/norm'.format(module_name, postfix),
         nn.BatchNorm2d(out_channels)),
        ('{}_{}/relu'.format(module_name, postfix),
         nn.ReLU(inplace=True)),
    ]


def conv1x1(in_channels, out_channels, module_name, postfix,
            stride=1, groups=1, kernel_size=1, padding=0):
    """1x1 convolution"""
    return [
        ('{}_{}/conv'.format(module_name, postfix),
         nn.Conv2d(in_channels, out_channels,
                   kernel_size=kernel_size,
                   stride=stride,
                   padding=padding,
                   groups=groups,
                   bias=False)),
        ('{}_{}/norm'.format(module_name, postfix),
         nn.BatchNorm2d(out_channels)),
        ('{}_{}/relu'.format(module_name, postfix),
         nn.ReLU(inplace=True)),
    ]


class _OSA_module(nn.Module):
    def __init__(self,
                 in_ch,
                 stage_ch,
                 concat_ch,
                 layer_per_block,
                 module_name,
                 identity=False):
        super(_OSA_module, self).__init__()

        self.identity = identity
        self.layers = nn.ModuleList()
        in_channel = in_ch
        for i in range(layer_per_block):
            self.layers.append(nn.Sequential(
                OrderedDict(conv3x3(in_channel, stage_ch, module_name, i))))
            in_channel = stage_ch

        # feature aggregation
        in_channel = in_ch + layer_per_block * stage_ch
        self.concat = nn.Sequential(
            OrderedDict(conv1x1(in_channel, concat_ch, module_name, 'concat')))

    def forward(self, x):
        identity_feat = x
        output = []
        output.append(x)
        for layer in self.layers:
            x = layer(x)
            output.append(x)

        x = torch.cat(output, dim=1)
        xt = self.concat(x)

        if self.identity:
            xt = xt + identity_feat

        return xt


class _OSA_stage(nn.Sequential):
    def __init__(self,
                 in_ch,
                 stage_ch,
                 concat_ch,
                 block_per_stage,
                 layer_per_block,
                 stage_num):
        super(_OSA_stage, self).__init__()

        if not stage_num == 2:
            self.add_module('Pooling',
                            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True))

        module_name = f'OSA{stage_num}_1'
        self.add_module(module_name,
                        _OSA_module(in_ch,
                                    stage_ch,
                                    concat_ch,
                                    layer_per_block,
                                    module_name))
        for i in range(block_per_stage - 1):
            module_name = f'OSA{stage_num}_{i + 2}'
            self.add_module(module_name,
                            _OSA_module(concat_ch,
                                        stage_ch,
                                        concat_ch,
                                        layer_per_block,
                                        module_name,
                                        identity=True))


class VoVNetBase(BaseBackbone):
    def __init__(self,
                 arch,
                 config_stage_ch,
                 config_concat_ch,
                 block_per_stage,
                 layer_per_block,
                 inplanes=64,
                 pretrain: bool = False,
                 progress: bool = True,
                 out_indices: Optional[Sequence[int]] = (1, 2, 3, 4)):
        super(VoVNetBase, self).__init__(out_indices)
        
        self.arch = arch

        # Stem module
        stem = conv3x3(3, inplanes, 'stem', '1', 2)
        stem += conv3x3(inplanes, inplanes, 'stem', '2', 1)
        stem += conv3x3(inplanes, 2 * inplanes, 'stem', '3', 2)
        self.add_module('stem', nn.Sequential(OrderedDict(stem)))

        stem_out_ch = [2 * inplanes]
        in_ch_list = stem_out_ch + config_concat_ch[:-1]
        self.stage_names = []
        for i in range(4):  # num_stages
            name = f'stage{i + 2}'
            self.stage_names.append(name)
            self.add_module(name,
                            _OSA_stage(in_ch_list[i],
                                       config_stage_ch[i],
                                       config_concat_ch[i],
                                       block_per_stage[i],
                                       layer_per_block,
                                       i + 2))

    @filter_by_out_idices
    def forward(self, x):
        skips = []
        x = self.stem(x)
        skips.append(x)
        for name in self.stage_names:
            x = getattr(self, name)(x)
            skips.append(x)
        return skips

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            if pretrained in model_urls:
                pretrained = model_urls[pretrained]
            load_pretrained_weights(self, pretrained, False)
        elif pretrained is None:
            self._initialize_weights()
        else:
            raise TypeError('pretrained must be a str or None')

@BACKBONES.register_module
class VoVNet57(VoVNetBase):
    r"""Constructs a VoVNet-57 model as described in
    `"An Energy and GPU-Computation Efficient Backbone Networks"
    <https://arxiv.org/abs/1904.09730>`_.
    Args:
        pretrain (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """

    def __init__(self, pretrain: bool = False, progress: bool = True,
                 out_indices: Optional[Sequence[int]] = (1, 2, 3, 4)):
        super().__init__(
            arch='vovnet57',
            config_stage_ch=[128, 160, 192, 224],
            config_concat_ch=[256, 512, 768, 1024],
            block_per_stage=[1, 1, 4, 3],
            layer_per_block=5,
            pretrain=pretrain,
            progress=progress,
            out_indices=out_indices)


@BACKBONES.register_module
class VoVNet39(VoVNetBase):
    r"""Constructs a VoVNet-39 model as described in
    `"An Energy and GPU-Computation Efficient Backbone Networks"
    <https://arxiv.org/abs/1904.09730>`_.
    Args:
        pretrain (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """

    def __init__(self, pretrain: bool = False, progress: bool = True,
                 out_indices: Optional[Sequence[int]] = (1, 2, 3, 4), 
                 norm_eval = False,
                 fix_backbone = False):
        super().__init__(
            arch='vovnet39',
            config_stage_ch=[128, 160, 192, 224],
            config_concat_ch=[256, 512, 768, 1024],
            block_per_stage=[1, 1, 2, 2],
            layer_per_block=5,
            pretrain=pretrain,
            progress=progress,
            out_indices=out_indices)
        self.norm_eval = norm_eval
        self.fix_backbone = fix_backbone
        if self.fix_backbone:
            for p in self.parameters():
                p.requires_grad = False
        
    def train(self, mode=True):
        super(VoVNet39, self).train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()


@BACKBONES.register_module
class VoVNet27Slim(VoVNetBase):
    r"""Constructs a VoVNet-39 model as described in
    `"An Energy and GPU-Computation Efficient Backbone Networks"
    <https://arxiv.org/abs/1904.09730>`_.
    Args:
        pretrain (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """

    def __init__(self, pretrain: bool = False, progress: bool = True,
                 out_indices: Optional[Sequence[int]] = (1, 2, 3, 4)):
        super().__init__(
            arch='vovnet27_slim',
            config_stage_ch=[64, 80, 96, 112],
            config_concat_ch=[128, 256, 384, 512],
            block_per_stage=[1, 1, 1, 1],
            layer_per_block=5,
            pretrain=pretrain,
            progress=progress,
            out_indices=out_indices)

@BACKBONES.register_module
class VoVNet18Tiny(VoVNetBase):
    r"""Constructs a VoVNet-39 model as described in
    `"An Energy and GPU-Computation Efficient Backbone Networks"
    <https://arxiv.org/abs/1904.09730>`_.
    Args:
        pretrain (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """

    def __init__(self, pretrain: bool = False, progress: bool = True,
                 out_indices: Optional[Sequence[int]] = (1, 2, 3, 4)):
        super().__init__(
            arch='vovnet18_tiny',
            inplanes=16,
            config_stage_ch=[16, 32, 48, 64],
            config_concat_ch=[32, 64, 96, 128],
            block_per_stage=[1, 1, 1, 1],
            layer_per_block=3,
            pretrain=pretrain,
            progress=progress,
            out_indices=out_indices)