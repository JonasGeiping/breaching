"""Helper code to instantiate various models."""

import torch
import torchvision

from .resnets import ResNet, resnet_depths_to_config
from .densenets import DenseNet, densenet_depths_to_config
from .nfnets import NFNet
from .vgg import VGG

def construct_model(cfg_model, cfg_data, pretrained=False):
    """Construct the neural net that is used."""
    channels = cfg_data.shape[0]
    classes = cfg_data.classes

    if cfg_data.name == 'ImageNet':
        try:
            model = getattr(torchvision.models, cfg_model.lower())(pretrained=pretrained)
        except AttributeError:
            if 'nfnet' in cfg_model.name:
                model = NFNet(channels, classes, variant='F0', stochdepth_rate=0.25, alpha=0.2, se_ratio=0.5,
                              activation='ReLU', stem='ImageNet', use_dropout=True)
            elif 'ResNetWSL' in cfg_model.name:
                model = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x48d_wsl')
            elif 'ResNet50SWSL' in cfg_model.name:
                model = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnet50_swsl')
            elif 'ResNet50SSL' in cfg_model.name:
                model = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnet50_ssl')
            else:
                raise ValueError(f'Could not find ImageNet model {cfg_model} in torchvision.models or custom models.')
    else:
        if 'resnet' in cfg_model.lower():
            block, layers = resnet_depths_to_config(int("".join(filter(str.isdigit, cfg_model))))
            model = ResNet(block, layers, channels, classes, stem='CIFAR', convolution_type='Standard',
                           nonlin='ReLU', norm='BatchNorm2d',
                           downsample='C', width_per_group=64,
                           zero_init_residual=False)
        elif 'densenet' in cfg_model.lower():
            growth_rate, block_config, num_init_features = densenet_depths_to_config(int("".join(filter(str.isdigit, cfg_model))))
            model = DenseNet(growth_rate=growth_rate,
                             block_config=block_config,
                             num_init_features=num_init_features,
                             bn_size=4,
                             drop_rate=0,
                             channels=channels,
                             num_classes=classes,
                             memory_efficient=False,
                             norm='BatchNorm2d',
                             nonlin='ReLU',
                             stem='CIFAR',
                             convolution_type='Standard')
        elif 'vgg' in cfg_model.name.lower():
            model = VGG(cfg_model, in_channels=channels, num_classes=classes, norm='BatchNorm2d',
                        nonlin='ReLU', head='CIFAR', convolution_type='Standard',
                        drop_rate=0, classical_weight_init=True)
        elif 'linear' in cfg_model.name:
            input_dim = cfg_data.shape[0] * cfg_data.shape[1] * cfg_data.shape[2]
            model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(input_dim, classes))
        elif 'nfnet' in cfg_model.name:
            model = NFNet(channels, classes, variant='F0', stochdepth_rate=0.25, alpha=0.2, se_ratio=0.5,
                          activation='ReLU', stem='CIFAR', use_dropout=True)
        else:
            raise ValueError('Model could not be found.')

    return model
