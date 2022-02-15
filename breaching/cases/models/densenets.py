"""Overwrite pytorch DenseNet for CIFAR-10 and additional scripting options."""

import torch
import torchvision
from collections import OrderedDict

# densenet hints
from typing import Tuple
from torch import Tensor

from .utils import get_layer_functions


def densenet_depths_to_config(depth):
    """Lookup DenseNet types based on depth."""
    if depth == 121:
        growth_rate = 32
        block_config = (6, 12, 24, 16)
        num_init_features = 64
    elif depth == 161:
        growth_rate = 48
        block_config = (6, 12, 36, 24)
        num_init_features = 96
    elif depth == 169:
        growth_rate = 32
        block_config = (6, 12, 32, 32)
        num_init_features = 64
    elif depth == 201:
        growth_rate = 32
        block_config = (6, 12, 48, 32)
        num_init_features = 64
    return growth_rate, block_config, num_init_features


class DenseNet(torch.nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_.
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_.

    Torchvision Densenet modified to contain additional stems and scriptable norms/nonlins/convolutions
    """

    def __init__(
        self,
        growth_rate: int = 32,
        block_config: Tuple[int, int, int, int] = (6, 12, 24, 16),
        num_init_features: int = 64,
        bn_size: int = 4,
        drop_rate: float = 0,
        num_classes: int = 1000,
        channels: int = 3,
        memory_efficient: bool = False,
        norm: str = "BatchNorm2d",
        nonlin: str = "ReLU",
        stem: str = "CIFAR",
        convolution_type: str = "standard",
    ) -> None:

        super().__init__()

        self._conv_layer, self._norm_layer, self._nonlin_layer = get_layer_functions(convolution_type, norm, nonlin)

        # First convolution in different variations
        if stem in ["imagenet", "standard"]:
            self.features = torch.nn.Sequential(
                OrderedDict(
                    [
                        (
                            "conv0",
                            self._conv_layer(
                                channels, num_init_features, kernel_size=7, stride=2, padding=3, bias=False
                            ),
                        ),
                        ("norm0", self._norm_layer(num_init_features)),
                        ("relu0", self._nonlin_layer()),
                        ("pool0", torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
                    ]
                )
            )
        elif stem == "CIFAR":
            self.features = torch.nn.Sequential(
                OrderedDict(
                    [
                        (
                            "conv0",
                            self._conv_layer(
                                channels, num_init_features, kernel_size=3, stride=1, padding=1, bias=False
                            ),
                        ),
                    ]
                )
            )
        elif stem == "efficient":
            stem_width = num_init_features // 2
            self.features = torch.nn.Sequential(
                OrderedDict(
                    [
                        (
                            "conv0",
                            self._conv_layer(channels, stem_width, kernel_size=3, stride=2, padding=1, bias=False),
                        ),
                        ("norm0", self._norm_layer(stem_width)),
                        ("relu0", self._nonlin_layer()),
                        (
                            "conv1",
                            self._conv_layer(stem_width, stem_width, kernel_size=3, stride=1, padding=1, bias=False),
                        ),
                        ("norm1", self._norm_layer(stem_width)),
                        ("relu1", self._nonlin_layer()),
                        (
                            "conv2",
                            self._conv_layer(
                                stem_width, num_init_features, kernel_size=3, stride=1, padding=1, bias=False
                            ),
                        ),
                        ("norm2", self._norm_layer(num_init_features)),
                        ("relu2", self._nonlin_layer()),
                        ("pool0", torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
                    ]
                )
            )

        # Normal DenseNet from here: #

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
                norm=self._norm_layer,
                nonlin=self._nonlin_layer,
                convolution=self._conv_layer,
            )
            self.features.add_module("denseblock%d" % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(
                    num_input_features=num_features,
                    num_output_features=num_features // 2,
                    norm=self._norm_layer,
                    nonlin=self._nonlin_layer,
                    convolution=self._conv_layer,
                )
                self.features.add_module("transition%d" % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module("norm5", self._norm_layer(num_features))
        self.nonlin = self._nonlin_layer()
        # Linear layer
        self.classifier = torch.nn.Linear(num_features, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, self._conv_layer):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        features = self.features(x)
        out = self.nonlin(features)
        out = torch.nn.functional.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out


class _DenseLayer(torchvision.models.densenet._DenseLayer):
    def __init__(
        self,
        num_input_features: int,
        growth_rate: int,
        bn_size: int,
        drop_rate: float,
        memory_efficient: bool = False,
        norm=torch.nn.BatchNorm2d,
        nonlin=torch.nn.ReLU,
        convolution=torch.nn.Conv2d,
    ) -> None:
        torch.nn.Module.__init__(self)
        self.norm1: norm
        self.add_module("norm1", norm(num_input_features))
        self.relu1: nonlin
        self.add_module("relu1", nonlin())
        self.conv1: convolution
        self.add_module(
            "conv1", convolution(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)
        )
        self.norm2: norm
        self.add_module("norm2", norm(bn_size * growth_rate))
        self.relu2: nonlin
        self.add_module("relu2", nonlin())
        self.conv2: convolution
        self.add_module(
            "conv2", convolution(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)
        )
        self.drop_rate = float(drop_rate)
        self.memory_efficient = memory_efficient


class _DenseBlock(torch.nn.ModuleDict):
    _version = 2

    def __init__(
        self,
        num_layers: int,
        num_input_features: int,
        bn_size: int,
        growth_rate: int,
        drop_rate: float,
        memory_efficient: bool = False,
        norm=torch.nn.BatchNorm2d,
        nonlin=torch.nn.ReLU,
        convolution=torch.nn.Conv2d,
    ) -> None:
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
                norm=norm,
                nonlin=nonlin,
                convolution=convolution,
            )
            self.add_module("denselayer%d" % (i + 1), layer)

    def forward(self, init_features: Tensor) -> Tensor:
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)


class _Transition(torch.nn.Sequential):
    def __init__(
        self,
        num_input_features: int,
        num_output_features: int,
        norm=torch.nn.BatchNorm2d,
        nonlin=torch.nn.ReLU,
        convolution=torch.nn.Conv2d,
    ) -> None:
        super(_Transition, self).__init__()
        self.add_module("norm", norm(num_input_features))
        self.add_module("relu", nonlin(inplace=True))
        self.add_module(
            "conv", convolution(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False)
        )
        self.add_module("pool", torch.nn.AvgPool2d(kernel_size=2, stride=2))
