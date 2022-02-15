"""This file is a modification of the torchvision resnet implementation.

Modified to change stuff like nonlinearity, block structure, number of blocks, depth ...

The original file was retrieved from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
for torchvision version 0.8.2
"""
import torch
from .utils import get_layer_functions


def resnet_depths_to_config(depth):
    if depth == 20:
        block = BasicBlock
        layers = [3, 3, 3]
    elif depth == 32:
        block = BasicBlock
        layers = [5, 5, 5]
    elif depth == 56:
        block = BasicBlock
        layers = [9, 9, 9]
    elif depth == 110:
        block = BasicBlock
        layers = [18, 18, 18]
    elif depth == 18:
        block = BasicBlock
        layers = [2, 2, 2, 2]
    elif depth == 34:
        block = BasicBlock
        layers = [3, 4, 6, 3]
    elif depth == 50:
        block = Bottleneck
        layers = [3, 4, 6, 3]
    elif depth == 101:
        block = Bottleneck
        layers = [3, 4, 23, 3]
    elif depth == 152:
        block = Bottleneck
        layers = [3, 8, 36, 3]
    else:
        raise ValueError(f"Invalid depth {depth} given.")
    return block, layers


class ResNet(torch.nn.Module):
    def __init__(
        self,
        block,
        layers,
        channels,
        classes,
        zero_init_residual=False,
        strides=[1, 2, 2, 2],
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=[False, False, False, False],
        norm="BatchNorm2d",
        nonlin="ReLU",
        stem="CIFAR",
        downsample="B",
        convolution_type="Standard",
    ):
        super(ResNet, self).__init__()
        self._conv_layer, self._norm_layer, self._nonlin_layer = get_layer_functions(convolution_type, norm, nonlin)
        self.use_bias = False
        self.inplanes = width_per_group if block is BasicBlock else 64
        self.dilation = 1
        if len(replace_stride_with_dilation) != 4:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 4-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group if block is Bottleneck else 64

        if stem == "CIFAR":
            conv1 = self._conv_layer(
                channels, self.inplanes, kernel_size=3, stride=1, padding=1, groups=1, bias=self.use_bias, dilation=1
            )
            bn1 = self._norm_layer(self.inplanes)
            nonlin = self._nonlin_layer()
            self.stem = torch.nn.Sequential(conv1, bn1, nonlin)
        elif stem == "standard":
            conv1 = self._conv_layer(channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=self.use_bias)
            bn1 = self._norm_layer(self.inplanes)
            nonlin = self._nonlin_layer()
            maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.stem = torch.nn.Sequential(conv1, bn1, nonlin, maxpool)
        elif stem == "efficient":
            stem_width = self.inplanes // 2
            conv1 = self._conv_layer(
                channels, stem_width, kernel_size=3, stride=2, padding=1, groups=1, bias=self.use_bias, dilation=1
            )
            bn1 = self._norm_layer(stem_width)
            conv2 = self._conv_layer(
                stem_width, stem_width, kernel_size=3, stride=1, padding=1, groups=1, bias=self.use_bias, dilation=1
            )
            bn2 = self._norm_layer(stem_width)
            conv3 = self._conv_layer(
                stem_width, self.inplanes, kernel_size=3, stride=1, padding=1, groups=1, bias=self.use_bias, dilation=1
            )
            bn3 = self._norm_layer(self.inplanes)

            nonlin = self._nonlin_layer()
            maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.stem = torch.nn.Sequential(conv1, bn1, nonlin, conv2, bn2, nonlin, conv3, bn3, nonlin, maxpool)
        else:
            raise ValueError(f"Invalid stem designation {stem}.")

        layer_list = []
        width = self.inplanes
        for idx, layer in enumerate(layers):
            layer_list.append(
                self._make_layer(
                    block,
                    width,
                    layer,
                    stride=strides[idx],
                    dilate=replace_stride_with_dilation[idx],
                    downsample=downsample,
                )
            )
            width *= 2
        self.layers = torch.nn.Sequential(*layer_list)

        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.linear = torch.nn.Linear(self.inplanes, classes)

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (torch.nn.BatchNorm2d, torch.nn.GroupNorm)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    if hasattr(m.bn3, "weight"):
                        torch.nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    if hasattr(m.bn2, "weight"):
                        torch.nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, downsample="B"):
        conv_layer = self._conv_layer
        norm_layer = self._norm_layer
        nonlin_layer = self._nonlin_layer
        downsample_op = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            if downsample == "A":
                downsample_op = torch.nn.Sequential(
                    conv_layer(
                        self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=self.use_bias
                    ),
                )
            elif downsample == "B":
                downsample_op = torch.nn.Sequential(
                    conv_layer(
                        self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=self.use_bias
                    ),
                    norm_layer(planes * block.expansion),
                )
            elif downsample == "C":
                downsample_op = torch.nn.Sequential(
                    torch.nn.AvgPool2d(kernel_size=stride, stride=stride),
                    conv_layer(self.inplanes, planes * block.expansion, kernel_size=1, stride=1, bias=self.use_bias),
                    norm_layer(planes * block.expansion),
                )
            elif downsample == "preact-B":
                downsample_op = torch.nn.Sequential(
                    nonlin_layer(),
                    conv_layer(
                        self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=self.use_bias
                    ),
                )
            elif downsample == "preact-C":
                downsample_op = torch.nn.Sequential(
                    nonlin_layer(),
                    torch.nn.AvgPool2d(kernel_size=stride, stride=stride),
                    conv_layer(self.inplanes, planes * block.expansion, kernel_size=1, stride=1, bias=self.use_bias),
                )
            else:
                raise ValueError("Invalid downsample block specification.")

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample_op,
                self.groups,
                self.base_width,
                previous_dilation,
                conv=conv_layer,
                nonlin=nonlin_layer,
                norm_layer=norm_layer,
                bias=self.use_bias,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                    nonlin=nonlin_layer,
                    bias=self.use_bias,
                )
            )

        return torch.nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.stem(x)

        x = self.layers(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


class BasicBlock(torch.nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        conv=torch.nn.Conv2d,
        nonlin=torch.nn.ReLU,
        norm_layer=torch.nn.BatchNorm2d,
        bias=False,
    ):
        super().__init__()
        # if groups != 1 or base_width != 64:
        #     raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv(inplanes, planes, kernel_size=3, stride=stride, padding=1, groups=1, bias=bias, dilation=1)
        self.bn1 = norm_layer(planes)
        self.nonlin = nonlin()
        self.conv2 = conv(planes, planes, kernel_size=3, stride=1, padding=1, groups=1, bias=bias, dilation=1)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.nonlin(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.nonlin(out)

        return out


class Bottleneck(torch.nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        conv=torch.nn.Conv2d,
        nonlin=torch.nn.ReLU,
        norm_layer=torch.nn.BatchNorm2d,
        bias=False,
    ):
        super(Bottleneck, self).__init__()
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv(inplanes, width, kernel_size=1, stride=1, bias=bias)
        self.bn1 = norm_layer(width)
        self.conv2 = conv(
            width, width, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=bias, dilation=dilation
        )
        self.bn2 = norm_layer(width)
        self.conv3 = conv(width, planes * self.expansion, kernel_size=1, stride=1, bias=bias)
        self.bn3 = norm_layer(planes * self.expansion)
        self.nonlin = nonlin()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.nonlin(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.nonlin(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.nonlin(out)

        return out
