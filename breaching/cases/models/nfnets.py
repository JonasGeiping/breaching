"""This is the NFNnets implementation from https://github.com/benjs/nfnets_pytorch.

with very minor modifications."""


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

nfnet_params = {
    "F0": {
        "width": [256, 512, 1536, 1536],
        "depth": [1, 2, 6, 3],
        "train_imsize": 192,
        "test_imsize": 256,
        "RA_level": "405",
        "drop_rate": 0.2,
    },
    "F1": {
        "width": [256, 512, 1536, 1536],
        "depth": [2, 4, 12, 6],
        "train_imsize": 224,
        "test_imsize": 320,
        "RA_level": "410",
        "drop_rate": 0.3,
    },
    "F2": {
        "width": [256, 512, 1536, 1536],
        "depth": [3, 6, 18, 9],
        "train_imsize": 256,
        "test_imsize": 352,
        "RA_level": "410",
        "drop_rate": 0.4,
    },
    "F3": {
        "width": [256, 512, 1536, 1536],
        "depth": [4, 8, 24, 12],
        "train_imsize": 320,
        "test_imsize": 416,
        "RA_level": "415",
        "drop_rate": 0.4,
    },
    "F4": {
        "width": [256, 512, 1536, 1536],
        "depth": [5, 10, 30, 15],
        "train_imsize": 384,
        "test_imsize": 512,
        "RA_level": "415",
        "drop_rate": 0.5,
    },
    "F5": {
        "width": [256, 512, 1536, 1536],
        "depth": [6, 12, 36, 18],
        "train_imsize": 416,
        "test_imsize": 544,
        "RA_level": "415",
        "drop_rate": 0.5,
    },
    "F6": {
        "width": [256, 512, 1536, 1536],
        "depth": [7, 14, 42, 21],
        "train_imsize": 448,
        "test_imsize": 576,
        "RA_level": "415",
        "drop_rate": 0.5,
    },
    "F7": {
        "width": [256, 512, 1536, 1536],
        "depth": [8, 16, 48, 24],
        "train_imsize": 480,
        "test_imsize": 608,
        "RA_level": "415",
        "drop_rate": 0.5,
    },
}

# These extra constant values ensure that the activations
# are variance preserving


class VPGELU(nn.Module):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.gelu(input) * 1.7015043497085571


class VPReLU(nn.Module):
    __constants__ = ["inplace"]
    inplace: bool

    def __init__(self, inplace: bool = False):
        super(VPReLU, self).__init__()
        self.inplace = inplace

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.relu(input, inplace=self.inplace) * 1.7139588594436646

    def extra_repr(self) -> str:
        inplace_str = "inplace=True" if self.inplace else ""
        return inplace_str


activations_dict = {"gelu": VPGELU(), "relu": VPReLU(inplace=True)}


class NFNet(nn.Module):
    def __init__(
        self,
        channels: int,
        classes: int,
        variant: str = "F0",
        stochdepth_rate: float = None,
        alpha: float = 0.2,
        se_ratio: float = 0.5,
        activation: str = "gelu",
        stem: str = "ImageNet",
        use_dropout=False,
    ):
        super(NFNet, self).__init__()

        if variant not in nfnet_params:
            raise RuntimeError(f"Variant {variant} does not exist and could not be loaded.")

        block_params = nfnet_params[variant]

        self.train_imsize = block_params["train_imsize"]
        self.test_imsize = block_params["test_imsize"]
        self.activation = activations_dict[activation]
        self.drop_rate = block_params["drop_rate"]
        self.use_dropout = use_dropout
        self.num_classes = classes

        if stem == "ImageNet":
            stride_stem = 2
        elif stem == "CIFAR":
            stride_stem = 1
        else:
            raise ValueError(f"Invalid stem type {stem} provided.")
        self.stem = Stem(channels, activation=activation, stride=stride_stem)

        num_blocks, index = sum(block_params["depth"]), 0

        blocks = []
        expected_std = 1.0
        in_channels = block_params["width"][0] // 2

        block_args = zip(
            block_params["width"],
            block_params["depth"],
            [0.5] * 4,  # bottleneck pattern
            [128] * 4,  # group pattern. Original groups [128] * 4
            [1, 2, 2, 2],  # stride pattern
        )

        for (block_width, stage_depth, expand_ratio, group_size, stride) in block_args:
            for block_index in range(stage_depth):
                beta = 1.0 / expected_std

                block_sd_rate = stochdepth_rate * index / num_blocks
                out_channels = block_width

                blocks.append(
                    NFBlock(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        stride=stride if block_index == 0 else 1,
                        alpha=alpha,
                        beta=beta,
                        se_ratio=se_ratio,
                        group_size=group_size,
                        stochdepth_rate=block_sd_rate,
                        activation=activation,
                        pad=1 if stem == "ImageNet" else 0,
                    )
                )

                in_channels = out_channels
                index += 1

                if block_index == 0:
                    expected_std = 1.0

                expected_std = (expected_std ** 2 + alpha ** 2) ** 0.5

        self.body = nn.Sequential(*blocks)

        final_conv_channels = 2 * in_channels
        self.final_conv = WSConv2D(in_channels=out_channels, out_channels=final_conv_channels, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d(1)

        if self.drop_rate > 0.0 and self.use_dropout:
            self.dropout = nn.Dropout(self.drop_rate)

        self.linear = nn.Linear(final_conv_channels, self.num_classes)
        nn.init.normal_(self.linear.weight, 0, 0.01)

    def forward(self, x):
        out = self.stem(x)
        out = self.body(out)
        out = self.activation(self.final_conv(out))
        pool = torch.mean(out, dim=(2, 3))

        if self.training and self.drop_rate > 0.0 and self.use_dropout:
            pool = self.dropout(pool)

        return self.linear(pool)


class Stem(nn.Module):
    def __init__(self, channels, activation: str = "gelu", stride: int = 2):
        super(Stem, self).__init__()

        self.activation = activations_dict[activation]
        self.conv0 = WSConv2D(in_channels=channels, out_channels=16, kernel_size=3, stride=stride)
        self.conv1 = WSConv2D(in_channels=16, out_channels=32, kernel_size=3, stride=1)
        self.conv2 = WSConv2D(in_channels=32, out_channels=64, kernel_size=3, stride=1)
        self.conv3 = WSConv2D(in_channels=64, out_channels=128, kernel_size=3, stride=stride)

    def forward(self, x):
        out = self.activation(self.conv0(x))
        out = self.activation(self.conv1(out))
        out = self.activation(self.conv2(out))
        out = self.conv3(out)
        return out


class NFBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expansion: float = 0.5,
        se_ratio: float = 0.5,
        stride: int = 1,
        beta: float = 1.0,
        alpha: float = 0.2,
        group_size: int = 1,
        stochdepth_rate: float = None,
        activation: str = "gelu",
        pad: int = 1,
    ):

        super(NFBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expansion = expansion
        self.se_ratio = se_ratio
        self.activation = activations_dict[activation]
        self.beta, self.alpha = beta, alpha
        self.group_size = group_size

        width = int(self.out_channels * expansion)
        self.groups = width // group_size
        self.width = group_size * self.groups
        self.stride = stride

        self.conv0 = WSConv2D(in_channels=self.in_channels, out_channels=self.width, kernel_size=1)
        self.conv1 = WSConv2D(
            in_channels=self.width, out_channels=self.width, kernel_size=3, stride=stride, padding=1, groups=self.groups
        )
        self.conv1b = WSConv2D(
            in_channels=self.width, out_channels=self.width, kernel_size=3, stride=1, padding=1, groups=self.groups
        )
        self.conv2 = WSConv2D(in_channels=self.width, out_channels=self.out_channels, kernel_size=1)

        self.use_projection = self.stride > 1 or self.in_channels != self.out_channels
        if self.use_projection:
            if stride > 1:
                self.shortcut_avg_pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=pad)
            else:
                self.shortcut_avg_pool = nn.Identity()
            self.conv_shortcut = WSConv2D(self.in_channels, self.out_channels, kernel_size=1)
        else:
            self.shortcut_avg_pool = nn.Identity()
            self.conv_shortcut = nn.Identity()

        self.squeeze_excite = SqueezeExcite(
            self.out_channels, self.out_channels, se_ratio=self.se_ratio, activation=activation
        )
        self.skip_gain = nn.Parameter(torch.zeros(()))

        self.use_stochdepth = stochdepth_rate is not None and stochdepth_rate > 0.0 and stochdepth_rate < 1.0
        if self.use_stochdepth:
            self.stoch_depth = StochDepth(stochdepth_rate)
        else:
            self.stoch_depth = nn.Identity()

    def forward(self, x):
        out = self.activation(x) * self.beta

        if self.stride > 1:
            shortcut = self.shortcut_avg_pool(out)
            shortcut = self.conv_shortcut(shortcut)
        elif self.use_projection:
            shortcut = self.conv_shortcut(out)
        else:
            shortcut = x

        out = self.activation(self.conv0(out))
        out = self.activation(self.conv1(out))
        out = self.activation(self.conv1b(out))
        out = self.conv2(out)
        out = (self.squeeze_excite(out) * 2) * out

        if self.use_stochdepth:
            out = self.stoch_depth(out)

        return out * self.alpha * self.skip_gain + shortcut


# Implementation mostly from https://arxiv.org/abs/2101.08692
# Implemented changes from https://arxiv.org/abs/2102.06171 and
#  https://github.com/deepmind/deepmind-research/tree/master/nfnets


class WSConv2D(nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
    ):

        super(WSConv2D, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode
        )

        nn.init.xavier_normal_(self.weight)
        self.gain = nn.Parameter(torch.ones(self.out_channels, 1, 1, 1))
        self.register_buffer("eps", torch.tensor(1e-4, requires_grad=False), persistent=False)
        self.register_buffer(
            "fan_in",
            torch.tensor(np.prod(self.weight.shape[1:]), requires_grad=False).type_as(self.weight),
            persistent=False,
        )

    def standardized_weights(self):
        # Original code: HWCN
        mean = torch.mean(self.weight, dim=(1, 2, 3), keepdim=True)
        var = torch.var(self.weight, dim=(1, 2, 3), keepdim=True)
        scale = torch.rsqrt(torch.maximum(var * self.fan_in, self.eps))
        return (self.weight - mean) * scale * self.gain

    def forward(self, x):
        return F.conv2d(
            input=x,
            weight=self.standardized_weights(),
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )


class SqueezeExcite(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, se_ratio: float = 0.5, activation: str = "gelu"):
        super(SqueezeExcite, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.se_ratio = se_ratio

        self.hidden_channels = max(1, int(self.in_channels * self.se_ratio))

        self.activation = activations_dict[activation]
        self.linear = nn.Linear(self.in_channels, self.hidden_channels)
        self.linear_1 = nn.Linear(self.hidden_channels, self.out_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = torch.mean(x, (2, 3))
        out = self.linear_1(self.activation(self.linear(out)))
        out = self.sigmoid(out)

        b, c, _, _ = x.size()
        return out.view(b, c, 1, 1).expand_as(x)


class StochDepth(nn.Module):
    def __init__(self, stochdepth_rate: float):
        super(StochDepth, self).__init__()

        self.drop_rate = stochdepth_rate

    def forward(self, x):
        if not self.training:
            return x

        batch_size = x.shape[0]
        rand_tensor = torch.rand(batch_size, 1, 1, 1).type_as(x).to(x.device)
        keep_prob = 1 - self.drop_rate
        binary_tensor = torch.floor(rand_tensor + keep_prob)

        return x * binary_tensor
