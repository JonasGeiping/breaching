"""Utilities for model scripting."""

import torch
from .nfnets import WSConv2D
from functools import partial


def get_layer_functions(convolution_type, norm, nonlin):

    if convolution_type.lower() in ["standard", "default", "zeros"]:
        conv_layer = torch.nn.Conv2d
    elif convolution_type.lower() in ["circular", "reflect", "replicate"]:
        conv_layer = partial(torch.nn.Conv2d, padding_mode=convolution_type.lower())
    elif convolution_type.lower() == "standardized":
        conv_layer = WSConv2D
    else:
        raise ValueError(f"Invalid convolution type {convolution_type} provided.")

    try:
        norm_layer = getattr(torch.nn, norm)
    except AttributeError:
        if norm.lower() == "sequentialghostnorm":
            norm_layer = SequentialGhostNorm
        elif norm.lower() == "groupnorm1":
            norm_layer = lambda C: torch.nn.GroupNorm(num_groups=1, num_channels=C, affine=True)  # noqa
        elif norm.lower() == "groupnorm8":
            norm_layer = lambda C: torch.nn.GroupNorm(num_groups=min(8, C), num_channels=C, affine=True)  # noqa
        elif norm.lower() == "groupnorm32":
            norm_layer = lambda C: torch.nn.GroupNorm(num_groups=min(32, C), num_channels=C, affine=True)  # noqa
        elif norm.lower() == "groupnorm4th":
            norm_layer = lambda C: torch.nn.GroupNorm(num_groups=C // 4, num_channels=C, affine=True)  # noqa
        elif norm.lower() in ["skipinit", "None", "Identity"]:
            norm_layer = torch.nn.Identity
        else:
            raise ValueError("Invalid norm layer found.")

    if nonlin.lower() == "relu" and norm.lower() != "skipinit":
        nonlin_layer = partial(torch.nn.ReLU, inplace=True)
    else:
        nonlin_layer = getattr(torch.nn, nonlin)

    return conv_layer, norm_layer, nonlin_layer
