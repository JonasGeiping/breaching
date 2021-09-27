"""Helper functions to modify models to have multiple gradient paths."""
import functools
import torch


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))


def _set_layer(weight, num_paths):
    out_planes = weight.shape[0]
    in_planes = weight.shape[1]
    ratio = out_planes / in_planes
    per_path = int(out_planes / num_paths / ratio)
    with torch.no_grad():
        for i in range(out_planes):
            temp_weight = torch.zeros_like(weight.data[i])
            block = (i % in_planes) // per_path
            start = block * per_path
            temp_weight[start:start + per_path] = weight.data[i % per_path][0:per_path]
            weight.data[i] = temp_weight
        if ratio > 1:
            weight.data = _zipper(weight.data, ratio)
    return ratio


def _zipper(weight, ratio):
    num_per_group = weight.shape[0] // ratio
    new_weight = torch.zeros_like(weight)
    for i in range(int(num_per_group)):
        for zipper_idx in range(int(ratio)):
            new_weight[int(i * ratio + zipper_idx)] = weight[int(zipper_idx * num_per_group + i)]
    return new_weight


def _set_bias(bias, ratio, num_paths):
    per_path = int(bias.data.shape[0] / num_paths / ratio)
    with torch.no_grad():
        for i in range(int(bias.data.shape[0] / per_path)):
            for j in range(per_path):
                bias.data[i * per_path + j] = bias.data[j]


def _eliminate_shortcut_weight(shortcut):
    with torch.no_grad():
        shortcut.data = torch.zeros_like(shortcut)


def _make_average_layer(weight, num_paths):
    with torch.no_grad():
        weight.data = 1 / weight.data.shape[-1] * torch.ones_like(weight.data)
    new_weight = torch.zeros_like(weight.data)
    per_block = weight.data.shape[-1] // num_paths
    for i in range(num_paths):
        new_weight[i][i * per_block:i * per_block + per_block] = 1 / per_block * \
            torch.ones_like(new_weight[i][i * per_block:i * per_block + per_block])


def _make_linear_biases(bias, bins):
    bins = bins[0]
    new_biases = torch.zeros_like(bias.data)
    for i in range(min(len(bins), len(new_biases))):
        new_biases[i] = bins[i]
    with torch.no_grad():
        bias.data = new_biases
