from pytorch_cifar import models as cifar_models
from scipy.stats import norm
import functools
import torch



def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)

def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))

def _set_layer(weight, num_paths):
    out_planes = weight.shape[0]
    in_planes = weight.shape[1]
    ratio = out_planes/in_planes
    per_path = int(out_planes/num_paths/ratio)
    with torch.no_grad():
        for i in range(out_planes):
            temp_weight = torch.zeros_like(weight.data[i])
            block = (i % in_planes) // per_path
            start = block * per_path
            temp_weight[start:start+per_path] = weight.data[i % per_path][0:per_path]
            weight.data[i] = temp_weight
        if ratio > 1:
            weight.data = _zipper(weight.data, ratio)
    return ratio

def _zipper(weight, ratio):
    num_per_group = weight.shape[0] // ratio
    new_weight = torch.zeros_like(weight)
    for i in range(int(num_per_group)):
        for zipper_idx in range(int(ratio)):
            new_weight[int(i*ratio + zipper_idx)] = weight[int(zipper_idx * num_per_group + i)]
    return new_weight

def _set_bias(bias, ratio, num_paths):
    per_path = int(bias.data.shape[0]/num_paths/ratio)
    with torch.no_grad():
        for i in range(int(bias.data.shape[0]/per_path)):
            for j in range(per_path):
                bias.data[i*per_path + j] = bias.data[j]

def _eliminate_shortcut_weight(shortcut):
    with torch.no_grad():
        shortcut.data = torch.zeros_like(shortcut)


def _get_bins(num_bins, mu, sigma):
    order_stats = [_get_order_stats(r + 1, num_bins, mu, sigma) for r in range(num_bins)]
    diffs = [order_stats[i] - order_stats[i + 1] for i in range(len(order_stats) - 1)]
    bin_val = -sum(diffs) / len(diffs)
    left_bins = [-i * bin_val + mu for i in range(num_bins // 2 + 1)]
    right_bins = [i * bin_val + mu for i in range(1, num_bins // 2)]
    left_bins.reverse()
    bins = left_bins + right_bins
    return bins, bin_val

def _get_order_stats(r, n, mu, sigma):
    """
    r Order statistics can be computed as follows:
    E(r:n) = \mu + \Phi^{-1}\left( \frac{r-a}{n-2a+1} \sigma \right)
    where a = 0.375
    """
    alpha = 0.375
    return mu + norm.ppf((r - alpha) / (n - 2 * alpha + 1)) * sigma

def _make_average_layer(weight, num_paths):
    with torch.no_grad():
        weight.data = 1 / weight.data.shape[-1] * torch.ones_like(weight.data)
    new_weight = torch.zeros_like(weight.data)
    per_block = weight.data.shape[-1] // num_paths
    for i in range(num_paths):
        new_weight[i][i*per_block:i*per_block + per_block] = 1/per_block * torch.ones_like(new_weight[i][i*per_block:i*per_block + per_block])


def _make_linear_biases(bias, bins):
    bins = bins[0]
    new_biases = torch.zeros_like(bias.data)
    for i in range(min(len(bins), len(new_biases))):
        new_biases[i] = bins[i]
    with torch.no_grad():
        bias.data = new_biases



def path_parameters(model, num_paths=8):
    """
    Setting the paths in the network (feature extractor)
    """
    ratio = 1
    num_paths = 8
    for (k, v) in model.named_parameters():
        #if 'layer' in k or 'linear' in k: # skip the first conv layer? 
        if 'layer' in k or 'linear' in k:
            if 'shortcut' in k:
                if 'weight' in k:
                    _eliminate_shortcut_weight(rgetattr(model, k))

            elif 'conv' in k:
                ratio = _set_layer(rgetattr(model, k), num_paths)

            elif 'bias' in k:
                _set_bias(rgetattr(model, k), ratio, num_paths)
                ratio = 1
            
            
def set_linear_layer(model, mu, sigma, num_paths=8, num_bins=10):
    """ 
    Setting the linear layer of the network appropriately once mean, std of 
    features has been figured out. 
    """
    bins = _get_bins(num_bins, mu, sigma)
    for (k, v) in model.named_parameters():
        if 'linear' in k:
                if 'weight' in k:
                    _make_average_layer(rgetattr(model, k), num_paths)
                elif 'bias' in k:
                    _make_linear_biases(rgetattr(model, k), bins)