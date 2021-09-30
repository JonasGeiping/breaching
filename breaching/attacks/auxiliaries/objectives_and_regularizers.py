"""Various utility functions that can be re-used for multiple attacks."""

import torch
from .deepinversion import DeepInversionFeatureHook

# @torch.jit.script  # ?
class Euclidean(torch.nn.Module):
    """Gradient matching based on the euclidean distance of two gradient vectors."""

    def __init__(self, scale=1.0):
        super().__init__()
        self.scale = scale

    def forward(self, gradient_rec, gradient_data):
        objective = 0
        # param_count = 0
        for rec, data in zip(gradient_rec, gradient_data):
            objective += (rec - data).pow(2).sum()
            # param_count += rec.numel()
        return 0.5 * self.scale * objective  # / param_count


class CosineSimilarity(torch.nn.Module):
    """Gradient matching based on cosine similarity of two gradient vectors."""

    def __init__(self, scale=1.0):
        super().__init__()
        self.scale = scale

    def forward(self, gradient_rec, gradient_data):
        scalar_product, rec_norm, data_norm = 0.0, 0.0, 0.0
        for rec, data in zip(gradient_rec, gradient_data):
            scalar_product += (rec * data).sum()
            rec_norm += rec.pow(2).sum()
            data_norm += data.pow(2).sum()

        objective = 1 - scalar_product / rec_norm.sqrt() / data_norm.sqrt()

        return objective * self.scale


class MaskedCosineSimilarity(torch.nn.Module):
    """Gradient matching based on cosine similarity of two gradient vectors.
    All positions that are zero in the data gradient are masked.
    """

    def __init__(self, scale=1.0, mask_value=1e-6):
        super().__init__()
        self.scale = scale
        self.mask_value = 1e-6

    def forward(self, gradient_rec, gradient_data):
        scalar_product, rec_norm, data_norm = 0.0, 0.0, 0.0
        for rec, data in zip(gradient_rec, gradient_data):
            mask = data.abs() > self.mask_value
            scalar_product += (rec * data * mask).sum()
            rec_norm += (rec * self.mask_value).pow(2).sum()
            data_norm += (data * self.mask_value).pow(2).sum()

        objective = 1 - scalar_product / rec_norm.sqrt() / data_norm.sqrt()

        return objective * self.scale


class CosineSimilarityFast(torch.nn.Module):
    """Gradient matching based on cosine similarity of two gradient vectors."""

    def __init__(self):
        super().__init__()

    def forward(self, gradient_rec, gradient_data):
        scalar_product, rec_norm, data_norm = 0.0, 0.0, 0.0
        for rec, data in zip(gradient_rec, gradient_data):
            scalar_product += (rec * data).sum()
            rec_norm += rec.pow(2).sum().detach()
            data_norm += data.pow(2).sum().detach()

        objective = 1 - scalar_product / rec_norm.sqrt() / data_norm.sqrt()

        return objective


class TotalVariationOld(torch.nn.Module):
    """Computes the total variation value of an (image) tensor, based on its last two dimensions.
       Optionally also Color TV based on its last three dimensions."""

    def __init__(self, setup, scale=0.1, inner_exp=1, outer_exp=1, gamma=0.0):
        """scale is the overall scaling. inner_exp and outer_exp control isotropy vs anisotropy.
           gamma optionally also includes proper color TV via double opponents."""
        super().__init__()
        self.setup = setup
        self.scale = scale
        self.inner_exp = inner_exp
        self.outer_exp = outer_exp
        self.gamma = gamma
        if self.gamma > 0:
            self.forward = self._forward_full
        else:
            if self.inner_exp == self.outer_exp == 1:
                self.forward = self._forward_simplified
            else:
                self.forward = self._forward_variso

    def initialize(self, model, **kwargs):
        pass

    def _forward_simplified(self, tensor, **kwargs):
        """Anisotropic TV."""
        dx = torch.mean(torch.abs(tensor[:, :, :, :-1] - tensor[:, :, :, 1:]))
        dy = torch.mean(torch.abs(tensor[:, :, :-1, :] - tensor[:, :, 1:, :]))
        return self.scale * (dx + dy)

    def _forward_variso(self, tensor, **kwargs):
        """Anisotropic TV."""
        dx = torch.abs(tensor[:, :, :, :-1] - tensor[:, :, :, 1:]).pow(self.inner_exp)
        dy = torch.abs(tensor[:, :, :-1, :] - tensor[:, :, 1:, :]).pow(self.inner_exp)
        return self.scale * (dx + dy).pow(self.outer_exp).mean()

    def _forward_full(self, tensor, **kwargs):
        """Double opponent TV as in Aström and Schnörr "Double-Opponent Vectorial Total Variation".

        TODO: Extract and move this mess into a proper Conv2d operation for efficiency reasons...
        """
        q, p = self.inner_exp, self.outer_exp

        dxdy = ((tensor[:, :, :, :-1] - tensor[:, :, :, 1:]).pow(q) +
                (tensor[:, :, :-1, :] - tensor[:, :, 1:, :]).pow(q)).pow(p)

        rg = tensor[:, 0, :, :] - tensor[:, 1, :, :]
        rb = tensor[:, 0, :, :] - tensor[:, 2, :, :]
        gb = tensor[:, 1, :, :] - tensor[:, 2, :, :]

        rg_dxdy = ((rg[:, :, :-1] - rg[:, :, 1:]).pow(q) + (rg[:, :-1, :] - rg[:, 1:, :]).pow(q)).pow(p)
        rb_dxdy = ((rb[:, :, :-1] - rb[:, :, 1:]).pow(q) + (rb[:, :-1, :] - rb[:, 1:, :]).pow(q)).pow(p)
        gb_dxdy = ((gb[:, :, :-1] - gb[:, :, 1:]).pow(q) + (gb[:, :-1, :] - gb[:, 1:, :]).pow(q)).pow(p)

        return self.scale * (dxdy.mean() + self.gamma * (rg_dxdy.mean() + rb_dxdy.mean() + gb_dxdy.mean()))


class TotalVariation(torch.nn.Module):
    """Computes the total variation value of an (image) tensor, based on its last two dimensions.
       Optionally also Color TV based on its last three dimensions."""

    def __init__(self, setup, scale=0.1, inner_exp=1, outer_exp=1, double_opponents=False, eps=1e-8):
        """scale is the overall scaling. inner_exp and outer_exp control isotropy vs anisotropy.
           Optionally also includes proper color TV via double opponents."""
        super().__init__()
        self.setup = setup
        self.scale = scale
        self.inner_exp = inner_exp
        self.outer_exp = outer_exp
        self.eps = eps
        self.double_opponents = double_opponents

        grad_weight = torch.tensor([[0, 0, 0],
                                    [0, -1, 1],
                                    [0, 0, 0]], **setup).unsqueeze(0).unsqueeze(1)
        grad_weight = torch.cat((torch.transpose(grad_weight, 2, 3), grad_weight), 0)
        if self.double_opponents:
            self.groups = 6
        else:
            self.groups = 3
        grad_weight = torch.cat([grad_weight] * self.groups, 0)
        self.register_buffer('weight', grad_weight)

    def initialize(self, models, **kwargs):
        pass

    def forward(self, tensor, **kwargs):
        """Use a convolution-based approach."""
        if self.double_opponents:
            tensor = torch.cat([tensor,
                                tensor[:, 0:1, :, :] - tensor[:, 1:2, :, :],
                                tensor[:, 0:1, :, :] - tensor[:, 2:3, :, :],
                                tensor[:, 1:2, :, :] - tensor[:, 2:3, :, :]], dim=1)
        diffs = torch.nn.functional.conv2d(tensor, self.weight, None, stride=1,
                                           padding=1, dilation=1, groups=self.groups)
        squares = diffs.abs().pow(self.inner_exp)
        squared_sums = (squares[:, 0::2] + squares[:, 1::2] + self.eps).pow(self.outer_exp)
        return squared_sums.mean() * self.scale


class OrthogonalityRegularization(torch.nn.Module):
    """This is the orthogonality regularizer described Qian et al.,

    "MINIMAL CONDITIONS ANALYSIS OF GRADIENT-BASED RECONSTRUCTION IN FEDERATED LEARNING"
    """

    def __init__(self, setup, scale=0.1):
        super().__init__()
        self.setup = setup
        self.scale = scale

    def initialize(self, models, **kwargs):
        pass

    def forward(self, tensor, **kwargs):
        if tensor.shape[0] == 1:
            return 0
        else:
            B = tensor.shape[0]
            full_products = (tensor.unsqueeze(0) * tensor.unsqueeze(1)).pow(2).view(B, B, -1).mean(dim=2)
            idx = torch.arange(0, B, out=torch.LongTensor())
            full_products[idx, idx] = 0
            return full_products.sum()


class NormRegularization(torch.nn.Module):
    """Implement basic norm-based regularization, e.g. an L2 penalty."""

    def __init__(self, setup, scale=0.1, pnorm=2.0):
        super().__init__()
        self.setup = setup
        self.scale = scale
        self.pnorm = pnorm

    def initialize(self, models, **kwargs):
        pass

    def forward(self, tensor, **kwargs):
        return 1 / self.pnorm * tensor.pow(self.pnorm).mean() * self.scale


class DeepInversion(torch.nn.Module):
    """Implement a DeepInversion based regularization as proposed in DeepInversion as used for reconstruction in
       Yin et al, "See through Gradients: Image Batch Recovery via GradInversion".
    """

    def __init__(self, setup, scale=0.1, first_bn_multiplier=10):
        super().__init__()
        self.setup = setup
        self.scale = scale
        self.first_bn_multiplier = first_bn_multiplier

    def initialize(self, models, **kwargs):
        """Initialize forward hooks."""
        self.losses = [list() for model in models]
        for idx, model in enumerate(models):
            for module in model.modules():
                if isinstance(module, torch.nn.BatchNorm2d):
                    self.losses[idx].append(DeepInversionFeatureHook(module))


    def forward(self, tensor, **kwargs):
        rescale = [self.first_bn_multiplier] + [1. for _ in range(len(self.losses[0]) - 1)]
        feature_reg = 0
        for loss in self.losses:
            feature_reg += sum([mod.r_feature * rescale[idx] for (idx, mod) in enumerate(loss)])
        return self.scale * feature_reg


regularizer_lookup = dict(
    total_variation=TotalVariation,
    orthogonality=OrthogonalityRegularization,
    norm=NormRegularization,
    deep_inversion=DeepInversion,
)
