"""Various utility functions that can be re-used for multiple attacks."""

import torch

# @torch.jit.script  # ?
class Euclidean(torch.nn.Module):
    """Gradient matching based on the euclidean distance of two gradient vectors."""
    def __init__(self):
        super().__init__()

    def forward(self, gradient_rec, gradient_data):
        objective = 0
        param_count = 0
        for rec, data in zip(gradient_rec, gradient_data):
            objective += (rec - data).pow(2).sum()
            param_count += rec.numel()
        return 0.5 * objective / param_count

class CosineSimilarity(torch.nn.Module):
    """Gradient matching based on cosine similarity of two gradient vectors."""
    def __init__(self):
        super().__init__()

    def forward(self, gradient_rec, gradient_data):
        scalar_product, rec_norm, data_norm = 0.0, 0.0, 0.0
        for rec, data in zip(gradient_rec, gradient_data):
            scalar_product += (rec * data).sum()
            rec_norm += rec.pow(2).sum()
            data_norm += data.pow(2).sum()

        objective = 1 - scalar_product / rec_norm.sqrt() / data_norm.sqrt()

        return objective


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


class TotalVariation(torch.nn.Module):
    """Computes the total variation value of an (image) tensor, based on its last two dimensions."""
    def __init__(self, scale=0.1, inner_exp=1, outer_exp=1):
        super().__init__()
        self.scale = scale
        self.inner_exp = 1
        self.outer_exp = 1
        if self.inner_exp == self.outer_exp == 1:
            self.forward = self._forward_simplified
        else:
            self.forward = self._forward_full

    def _forward_simplified(self, tensor):
        """Anisotropic TV."""
        dx = torch.mean(torch.abs(tensor[:, :, :, :-1] - tensor[:, :, :, 1:]))
        dy = torch.mean(torch.abs(tensor[:, :, :-1, :] - tensor[:, :, 1:, :]))
        return self.scale * (dx + dy)

    def _forward_full(self, tensor):
        """Anisotropic TV."""
        dx = torch.abs(tensor[:, :, :, :-1] - tensor[:, :, :, 1:]).pow(self.inner_exp)
        dy = torch.abs(tensor[:, :, :-1, :] - tensor[:, :, 1:, :]).pow(self.inner_exp)
        return self.scale * (dx + dy).pow(self.outer_exp).mean()


class OrthogonalityRegularization(torch.nn.Module):
    """This is the orthogonality regularizer described Qian et al.,

    "MINIMAL CONDITIONS ANALYSIS OF GRADIENT-BASED RECONSTRUCTION IN FEDERATED LEARNING"
    """
    def __init__(self, scale=0.1):
        super().__init__()
        self.scale = scale

    def forward(self, tensor):
        if tensor.shape[0] == 1:
            return 0
        else:
            B = tensor.shape[0]
            full_products = (tensor.unsqueeze(0) * tensor.unsqueeze(1)).pow(2).view(B, B, -1).mean(dim=2)
            idx = torch.arange(0, B, out=torch.LongTensor())
            full_products[idx, idx] = 0
            return full_products.sum()
