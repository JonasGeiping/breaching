"""Various utility functions that can be re-used for multiple attacks."""

import torch

# @torch.jit.script  # ?
class Euclidean(torch.nn.Module):
    """Gradient matching based on the euclidean distance of two gradient vectors."""
    def __init__(self):
        super().__init__()

    def forward(self, gradient_rec, gradient_data):
        objective = 0
        for rec, data in zip(gradient_rec, gradient_data):
            objective += (rec - data).pow(2).sum()
        return objective

class CosineSimilarity(torch.nn.Module):
    """Gradient matching based on cosine similarity of two gradient vectors."""
    def __init__(self):
        super().__init__()
        self._gradient_data_ref = None

    def forward(self, gradient_rec, gradient_data):
        scalar_product = 0.0
        rec_norm = 0.0
        if self._gradient_data_ref is not gradient_data:
            data_norm = 0.0
            for rec, data in zip(gradient_rec, gradient_data):
                scalar_product += (rec * data).sum()
                rec_norm += rec.pow(2).sum()
                data_norm += data.pow(2).sum()
            self._data_norm_cache = data_norm.detach()
            self._gradient_data_ref = gradient_data
        else:
            for rec, data in zip(gradient_rec, gradient_data):
                scalar_product += (rec * data).sum()
                rec_norm += rec.pow(2).sum()
            data_norm = self._data_norm_cache

        objective = 1 - scalar_product / rec_norm.sqrt() / data_norm.sqrt()

        return objective

class TotalVariation(torch.nn.Module):
    """Computes the total variation value of an (image) tensor, based on its last two dimensions."""
    def __init__(self, scale=0.1):
        super().__init__()
        self.scale = scale

    def forward(self, tensor):
        """Anisotropic TV."""
        dx = torch.mean(torch.abs(tensor[:, :, :, :-1] - tensor[:, :, :, 1:]))
        dy = torch.mean(torch.abs(tensor[:, :, :-1, :] - tensor[:, :, 1:, :]))
        return self.scale * (dx + dy)
