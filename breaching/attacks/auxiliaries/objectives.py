"""Various utility functions that can be re-used for multiple attacks."""

import torch


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
            rec_norm += (rec * mask).pow(2).sum()
            data_norm += (data * mask).pow(2).sum()

        objective = 1 - scalar_product / rec_norm.sqrt() / data_norm.sqrt()

        return objective * self.scale


class ParameterCosineSimilarity(torch.nn.Module):
    """Gradient matching based on cosine similarity of two gradient vectors."""

    def __init__(self, scale=1.0):
        super().__init__()
        self.scale = scale

    def forward(self, gradient_rec, gradient_data):
        similarities, counter = 0, 0
        for rec, data in zip(gradient_rec, gradient_data):
            similarities += torch.nn.functional.cosine_similarity(rec.view(-1), data.view(-1), dim=0, eps=1e-8)
            counter += 1
        objective = 1 - similarities / counter

        return objective * self.scale


class FastCosineSimilarity(torch.nn.Module):
    """Gradient matching based on cosine similarity of two gradient vectors.
    No gradient flows through the normalization."""

    def __init__(self, scale=1.0):
        super().__init__()
        self.scale = scale

    def forward(self, gradient_rec, gradient_data):
        scalar_product, rec_norm, data_norm = 0.0, 0.0, 0.0
        for rec, data in zip(gradient_rec, gradient_data):
            scalar_product += (rec * data).sum()
            rec_norm += rec.pow(2).sum().detach()
            data_norm += data.pow(2).sum().detach()

        objective = 1 - scalar_product / rec_norm.sqrt() / data_norm.sqrt()

        return objective * self.scale



objective_lookup = {'euclidean': Euclidean,
                    'cosine-similarity': CosineSimilarity,
                    'masked-cosine-similarity': MaskedCosineSimilarity,
                    'paramwise-cosine-similarity': ParameterCosineSimilarity,
                    'fast-cosine-similarity': FastCosineSimilarity
                    }
