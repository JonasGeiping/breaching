"""Various utility functions that can be re-used for multiple attacks."""

import torch
from .make_functional import make_functional_with_buffers


class GradientLoss(torch.nn.Module):
    """Super-class to simplify gradient-based objectives."""
    def __init__(self):
        super().__init__()

    def initialize(self, loss_fn, cfg_impl, local_hyperparams=None):
        self.loss_fn = loss_fn
        self.local_hyperparams = local_hyperparams
        if self.local_hyperparams is None:
            self._grad_fn = self._grad_fn_single_step
        else:
            self._grad_fn = self._grad_fn_multi_step

        self.cfg_impl = cfg_impl

    def forward(self, model, gradient_data, candidate, labels):
        gradient, loss = self._grad_fn(model, candidate, labels)
        with torch.autocast(candidate.device.type, enabled=self.cfg_impl.mixed_precision):
            objective = self.gradient_based_loss(gradient, gradient_data)
        return objective, loss

    def gradient_based_loss(self, gradient_rec, gradient_data):
        raise NotImplementedError()

    def _grad_fn_single_step(self, model, candidate, labels):
        """Compute a single gradient."""
        model.zero_grad()
        with torch.autocast(candidate.device.type, enabled=self.cfg_impl.mixed_precision):
            task_loss = self.loss_fn(model(candidate), labels)
        gradient = torch.autograd.grad(task_loss, model.parameters(), create_graph=True)
        return gradient, task_loss.detach()

    def _grad_fn_multi_step(self, model, candidate, labels):
        """Compute the full graph for multiple local update steps."""
        model.zero_grad()
        func_model, params, buffers = make_functional_with_buffers(model)
        initial_params = [p.clone() for p in params]

        seen_data_idx = 0
        for i in range(local_hyperparams['steps']):
            data = candidate[seen_data_idx: seen_data_idx + self.local_hyperparams['data_per_step']]
            seen_data_idx += self.local_hyperparams['data_per_step']
            seen_data_idx = seen_data_idx % candidate.shape[0]
            labels = self.local_hyperparams['labels'][i]
            with torch.autocast(candidate.device.type, enabled=self.cfg_impl.mixed_precision):
                task_loss = self.loss_fn(func_model(params, buffers, data), labels)

            step_gradient = torch.autograd.grad(task_loss, params, create_graph=True)

            # Update parameters in graph:
            params = [param - self.local_hyperparams['lr'] * grad for param, grad in zip(params, step_gradient)]

        # Finally return differentiable difference in state:
        gradient = [p_local - p_server for p_local, p_server in zip(params, initial_params)]

        # Return last loss as the "best" task loss
        return gradient, task_loss.detach()


class Euclidean(GradientLoss):
    """Gradient matching based on the euclidean distance of two gradient vectors."""

    def __init__(self, scale=1.0, **kwargs):
        super().__init__()
        self.scale = scale

    def gradient_based_loss(self, gradient_rec, gradient_data):
        objective = 0
        # param_count = 0
        for rec, data in zip(gradient_rec, gradient_data):
            objective += (rec - data).pow(2).sum()
            # param_count += rec.numel()
        return 0.5 * self.scale * objective  # / param_count


class CosineSimilarity(GradientLoss):
    """Gradient matching based on cosine similarity of two gradient vectors."""

    def __init__(self, scale=1.0, **kwargs):
        super().__init__()
        self.scale = scale

    def gradient_based_loss(self, gradient_rec, gradient_data):
        scalar_product, rec_norm, data_norm = 0.0, 0.0, 0.0
        for rec, data in zip(gradient_rec, gradient_data):
            scalar_product += (rec * data).sum()
            rec_norm += rec.pow(2).sum()
            data_norm += data.pow(2).sum()

        objective = 1 - scalar_product / rec_norm.sqrt() / data_norm.sqrt()

        return objective * self.scale


class MaskedCosineSimilarity(GradientLoss):
    """Gradient matching based on cosine similarity of two gradient vectors.
    All positions that are zero in the data gradient are masked.
    """

    def __init__(self, scale=1.0, mask_value=1e-6, **kwargs):
        super().__init__()
        self.scale = scale
        self.mask_value = 1e-6

    def gradient_based_loss(self, gradient_rec, gradient_data):
        scalar_product, rec_norm, data_norm = 0.0, 0.0, 0.0
        for rec, data in zip(gradient_rec, gradient_data):
            mask = data.abs() > self.mask_value
            scalar_product += (rec * data * mask).sum()
            rec_norm += (rec * mask).pow(2).sum()
            data_norm += (data * mask).pow(2).sum()

        objective = 1 - scalar_product / rec_norm.sqrt() / data_norm.sqrt()

        return objective * self.scale


class ParameterCosineSimilarity(GradientLoss):
    """Gradient matching based on cosine similarity of two gradient vectors."""

    def __init__(self, scale=1.0, **kwargs):
        super().__init__()
        self.scale = scale

    def gradient_based_loss(self, gradient_rec, gradient_data):
        similarities, counter = 0, 0
        for rec, data in zip(gradient_rec, gradient_data):
            similarities += torch.nn.functional.cosine_similarity(rec.view(-1), data.view(-1), dim=0, eps=1e-8)
            counter += 1
        objective = 1 - similarities / counter

        return objective * self.scale


class FastCosineSimilarity(GradientLoss):
    """Gradient matching based on cosine similarity of two gradient vectors.
    No gradient flows through the normalization."""

    def __init__(self, scale=1.0, **kwargs):
        super().__init__()
        self.scale = scale

    def gradient_based_loss(self, gradient_rec, gradient_data):
        scalar_product, rec_norm, data_norm = 0.0, 0.0, 0.0
        for rec, data in zip(gradient_rec, gradient_data):
            scalar_product += (rec * data).sum()
            rec_norm += rec.pow(2).sum().detach()
            data_norm += data.pow(2).sum().detach()

        objective = 1 - scalar_product / rec_norm.sqrt() / data_norm.sqrt()

        return objective * self.scale



class PearlmutterLoss(torch.nn.Module):
    """Use a first-order approximation of \nabla_x \nabla_g instead of the correct autograd value."""
    def __init__(self, scale=1.0, eps=1e-3, level_gradients=True, fudge_factor=1e-6, **kwargs):
        super().__init__()
        self.scale = scale
        self.eps = eps
        self.level_gradients = level_gradients

        self.fudge_factor = fudge_factor

    def initialize(self, loss_fn, cfg_impl, local_hyperparams=None):
        self.loss_fn = loss_fn
        self.local_hyperparams = local_hyperparams
        if self.local_hyperparams is not None:
            raise ValueError('This loss is only implemented for local gradients so far.')

        self.cfg_impl = cfg_impl

    def forward(self, model, gradient_data, candidate, labels):
        """Run through model twice to approximate 2nd-order derivative on residual."""

        model.zero_grad()
        with torch.autocast(candidate.device.type, enabled=self.cfg_impl.mixed_precision):
            task_loss = self.loss_fn(model(candidate), labels)
        # Compute both model gradients and candidate gradients
        *gradients, dLdx = torch.autograd.grad(task_loss, (*model.parameters(), candidate), create_graph=False)
        if self.level_gradients:
            grad_norm = torch.stack([g.pow(2).sum() for g in gradients]).sum().sqrt()
            torch._foreach_div_(gradients, max(grad_norm, self.fudge_factor))

        # Patch model and compute loss at offset vector:
        torch._foreach_sub_(gradients, gradient_data)  # Save one copy of the gradient list here
        residuals = gradients
        torch._foreach_add_(list(model.parameters()), residuals, alpha=self.eps)
        with torch.autocast(candidate.device.type, enabled=self.cfg_impl.mixed_precision):
            offset_task_loss = self.loss_fn(model(candidate), labels)
        dLv_dx = torch.autograd.grad(offset_task_loss, (candidate,), create_graph=False)[0]

        # Compute finite difference approximation
        candidate.grad += (dLv_dx - dLdx) / self.eps * self.scale
        # Unpatch model:
        torch._foreach_sub_(list(model.parameters()), residuals, alpha=self.eps)

        # Gradients have already been populated. Make sure not to kill them later on.
        with torch.no_grad():
            with torch.autocast(candidate.device.type, enabled=self.cfg_impl.mixed_precision):
                objective_value = 0.5 * self.scale * torch.stack([r.detach().pow(2).sum() for r in residuals]).sum()
        return objective_value, task_loss



objective_lookup = {'euclidean': Euclidean,
                    'cosine-similarity': CosineSimilarity,
                    'masked-cosine-similarity': MaskedCosineSimilarity,
                    'paramwise-cosine-similarity': ParameterCosineSimilarity,
                    'fast-cosine-similarity': FastCosineSimilarity,
                    'pearlmutter-loss': PearlmutterLoss,
                    }
