"""Simple analytic attack that works for (dumb) fully connected models."""

import torch
from collections import defaultdict

from .base_attack import _BaseAttacker


class ImprintAttacker(_BaseAttacker):
    """Implements a sanity-check analytic inversion

        Only works for a torch.nn.Sequential model with input-sized FC layers."""

    def __init__(self, model, loss_fn, cfg_attack, setup=dict(dtype=torch.float, device=torch.device('cpu'))):
        super().__init__(model, loss_fn, cfg_attack, setup)

    def reconstruct(self, server_payload, shared_data, dryrun=False):
        # Initialize stats module for later usage:
        stats = defaultdict(list)

        # Load preprocessing constants:
        self.data_shape = server_payload['data'].shape

        # Load server_payload into state:
        rec_models = self._construct_models_from_payload_and_buffers(server_payload, shared_data['buffers'])

        # Consider label information
        if shared_data['labels'] is None:
            labels = self._recover_label_information(shared_data)
        else:
            labels = shared_data['labels']


        # Main reconstruction: loop starts here:
        inputs_from_queries = []
        for model, user_gradient in zip(rec_models, shared_data['gradients']):
            idx = len(user_gradient) - 1
        
            bias_grad = user_gradient[idx]
            weight_grad = user_gradient[idx - 1]
            layer_inputs = self.invert_fc_layer(weight_grad, bias_grad, labels)  # a smarter reconstructon actually uses this value
            #inputs = layer_inputs.reshape(shared_data['num_data_points'], *self.data_shape)
            inputs = layer_inputs.reshape(layer_inputs.shape[0], *self.data_shape)
            idx -= 2
            inputs_from_queries += [inputs]

        final_reconstruction = torch.stack(inputs_from_queries).mean(dim=0)
        reconstructed_data = dict(data=inputs, labels=labels)

        return reconstructed_data, stats


    def invert_fc_layer(self, weight_grad, bias_grad, labels):
        """The basic trick to invert a FC layer."""
        # By the way the labels are exactly at (bias_grad < 0).nonzero() if they are unique
        valid_classes = bias_grad != 0
        intermediates = (weight_grad[valid_classes, :] / bias_grad[valid_classes, None])
        return intermediates
