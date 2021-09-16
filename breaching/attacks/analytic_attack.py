"""Simple analytic attack that works for (dumb) fully connected models."""

import torch

from .base_attack import _BaseAttacker


class AnalyticAttacker(_BaseAttacker):
    """Implements a sanity-check analytic inversion

        Only works for a torch.nn.Sequential model with input-sized FC layers."""

    def __init__(self, model, loss_fn, cfg_attack, setup=dict(dtype=torch.float, device=torch.device('cpu'))):
        super().__init__(model, loss_fn, cfg_attack, setup)

    def reconstruct(self, server_payload, shared_data, dryrun=False):
        # Initialize stats module for later usage:
        rec_models, labels, stats = self.prepare_attack(server_payload, shared_data)

        # Main reconstruction: loop starts here:
        inputs_from_queries = []
        for model, user_gradient in zip(rec_models, shared_data['gradients']):
            layer_inputs = None
            idx = 0
            for layer in model.modules():
                # Look for the first fully connected layer:
                if isinstance(layer, torch.nn.Linear):
                    bias_grad = user_gradient[idx + 1]
                    weight_grad = user_gradient[idx]

                    # Overestimate image positions:
                    image_positions = bias_grad.nonzero()
                    layer_inputs = self.invert_fc_layer(weight_grad, bias_grad, image_positions)
                    break
                elif isinstance(layer, (torch.nn.Conv2d, torch.nn.BatchNorm2d)):
                    idx += len(list(layer.parameters()))
                else:
                    # Skip all containers
                    pass

            if layer_inputs is None:
                raise ValueError('No linear layer found for analytic reconstruction.')
            else:
                inputs = layer_inputs.reshape(len(image_positions), *self.data_shape)
                inputs_from_queries += [inputs]

        final_reconstruction = torch.stack(inputs_from_queries).mean(dim=0)
        reconstructed_data = dict(data=inputs, labels=labels)

        return reconstructed_data, stats


    def invert_fc_layer(self, weight_grad, bias_grad, image_positions):
        """The basic trick to invert a FC layer."""
        reconstruction_data = (weight_grad[image_positions, :] / bias_grad[image_positions, None])
        return reconstruction_data
