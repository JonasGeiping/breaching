"""Simple analytic attack that works for (dumb) fully connected models."""

import torch

from .base_attack import _BaseAttacker


class AnalyticAttacker(_BaseAttacker):
    """Implements a sanity-check analytic inversion

    Only works for a torch.nn.Sequential model with input-sized FC layers."""

    def __init__(self, model, loss_fn, cfg_attack, setup=dict(dtype=torch.float, device=torch.device("cpu"))):
        super().__init__(model, loss_fn, cfg_attack, setup)

    def __repr__(self):
        return f"""Attacker (of type {self.__class__.__name__})."""

    def reconstruct(self, server_payload, shared_data, server_secrets=None, dryrun=False):
        # Initialize stats module for later usage:
        rec_models, labels, stats = self.prepare_attack(server_payload, shared_data)

        # Main reconstruction: loop starts here:
        inputs_from_queries = []
        for model, user_gradient in zip(rec_models, shared_data["gradients"]):
            idx = len(user_gradient) - 1
            for layer in list(model)[::-1]:  # Only for torch.nn.Sequential
                if isinstance(layer, torch.nn.Linear):
                    bias_grad = user_gradient[idx]
                    weight_grad = user_gradient[idx - 1]
                    layer_inputs = self.invert_fc_layer(weight_grad, bias_grad, labels)
                    idx -= 2
                elif isinstance(layer, torch.nn.Flatten):
                    inputs = layer_inputs.reshape(shared_data["num_data_points"], *self.data_shape)
                else:
                    raise ValueError(f"Layer {layer} not supported for this sanity-check attack.")
            inputs_from_queries += [inputs]

        final_reconstruction = torch.stack(inputs_from_queries).mean(dim=0)
        reconstructed_data = dict(data=inputs, labels=labels)

        return reconstructed_data, stats

    def invert_fc_layer(self, weight_grad, bias_grad, image_positions):
        """The basic trick to invert a FC layer."""
        # By the way the labels are exactly at (bias_grad < 0).nonzero() if they are unique
        valid_classes = bias_grad != 0
        intermediates = weight_grad[valid_classes, :] / bias_grad[valid_classes, None]
        if len(image_positions) == 0:
            reconstruction_data = intermediates
        elif len(image_positions) == 1:
            reconstruction_data = intermediates.mean(dim=0)
        else:
            reconstruction_data = intermediates[image_positions]
        return reconstruction_data


class ImprintAttacker(AnalyticAttacker):
    """Abuse imprint secret for near-perfect attack success."""

    def reconstruct(self, server_payload, shared_data, server_secrets=None, dryrun=False):
        """This is somewhat hard-coded for images, but that is not a necessity."""
        # Initialize stats module for later usage:
        rec_models, labels, stats = self.prepare_attack(server_payload, shared_data)

        if "ImprintBlock" in server_secrets.keys():
            weight_idx = server_secrets["ImprintBlock"]["weight_idx"]
            bias_idx = server_secrets["ImprintBlock"]["bias_idx"]
            data_shape = server_secrets["ImprintBlock"]["shape"]
        else:
            raise ValueError(f"No imprint hidden in model {rec_models[0]} according to server.")

        bias_grad = shared_data["gradients"][0][bias_idx].clone()
        weight_grad = shared_data["gradients"][0][weight_idx].clone()
        if server_secrets["ImprintBlock"]["structure"] == "cumulative":
            for i in reversed(list(range(1, weight_grad.shape[0]))):
                weight_grad[i] -= weight_grad[i - 1]
                bias_grad[i] -= bias_grad[i - 1]

        image_positions = bias_grad.nonzero()
        layer_inputs = self.invert_fc_layer(weight_grad, bias_grad, [])

        if "decoder" in server_secrets["ImprintBlock"].keys():
            inputs = server_secrets["ImprintBlock"]["decoder"](layer_inputs)
        else:
            inputs = layer_inputs.reshape(layer_inputs.shape[0], *data_shape)[:, :3, :, :]
        if weight_idx > 0:  # An imprint block later in the network:
            inputs = torch.nn.functional.interpolate(
                inputs, size=self.data_shape[1:], mode="bicubic", align_corners=False
            )
        inputs = torch.max(torch.min(inputs, (1 - self.dm) / self.ds), -self.dm / self.ds)

        if len(labels) >= inputs.shape[0]:
            # Fill up with zero if not enough data can be found:
            missing_entries = torch.zeros(len(labels) - inputs.shape[0], *self.data_shape, **self.setup)
            inputs = torch.cat([inputs, missing_entries], dim=0)
        else:
            print(f"Initially produced {inputs.shape[0]} hits.")
            # Cut additional hits:
            # this rule is optimal for clean data with few bins:
            # best_guesses = torch.topk(bias_grad[bias_grad != 0].abs(), len(labels), largest=False)
            # this rule is best when faced with differential privacy:
            best_guesses = torch.topk(weight_grad.mean(dim=1)[bias_grad != 0].abs(), len(labels), largest=True)
            print(f"Reduced to {len(labels)} hits.")
            # print(best_guesses.indices.sort().values)
            inputs = inputs[best_guesses.indices]

        reconstructed_data = dict(data=inputs, labels=labels)
        return reconstructed_data, stats
