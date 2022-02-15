"""This is the recursive attack "R-GAP" from Zhu and Blaschko
"R-GAP: Recursive Gradient Attack on Privacy"

and a wrapper around the original code from https://github.com/JunyiZhu-AI/R-GAP/blob/main/main.py
"""

import torch
import numpy as np

from .base_attack import _BaseAttacker
from .auxiliaries.recursive_attack import (
    r_gap,
    peeling,
    fcn_reconstruction,
    inverse_udldu,
    derive_leakyrelu,
    inverse_leakyrelu,
)


class RecursiveAttacker(_BaseAttacker):
    """Implements a thin wrapper around the original R-GAP code.
    Check out the original implementation at https://github.com/JunyiZhu-AI/R-GAP/blob/main/main.py

    This implements work (best/only) with cnn6, e.g.
    python breach.py case=0_sanity_check attack=rgap case.model=cnn6 case/data=CIFAR10
    """

    def __init__(self, model, loss_fn, cfg_attack, setup=dict(dtype=torch.float, device=torch.device("cpu"))):
        super().__init__(model, loss_fn, cfg_attack, setup)

    def __repr__(self):
        return f"""Attacker (of type {self.__class__.__name__}) with settings:
               inversion:
                - step size: {self.cfg.inversion.step_size}
                - steps    : {self.cfg.inversion.step_size}
                """

    def reconstruct(self, server_payload, shared_data, server_secrets=None, dryrun=False):
        # Initialize stats module for later usage:
        rec_models, labels, stats = self.prepare_attack(server_payload, shared_data)

        # Get model feature shapes:
        feature_shapes = self._retrieve_feature_shapes(rec_models[0], shared_data)

        # Main reconstruction: loop starts here:
        inputs_from_queries = []
        for model, user_data in zip(rec_models, shared_data):
            inputs = self._rgap(list(user_data["gradients"]), labels, model, feature_shapes)
            inputs_from_queries += [torch.as_tensor(inputs, **self.setup)]

        final_reconstruction = torch.stack(inputs_from_queries).mean(dim=0)
        reconstructed_data = dict(data=final_reconstruction, labels=labels)

        return reconstructed_data, stats

    def _rgap(self, original_dy_dx, labels, model, x_shape):
        """This is the main part of https://github.com/JunyiZhu-AI/R-GAP/blob/main/main.py

        Rewritten in minor ways to all for some additional features such as torch.nn.Sequential architectures.
        Cross-reference the original implementation when running crucial comparisons."""
        # reconstruction procedure
        all_modules = list(model.modules())[::-1]

        k = None
        last_weight = []

        grad_idx = len(original_dy_dx) - 1

        # Initialize at last layer:
        module = all_modules[0]
        print(module)

        w = module.weight.detach().cpu().numpy()
        if module.bias is None:
            g = original_dy_dx[grad_idx].numpy()
            grad_idx -= 1
            udldu = np.dot(g.reshape(-1), w.reshape(-1))
            u = inverse_udldu(udldu, self.setup, step_size=self.cfg.inversion.step_size, steps=self.cfg.inversion.steps)

            # For simplicity assume y as known here. For details please refer to the paper.
            y = 0.1  # this is a simplification the orignal paper's code only works for binary class.

            print(f"pred_: {u/y:.1e}, udldu: {udldu:.1e}, udldu_:{-u/(1+np.exp(u)):.1e}")
            k = -y / (1 + np.exp(u))
            k = k.reshape(1, -1).astype(np.float32)
            x_, last_weight = fcn_reconstruction(k=k, gradient=g), w
        else:
            # Replace the old construction with a new one for multi-label classification:
            # Using a bias in the last layer:
            bias_grad = original_dy_dx[grad_idx].cpu().numpy()
            weight_grad = original_dy_dx[grad_idx - 1].cpu().numpy()
            grad_idx -= 2
            valid_classes = bias_grad != 0
            layer_inputs = (weight_grad[valid_classes, :] / bias_grad[valid_classes, None]).mean(axis=0)
            k = bias_grad.reshape(-1, 1).astype(np.float32)
            x_, last_weight = layer_inputs, w

        # Recurse through all other layers. Expects an alternating structure  of activation and linear layer
        # For R-GAP the activation has to be invertible
        for idx, module in enumerate(all_modules[1:]):
            print(module)

            if isinstance(module, (torch.nn.LeakyReLU, torch.nn.Identity)):  # or any activation function really!
                # derive activation function
                if isinstance(module, torch.nn.LeakyReLU):
                    da = derive_leakyrelu(x_, slope=module.negative_slope)
                elif isinstance(module, torch.nn.Identity):
                    da = derive_identity(x_)
                else:
                    ValueError(f"Please implement the derivative function of {module}")

                # back out neuron output
                if isinstance(module, torch.nn.LeakyReLU):
                    out = inverse_leakyrelu(x_, slope=module.negative_slope)
                elif isinstance(module, torch.nn.Identity):
                    out = inverse_identity(x_)
                else:
                    ValueError(f"Please implement the inverse function of {module}")
                if hasattr(all_modules[idx], "padding"):
                    padding = all_modules[idx].padding[0]  # JG: lets hope that the padding is symmetric
                else:
                    padding = 0

                # For a mini-batch setting, reconstruct the combination
                in_shape = np.array(x_shape[idx // 2])
                in_shape[0] = 1
                # peel off padded entries
                x_mask = peeling(in_shape=in_shape, padding=padding)
                k = np.multiply(np.matmul(last_weight.transpose(), k)[x_mask], da.transpose())

            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                g = original_dy_dx[grad_idx].numpy()  # this only works for the given nets with bias=None
                grad_idx -= 1
                w = module.weight.detach().cpu().numpy()
                if isinstance(module, torch.nn.Conv2d):
                    x_, last_weight = r_gap(out=out, k=k, x_shape=x_shape[idx // 2], module=module, g=g, weight=w)
                else:
                    # In consideration of computational efficiency, for FCN only takes gradient constraints into account.
                    x_, last_weight = fcn_reconstruction(k=k, gradient=g), w

        inputs = x_.reshape([1, *self.data_shape])
        return inputs

    def _retrieve_feature_shapes(self, model, shared_data):
        """Retrieve x_shape by hooking into the model and recording it.

        Feature shapes are returned in reverse order!"""
        feature_shapes = []

        def hook_fn(module, input, output):
            feature_shapes.append(input[0].shape)

        hooks_list = []
        for module in model.modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                hooks_list.append(module.register_forward_hook(hook_fn))

        # Run the model with random data to query it
        # This requires the model to be in eval mode!
        model(torch.randn([shared_data[0]["metadata"]["num_data_points"], *self.data_shape], **self.setup))
        for hook in hooks_list:
            hook.remove()

        feature_shapes.reverse()
        return feature_shapes
