"""Implementation for base attacker class.

Inherit from this class for a consistent interface with attack cases."""

import torch
from collections import defaultdict
import copy

from ..common import optimizer_lookup


class _BaseAttacker():
    """This is a template class for an attack."""

    def __init__(self, model, loss_fn, cfg_attack, setup=dict(dtype=torch.float, device=torch.device('cpu'))):
        self.cfg = cfg_attack
        self.memory_format = torch.channels_last if cfg_attack.impl.mixed_precision else torch.contiguous_format
        self.setup = dict(device=setup['device'], dtype=getattr(torch, cfg_attack.impl.dtype))
        self.model_template = copy.deepcopy(model)
        self.loss_fn = copy.deepcopy(loss_fn)

    def reconstruct(self, server_payload, shared_data, server_secrets=None, dryrun=False):

        stats = defaultdict(list)

        # Implement the attack here
        # The attack should consume the shared_data and server payloads and reconstruct
        raise NotImplementedError()

        return reconstructed_data, stats

    def prepare_attack(self, server_payload, shared_data):
        """Basic startup common to many reconstruction methods."""
        stats = defaultdict(list)

        # Load preprocessing constants:
        self.data_shape = server_payload['data'].shape
        self.dm = torch.as_tensor(server_payload['data'].mean, **self.setup)[None, :, None, None]
        self.ds = torch.as_tensor(server_payload['data'].std, **self.setup)[None, :, None, None]

        # Load server_payload into state:
        rec_models = self._construct_models_from_payload_and_buffers(server_payload, shared_data['buffers'])
        shared_data = self._cast_shared_data(shared_data)

        # Consider label information
        if shared_data['labels'] is None:
            labels = self._recover_label_information(shared_data)
        else:
            labels = shared_data['labels']

        # Condition gradients?
        if self.cfg.normalize_gradients:
            shared_data = self._normalize_gradients(shared_data)
        return rec_models, labels, stats

    def _construct_models_from_payload_and_buffers(self, server_payload, user_buffers):
        """Construct the model (or multiple) that is sent by the server and include user buffers if any."""

        # Load states into multiple models if necessary
        models = []
        for idx, payload in enumerate(server_payload['queries']):
            parameters = payload['parameters']
            if user_buffers is not None and idx < len(user_buffers):
                buffers = user_buffers[idx]
            else:
                buffers = payload['buffers']
            new_model = copy.deepcopy(self.model_template)
            new_model.to(**self.setup, memory_format=self.memory_format)

            with torch.no_grad():
                for param, server_state in zip(new_model.parameters(), parameters):
                    param.copy_(server_state.to(**self.setup))
                for buffer, server_state in zip(new_model.buffers(), buffers):
                    buffer.copy_(server_state.to(**self.setup))

            if self.cfg.impl.JIT == 'script':
                example_inputs = self._initialize_data((1, *self.data_shape))
                new_model = torch.jit.script(new_model, example_inputs=[(example_inputs,)])
            elif self.cfg.impl.JIT == 'trace':
                example_inputs = self._initialize_data((1, *self.data_shape))
                new_model = torch.jit.trace(new_model, example_inputs=example_inputs)
            models.append(new_model)
        return models

    def _cast_shared_data(self, shared_data):
        """Cast user data to reconstruction data type."""
        cast_grad_list = []
        for shared_grad in shared_data['gradients']:
            cast_grad_list += [[g.to(dtype=self.setup['dtype']) for g in shared_grad]]
        shared_data['gradients'] = cast_grad_list
        return shared_data

    def _initialize_data(self, data_shape):
        """Note that data is initialized "inside" the network normalization."""
        init_type = self.cfg.init
        if init_type == 'randn':
            candidate = torch.randn(data_shape, **self.setup)
        elif init_type == 'rand':
            candidate = (torch.rand(data_shape, **self.setup) * 2) - 1.0
        elif init_type == 'zeros':
            candidate = torch.zeros(data_shape, **self.setup)
        # Initializations from Wei et al, "A Framework for Evaluating Gradient Leakage
        #                                  Attacks in Federated Learning"
        elif any(c in init_type for c in ['red', 'green', 'blue', 'dark', 'light']):  # init_types like 'red-true'
            candidate = torch.zeros(data_shape, **self.setup)
            if 'light' in init_type:
                candidate = torch.ones(data_shape, **self.setup)
            else:
                nonzero_channel = 0 if 'red' in init_type else 1 if 'yellow' in init_type else 2
                candidate[:, nonzero_channel, :, :] = 1
            if '-true' in init_type:
                # Shift to be truly RGB, not just normalized RGB
                candidate = (candidate - self.dm) / self.ds
        elif 'patterned' in init_type:  # Look for init_type=rand-patterned-4
            pattern_width = int("".join(filter(str.isdigit, init_type)))
            if 'rand' in init_type:
                seed = torch.rand([1, 3, pattern_width, pattern_width], **self.setup)
            else:
                seed = torch.rand([1, 3, pattern_width, pattern_width], **self.setup)
            # Shape expansion:
            x_factor, y_factor = data_shape[2] // pattern_width, data_shape[3] // pattern_width
            candidate = torch.tile(seed, (1, 1, x_factor, y_factor))[:, :, :data_shape[2], :data_shape[3]]
        else:
            raise ValueError(f'Unknown initialization scheme {init_type} given.')

        candidate.to(memory_format=self.memory_format)
        candidate.requires_grad = True
        candidate.grad = torch.zeros_like(candidate)
        return candidate

    def _init_optimizer(self, candidate):

        optimizer, scheduler = optimizer_lookup([candidate], self.cfg.optim.optimizer, self.cfg.optim.step_size,
                                                scheduler=self.cfg.optim.step_size_decay, warmup=self.cfg.optim.warmup,
                                                max_iterations=self.cfg.optim.max_iterations)
        return optimizer, scheduler


    def _normalize_gradients(self, shared_data, fudge_factor=1e-6):
        """Normalize gradients to have norm of 1. No guarantees that this would be a good idea for FL updates."""
        for shared_grad in shared_data['gradients']:
            grad_norm = torch.stack([g.pow(2).sum() for g in shared_grad]).sum().sqrt()
            torch._foreach_div_(shared_grad, max(grad_norm, fudge_factor))
        return shared_data

    def _recover_label_information(self, user_data):
        raise NotImplementedError()
