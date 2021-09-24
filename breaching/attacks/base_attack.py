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
        self.setup = setup
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

        # Consider label information
        if shared_data['labels'] is None:
            labels = self._recover_label_information(shared_data)
        else:
            labels = shared_data['labels']
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
            new_model.to(**self.setup)

            with torch.no_grad():
                for param, server_state in zip(new_model.parameters(), parameters):
                    param.copy_(server_state.to(**self.setup))
                for buffer, server_state in zip(new_model.buffers(), buffers):
                    buffer.copy_(server_state.to(**self.setup))
            models.append(new_model)
        return models

    def _initialize_data(self, data_shape):
        init_type = self.cfg.init
        if init_type == 'randn':
            candidate = torch.randn(data_shape, **self.setup)
        elif init_type == 'rand':
            candidate = (torch.rand(data_shape, **self.setup) * 2) - 1.0
        elif init_type == 'zeros':
            candidate = torch.zeros(data_shape, **self.setup)
        candidate.requires_grad = True
        return candidate

    def _init_optimizer(self, candidate):

        optimizer, scheduler = optimizer_lookup([candidate], self.cfg.optim.optimizer, self.cfg.optim.step_size,
                                                scheduler=self.cfg.optim.step_size_decay, warmup=self.cfg.optim.warmup,
                                                max_iterations=self.cfg.optim.max_iterations)
        
        return optimizer, scheduler

    def _recover_label_information(self, user_data):
        raise NotImplementedError()
