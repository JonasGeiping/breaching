"""Implement server code. This will be short, if the server is honest."""

"""Payload template:

payload should be a dict containing the key data and a list of payloads. The length of this list is num_queries.
Each entry in the list of payloads contains at least the keys "parameters" and "buffers".
"""
import torch
from torch.hub import load_state_dict_from_url

from .malicious_modifications import ImprintBlock

class HonestServer():
    """Implement an honest server protocol."""

    def __init__(self, model, loss, cfg_server, num_queries=1, cfg_data=None, training=False):
        """Inialize the server settings."""
        self.model = model
        if training:
            self.model.training()
        else:
            self.model.eval()
        self.loss = loss

        self.num_queries = num_queries

        self.cfg_data = cfg_data  # Data configuration has to be shared across all parties to keep preprocessing consistent
        self.cfg_server = cfg_server

    def reconfigure_model(self, model_state):
        """Reinitialize, continue training or otherwise modify model parameters in a benign way."""
        for name, module in self.model.named_modules():
            if model_state == 'untrained':
                if hasattr(module, 'reset_parameters'):
                    module.reset_parameters()
            elif model_state == 'trained':
                pass  # model was already loaded as pretrained model
            elif model_state == 'moco':
                pass  # will be loaded below
            elif model_state == 'orthogonal':
                # reinit model with orthogonal parameters:
                if hasattr(module, 'reset_parameters'):
                    module.reset_parameters()
                if 'conv' in name or 'linear' in name:
                    torch.nn.init.orthogonal_(module.weight, gain=1)
        if model_state == 'moco':
            try:
                # url = 'https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v2_800ep/moco_v2_800ep_pretrain.pth.tar'
                # url = 'https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v2_200ep/moco_v2_200ep_pretrain.pth.tar'
                url = 'https://dl.fbaipublicfiles.com/moco-v3/r-50-1000ep/linear-1000ep.pth.tar'
                state_dict = load_state_dict_from_url(url, progress=True, map_location=torch.device('cpu'))['state_dict']
                for key in list(state_dict.keys()):
                    val = state_dict.pop(key)
                    # sanitized_key = key.replace('module.encoder_q.', '') # for mocov2
                    sanitized_key = key.replace('module.', '')
                    state_dict[sanitized_key] = val

                self.model.load_state_dict(state_dict, strict=True)  # The fc layer is not actually loaded here
            except FileNotFoundError:
                raise ValueError('no MoCo data found for this architecture.')


    def distribute_payload(self):
        """Server payload to send to users. These are only references, to simplfiy the simulation."""

        queries = []
        for round in range(self.num_queries):
            self.reconfigure_model(self.cfg_server.model_state)

            honest_model_parameters = [p for p in self.model.parameters()]  # do not send only the generators
            honest_model_buffers = [b for b in self.model.buffers()]
            queries.append(dict(parameters=honest_model_parameters, buffers=honest_model_buffers))
        return dict(queries=queries, data=self.cfg_data)

    def prepare_model(self):
        """This server is honest."""
        return self.model


class MaliciousServer(HonestServer):
    """Implement a malicious server protocol."""

    def __init__(self, model, loss, cfg_server, num_queries=1, cfg_data=None, training=False):
        """Inialize the server settings."""
        super().__init__(model, loss, cfg_server, num_queries, cfg_data, training)
        self.model_state = 'custom'  # Do not mess with model parameters no matter what init is agreed upon

    def prepare_model(self):
        """This server is not honest :>"""
        modified_model = self.model
        for key, val in self.cfg_server.model_modification.items():
            if key == 'ImprintBlock':
                input_dim = self.cfg_data.shape[0] * self.cfg_data.shape[1] * self.cfg_data.shape[2]
                block = ImprintBlock(input_dim, num_bins=val['num_bins'], alpha=val['alpha'])
                modified_model = torch.nn.Sequential(torch.nn.Flatten(),
                                                     block,
                                                     torch.nn.Unflatten(dim=1, unflattened_size=tuple(self.cfg_data.shape)),
                                                     modified_model)
        self.model = modified_model
        return self.model
