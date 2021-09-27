"""Implement server code. This will be short, if the server is honest."""

"""Payload template:

payload should be a dict containing the key data and a list of payloads. The length of this list is num_queries.
Each entry in the list of payloads contains at least the keys "parameters" and "buffers".
"""




import torch
from torch.hub import load_state_dict_from_url
from .malicious_modifications import ImprintBlock, RecoveryOptimizer
from .malicious_modifications.parameter_utils import _make_average_layer, _make_linear_biases
from .malicious_modifications.parameter_utils import _eliminate_shortcut_weight, _set_bias, _set_layer, rgetattr

class HonestServer():
    """Implement an honest server protocol."""

    def __init__(self, model, loss, cfg_case, setup=dict(dtype=torch.float, device=torch.device('cpu')), external_dataloader=None):
        """Inialize the server settings."""
        self.model = model
        if cfg_case.user.batch_norm_training:
            self.model.train()
        else:
            self.model.eval()
        self.loss = loss
        self.setup = setup

        self.num_queries = cfg_case.num_queries

        self.cfg_data = cfg_case.data  # Data configuration has to be shared across all parties to keep preprocessing consistent
        self.cfg_server = cfg_case.server

        self.external_dataloader = external_dataloader

        self.secrets = dict()  # Should be nothing in here

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
            elif model_state == 'linearized':
                with torch.no_grad():
                    if isinstance(module, torch.nn.BatchNorm2d):
                        module.weight.data = module.running_var.data.clone()
                        module.bias.data = module.running_mean.data.clone() + 10
                    if isinstance(module, torch.nn.Conv2d) and hasattr(module, 'bias'):
                        module.bias.data += 10
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
                state_dict = load_state_dict_from_url(
                    url, progress=True, map_location=torch.device('cpu'))['state_dict']
                for key in list(state_dict.keys()):
                    val = state_dict.pop(key)
                    # sanitized_key = key.replace('module.encoder_q.', '') # for mocov2
                    sanitized_key = key.replace('module.', '')
                    state_dict[sanitized_key] = val

                self.model.load_state_dict(state_dict, strict=True)  # The fc layer is not actually loaded here
            except FileNotFoundError:
                raise ValueError('No MoCo data found for this architecture.')

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


class MaliciousModelServer(HonestServer):
    """Implement a malicious server protocol."""

    def __init__(self, model, loss, cfg_case, setup=dict(dtype=torch.float, device=torch.device('cpu')), external_dataloader=None):
        """Inialize the server settings."""
        super().__init__(model, loss, cfg_case, setup, external_dataloader)
        self.model_state = 'custom'  # Do not mess with model parameters no matter what init is agreed upon
        self.secrets = dict()

    def prepare_model(self):
        """This server is not honest :>"""
        modified_model = self.model
        for key, val in self.cfg_server.model_modification.items():
            if key == 'ImprintBlock':
                input_dim = self.cfg_data.shape[0] * self.cfg_data.shape[1] * self.cfg_data.shape[2]
                block = ImprintBlock(input_dim, num_bins=val['num_bins'], alpha=val['alpha'])
                modified_model = torch.nn.Sequential(torch.nn.Flatten(),
                                                     block,
                                                     torch.nn.Unflatten(
                                                         dim=1, unflattened_size=tuple(self.cfg_data.shape)),
                                                     modified_model)
                self.secrets['ImprintBlock'] = dict(weight_idx=0, bias_idx=1)  # block position
        self.model = modified_model
        return self.model


class MaliciousParameterServer(HonestServer):
    """Implement a malicious server protocol."""

    def __init__(self, model, loss, cfg_case, setup=dict(dtype=torch.float, device=torch.device('cpu')), external_dataloader=None):
        """Inialize the server settings."""
        super().__init__(model, loss, cfg_case, setup, external_dataloader)
        self.secrets = dict()

        if 'optimization' in cfg_case.server.param_modification.keys():
            self.parameter_algorithm = RecoveryOptimizer(self.model, self.loss, self.cfg_data, cfg_case.impl,
                                                         cfg_optim=cfg_case.server.param_modification['optimization'],
                                                         setup=setup, external_data=external_dataloader)
            self.secrets['layers'] = cfg_case.server.param_modification.optimization.layers

    def prepare_model(self):
        """This server is not honest, but the model architecture stays normal."""
        return self.model

    def reconfigure_model(self, model_state):
        """Reinitialize, continue training or otherwise modify model parameters."""
        super().reconfigure_model(model_state)  # Load the benign model state first

        # Then do fun things:
        self.parameter_algorithm.optimize_recovery()


class PathParameterServer(HonestServer):

    def __init__(self, model, loss, cfg_case, setup=dict(dtype=torch.float, device=torch.device('cpu')), external_dataloader=None):
        """Inialize the server settings."""
        super().__init__(model, loss, cfg_case, setup, external_dataloader)
        self.secrets = dict()

        self.num_bins = self.cfg_server.num_bins
        self.alpha = self.cfg_server.alpha
        self.num_paths = self.cfg_server.num_paths

        self.bins, self.bin_val = self._get_bins()

    def prepare_model(self):
        """This server is not honest, we sneak in a hardtanh layer."""
        self.model.avgpool = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d((1, 1)),
                                                 torch.nn.Hardtanh(min_val=0, max_val=self.bin_val))
        return self.model

    def reconfigure_model(self, model_state):
        """Reinitialize, continue training or otherwise modify model parameters."""
        super().reconfigure_model(model_state)  # Load the benign model state first
        self._path_parameters()
        std, mu = self._feature_distribution()
        self._set_linear_layer(mu.item(), std.item())


    def _path_parameters(self):
        """Setting the paths in the network (feature extractor)"""
        ratio = 1
        for (name, _) in self.model.named_parameters():
            # if 'layer' in k or 'linear' in k: # skip the first conv layer?
            if 'layer' in name or 'linear' in name:
                if 'shortcut' in name:
                    if 'weight' in name:
                        _eliminate_shortcut_weight(rgetattr(self.model, name))
                elif 'conv' in name:
                    ratio = _set_layer(rgetattr(self.model, name), self.num_paths)
                elif 'bias' in name:
                    _set_bias(rgetattr(self.model, name), ratio, self.num_paths)
                    ratio = 1


    def _set_linear_layer(self, mu, sigma):
        """Setting the linear layer of the network appropriately once mean, std of features has been figured out."""
        for (name, _) in self.model.named_parameters():
            if 'linear' in name:
                if 'weight' in name:
                    _make_average_layer(rgetattr(self.model, name), self.num_paths)
                elif 'bias' in name:
                    _make_linear_biases(rgetattr(self.model, name), self.bins)


    def _feature_distribution(self):
        """Compute the mean and std of the feature layer of the given network."""
        features = dict()

        def named_hook(name):
            def hook_fn(module, input, output):
                features[name] = input[0]
            return hook_fn
        for name, module in reversed(list(self.model.named_modules())):
            if isinstance(module, (torch.nn.Hardtanh)):
                hook = module.register_forward_hook(named_hook(name))
                feature_layer_name = name
                break
        feats = []
        self.model.train()
        self.model.to(**self.setup)
        print('Computing feature distribution from external data.')
        for i, (inputs, target) in enumerate(self.external_dataloader):
            inputs = inputs.to(**self.setup)
            self.model(inputs)
            feats.append(features[feature_layer_name].detach().view(inputs.shape[0], -1).clone().cpu())

        std, mu = torch.std_mean(torch.cat(feats))
        print(f'Feature mean is {mu.item()}, feature std is {std.item()}.')
        self.model.eval()
        self.model.cpu()
        hook.remove()

        return std, mu

    def _get_bins(self):
        order_stats = [self._get_order_stats(r + 1, self.num_bins) for r in range(self.num_bins)]
        diffs = [order_stats[i] - order_stats[i + 1] for i in range(len(order_stats) - 1)]
        bin_val = -sum(diffs) / len(diffs)
        left_bins = [-i * bin_val for i in range(self.num_bins // 2 + 1)]
        right_bins = [i * bin_val for i in range(1, self.num_bins // 2)]
        left_bins.reverse()
        bins = left_bins + right_bins
        return bins, bin_val

    def _get_order_stats(self, r, n, mu=0, sigma=1):
        r"""
        r Order statistics can be computed as follows:
        E(r:n) = \mu + \Phi^{-1}\left( \frac{r-a}{n-2a+1} \sigma \right)
        where a = 0.375
        """
        from scipy.stats import norm
        return mu + norm.ppf((r - self.alpha) / (n - 2 * self.alpha + 1)) * sigma
