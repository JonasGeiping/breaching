"""Implement server code. This will be short, if the server is honest."""

"""Payload template:

payload should be a dict containing the key data and a list of payloads. The length of this list is num_queries.
Each entry in the list of payloads contains at least the keys "parameters" and "buffers".
"""

import torch
from torch.hub import load_state_dict_from_url
from .malicious_modifications import ImprintBlock, RecoveryOptimizer, SparseImprintBlock, OneShotBlock
from .malicious_modifications.parameter_utils import introspect_model, replace_module_by_instance

from .aux_training import train_encoder_decoder
from .malicious_modifications.feat_decoders import generate_decoder


class HonestServer:
    """Implement an honest server protocol.

    This class loads and selects the initial model and then sends this model to the (simulated) user.
    If multiple queries are possible, then the query sent to the user will contain multiple model states.

    Central output: self.distribute_payload -> Dict[queries=List[Queries], data=DataHyperparams]
    For each Query -> Dict[parameters=parameters, buffers=buffers]
    """

    THREAT = "Honest-but-curious"

    def __init__(
        self, model, loss, cfg_case, setup=dict(dtype=torch.float, device=torch.device("cpu")), external_dataloader=None
    ):
        """Inialize the server settings."""
        self.model = model
        self.model.eval()

        self.loss = loss
        self.setup = setup

        self.num_queries = cfg_case.num_queries

        self.cfg_data = (
            cfg_case.data
        )  # Data configuration has to be shared across all parties to keep preprocessing consistent
        self.cfg_server = cfg_case.server

        self.external_dataloader = external_dataloader

        self.secrets = dict()  # Should be nothing in here

    def __repr__(self):
        return f"""Server (of type {self.__class__.__name__}) with settings:
    Threat model: {self.THREAT}
    Has external/public data: {self.cfg_server.has_external_data}

    Model:
        model specification: {str(self.model.name)}
        model state: {self.cfg_server.model_state}
        {f'public buffers: {self.cfg_server.provide_public_buffers}' if len(list(self.model.buffers())) > 0 else ''}

    Secrets: {self.secrets}
    """

    def reconfigure_model(self, model_state):
        """Reinitialize, continue training or otherwise modify model parameters in a benign way."""
        for name, module in self.model.named_modules():
            if model_state == "untrained":
                if hasattr(module, "reset_parameters"):
                    module.reset_parameters()
            elif model_state == "trained":
                pass  # model was already loaded as pretrained model
            elif model_state == "moco":
                pass  # will be loaded below
            elif model_state == "linearized":
                with torch.no_grad():
                    if isinstance(module, torch.nn.BatchNorm2d):
                        module.weight.data = module.running_var.data.clone()
                        module.bias.data = module.running_mean.data.clone() + 10
                    if isinstance(module, torch.nn.Conv2d) and hasattr(module, "bias"):
                        module.bias.data += 10
            elif model_state == "orthogonal":
                # reinit model with orthogonal parameters:
                if hasattr(module, "reset_parameters"):
                    module.reset_parameters()
                if "conv" in name or "linear" in name:
                    torch.nn.init.orthogonal_(module.weight, gain=1)
        if model_state == "moco":
            try:
                # url = 'https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v2_800ep/moco_v2_800ep_pretrain.pth.tar'
                # url = 'https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v2_200ep/moco_v2_200ep_pretrain.pth.tar'
                url = "https://dl.fbaipublicfiles.com/moco-v3/r-50-1000ep/linear-1000ep.pth.tar"
                state_dict = load_state_dict_from_url(url, progress=True, map_location=torch.device("cpu"))[
                    "state_dict"
                ]
                for key in list(state_dict.keys()):
                    val = state_dict.pop(key)
                    # sanitized_key = key.replace('module.encoder_q.', '') # for mocov2
                    sanitized_key = key.replace("module.", "")
                    state_dict[sanitized_key] = val

                self.model.load_state_dict(state_dict, strict=True)  # The fc layer is not actually loaded here
            except FileNotFoundError:
                raise ValueError("No MoCo data found for this architecture.")

    def distribute_payload(self):
        """Server payload to send to users. These are only references to simplfiy the simulation."""

        queries = []
        for round in range(self.num_queries):
            self.reconfigure_model(self.cfg_server.model_state)
            honest_model_parameters = [p for p in self.model.parameters()]  # do not send only the generators
            if self.cfg_server.provide_public_buffers:
                honest_model_buffers = [b for b in self.model.buffers()]
            else:
                honest_model_buffers = None
            queries.append(dict(parameters=honest_model_parameters, buffers=honest_model_buffers))
        return dict(queries=queries, data=self.cfg_data)

    def prepare_model(self):
        """This server is honest."""
        return self.model


class MaliciousModelServer(HonestServer):
    """Implement a malicious server protocol.

    This server is now also able to modify the model maliciously, before sending out payloads.
    Architectural changes (via self.prepare_model) are triggered before instantation of user objects.
    These architectural changes can also be understood as a 'malicious analyst' and happen first.
    """

    THREAT = "Malicious (Analyst)"

    def __init__(
        self, model, loss, cfg_case, setup=dict(dtype=torch.float, device=torch.device("cpu")), external_dataloader=None
    ):
        """Inialize the server settings."""
        super().__init__(model, loss, cfg_case, setup, external_dataloader)
        self.model_state = "custom"  # Do not mess with model parameters no matter what init is agreed upon
        self.secrets = dict()

    def prepare_model(self):
        """This server is not honest :>"""

        modified_model = self.model
        if self.cfg_server.model_modification.type == "ImprintBlock":
            block_fn = ImprintBlock
        elif self.cfg_server.model_modification.type == "SparseImprintBlock":
            block_fn = SparseImprintBlock
        elif self.cfg_server.model_modification.type == "OneShotBlock":
            block_fn = OneShotBlock
        else:
            raise ValueError("Unknown modification")

        modified_model, secrets = self._place_malicious_block(
            modified_model, block_fn, **self.cfg_server.model_modification
        )
        self.secrets["ImprintBlock"] = secrets

        if self.cfg_server.model_modification.position is not None:
            if self.cfg_server.model_modification.type == "SparseImprintBlock":
                block_fn = type(None)  # Linearize the full model for SparseImprint
            if self.cfg_server.model_modification.handle_preceding_layers == "identity":
                self._linearize_up_to_imprint(modified_model, block_fn)
            elif self.cfg_server.model_modification.handle_preceding_layers == "VAE":
                # Train preceding layers to be a VAE up to the target dimension
                modified_model, decoder = self.train_encoder_decoder(modified_model, block_fn)
                self.secrets["ImprintBlock"]["decoder"] = decoder
            else:
                # Otherwise do not modify the preceding layers. The attack then returns the layer input at this position directly
                pass

        # Reduce failures in later layers:
        # Note that this clashes with the VAE option!
        self._normalize_throughput(
            modified_model, gain=self.cfg_server.model_gain, trials=self.cfg_server.normalize_rounds
        )
        self.model = modified_model
        return self.model

    def _place_malicious_block(
        self, modified_model, block_fn, type, position=None, handle_preceding_layers=None, **kwargs
    ):
        """The block is placed directly before the named module. If none is given, the block is placed at the start."""
        if position is None:
            input_dim = self.cfg_data.shape[0] * self.cfg_data.shape[1] * self.cfg_data.shape[2]
            block = block_fn(input_dim, **kwargs)
            original_name = modified_model.name
            modified_model = torch.nn.Sequential(
                torch.nn.Flatten(),
                block,
                torch.nn.Unflatten(dim=1, unflattened_size=tuple(self.cfg_data.shape)),
                modified_model,
            )
            modified_model.name = original_name
            secrets = dict(weight_idx=0, bias_idx=1, shape=tuple(self.cfg_data.shape), structure=block.structure)
        else:
            block_found = False
            for name, module in modified_model.named_modules():
                if position in name:  # give some leeway for additional containers.
                    feature_shapes = introspect_model(modified_model, tuple(self.cfg_data.shape))
                    data_shape = feature_shapes[name]["shape"][1:]
                    print(f"Block inserted at feature shape {data_shape}.")
                    module_to_be_modified = module
                    block_found = True
                    break

            if not block_found:
                raise ValueError(f"Could not find module {position} in model to insert layer.")
            input_dim = torch.prod(torch.as_tensor(data_shape))
            block = block_fn(input_dim, **kwargs)

            replacement = torch.nn.Sequential(
                torch.nn.Flatten(), block, torch.nn.Unflatten(dim=1, unflattened_size=data_shape), module_to_be_modified
            )
            replace_module_by_instance(modified_model, module_to_be_modified, replacement)
            for idx, param in enumerate(modified_model.parameters()):
                if param is block.linear0.weight:
                    weight_idx = idx
                if param is block.linear0.bias:
                    bias_idx = idx
            secrets = dict(weight_idx=weight_idx, bias_idx=bias_idx, shape=data_shape, structure=block.structure)

        return modified_model, secrets

    def _linearize_up_to_imprint(self, model, block_fn):
        """This linearization option only works for a ResNet architecture."""
        first_conv_set = False  # todo: make this nice
        for name, module in self.model.named_modules():
            if isinstance(module, block_fn):
                break
            with torch.no_grad():
                if isinstance(module, torch.nn.BatchNorm2d):
                    # module.weight.data = (module.running_var.data.clone() + module.eps).sqrt()
                    # module.bias.data = module.running_mean.data.clone()
                    torch.nn.init.ones_(module.running_var)
                    torch.nn.init.ones_(module.weight)
                    torch.nn.init.zeros_(module.running_mean)
                    torch.nn.init.zeros_(module.bias)
                if isinstance(module, torch.nn.Conv2d):
                    if not first_conv_set:
                        torch.nn.init.dirac_(module.weight)
                        num_groups = module.out_channels // 3
                        module.weight.data[: num_groups * 3] = torch.cat(
                            [module.weight.data[:3, :3, :, :]] * num_groups
                        )
                        first_conv_set = True
                    else:
                        torch.nn.init.zeros_(module.weight)  # this is the resnet rule
                if "downsample.0" in name:
                    torch.nn.init.dirac_(module.weight)
                    num_groups = module.out_channels // module.in_channels
                    concat = torch.cat(
                        [module.weight.data[: module.in_channels, : module.in_channels, :, :]] * num_groups
                    )
                    module.weight.data[: num_groups * module.in_channels] = concat
                if isinstance(module, torch.nn.ReLU):
                    replace_module_by_instance(model, module, torch.nn.Identity())

    @torch.inference_mode()
    def _normalize_throughput(self, model, gain=1, trials=1, bn_modeset=False):
        """Reset throughput to be within standard mean and gain-times standard deviation."""
        features = dict()

        def named_hook(name):
            def hook_fn(module, input, output):
                features[name] = output

            return hook_fn

        if trials > 0:
            print(f"Normalizing model throughput with gain {gain}...")
            model.to(**self.setup)
        for round in range(trials):
            if not bn_modeset:
                for name, module in model.named_modules():
                    if isinstance(module, (torch.nn.Conv2d, torch.nn.BatchNorm2d)):
                        if isinstance(module, torch.nn.Conv2d) and module.bias is None:
                            if "downsample.0" in name:
                                module.weight.data.zero_()
                                print(f"Reset weight in downsample {name} to zero.")
                            continue

                        if "downsample.1" in name:
                            continue
                        hook = module.register_forward_hook(named_hook(name))
                        if self.external_dataloader is not None:
                            random_data_sample = next(iter(self.external_dataloader))[0].to(**self.setup)
                        else:
                            random_data_sample = torch.randn(
                                self.cfg_data.batch_size, *self.cfg_data.shape, **self.setup
                            )

                        model(random_data_sample)
                        std, mu = torch.std_mean(features[name])
                        print(f"Current mean of layer {name} is {mu.item()}, std is {std.item()} in round {round}.")

                        with torch.no_grad():
                            module.weight.data /= std / gain + 1e-8
                            module.bias.data -= mu / (std / gain + 1e-8)
                        hook.remove()
                        del features[name]
            else:
                model.train()
                if self.external_dataloader is not None:
                    random_data_sample = next(iter(self.external_dataloader))[0].to(**self.setup)
                else:
                    random_data_sample = torch.randn(self.cfg_data.batch_size, *self.cfg_data.shape, **self.setup)
                model(random_data_sample)
                model.eval()
        # Free up GPU:
        model.to(device=torch.device("cpu"))

    def train_encoder_decoder(self, modified_model, block_fn):
        """Train a compressed code (with VAE) that will then be found by the attacker."""
        if self.external_dataloader is None:
            raise ValueError("External data is necessary to train an optimal encoder/decoder structure.")

        # Unroll model up to imprint block
        # For now only the last position is allowed:
        layer_cake = list(modified_model.children())
        encoder = torch.nn.Sequential(*(layer_cake[:-1]), torch.nn.Flatten())
        decoder = generate_decoder(modified_model)
        print(encoder)
        print(decoder)
        stats = train_encoder_decoder(encoder, decoder, self.external_dataloader, self.setup)
        return modified_model, decoder


class MaliciousParameterServer(HonestServer):
    """Implement a malicious server protocol.

    This server cannot modify the 'honest' model architecture posed by an analyst,
    but may modify the model parameters freely."""

    THREAT = "Malicious (Parameters)"

    def __init__(
        self, model, loss, cfg_case, setup=dict(dtype=torch.float, device=torch.device("cpu")), external_dataloader=None
    ):
        """Inialize the server settings."""
        super().__init__(model, loss, cfg_case, setup, external_dataloader)
        self.secrets = dict()

        if "optimization" in cfg_case.server.param_modification.keys():
            self.parameter_algorithm = RecoveryOptimizer(
                self.model,
                self.loss,
                self.cfg_data,
                cfg_case.impl,
                cfg_optim=cfg_case.server.param_modification["optimization"],
                setup=setup,
                external_data=external_dataloader,
            )
            self.secrets["layers"] = cfg_case.server.param_modification.optimization.layers

    def prepare_model(self):
        """This server is not honest, but the model architecture stays normal."""
        return self.model

    def reconfigure_model(self, model_state):
        """Reinitialize, continue training or otherwise modify model parameters."""
        super().reconfigure_model(model_state)  # Load the benign model state first

        # Then do fun things:
        self.parameter_algorithm.optimize_recovery()


class PathParameterServer(HonestServer):
    def __init__(
        self, model, loss, cfg_case, setup=dict(dtype=torch.float, device=torch.device("cpu")), external_dataloader=None
    ):
        """Inialize the server settings."""
        super().__init__(model, loss, cfg_case, setup, external_dataloader)
        self.secrets = dict()
        self.num_bins = cfg_case.num_paths
        self.alpha = self.cfg_server.alpha
        self.num_paths = self.cfg_server.num_paths

        self.bins, self.bin_val = self._get_bins()

    def prepare_model(self):
        """This server is not honest, we sneak in a hardtanh layer."""
        """
        self.model.avgpool = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d((1, 1)),
                                                 torch.nn.Hardtanh(min_val=0, max_val=self.bin_val))
        """
        self.model.hardtan = torch.nn.Hardtanh(min_val=0, max_val=self.bin_val)
        return self.model

    def reconfigure_model(self, model_state):
        """Reinitialize, continue training or otherwise modify model parameters."""
        super().reconfigure_model(model_state)  # Load the benign model state first
        self._path_parameters()
        self.std, self.mu = self._feature_distribution()
        self.bins = (torch.tensor(self.bins) - (self.mu / self.std)).tolist()
        self._set_linear_layer(self.mu.item(), self.std.item())

    def _path_parameters(self):
        """Setting the paths in the network (feature extractor)
        TODO: This shouldn't go just by names in the future
        """
        ratio = 1
        for (name, _) in self.model.named_parameters():
            if "shortcut" in name:
                if "weight" in name:
                    _eliminate_shortcut_weight(rgetattr(self.model, name))
            elif "conv" in name:
                ratio = _set_layer(rgetattr(self.model, name), self.num_paths)
            elif "bias" in name:
                ratio = 1

    def _set_linear_layer(self, mu, sigma):
        """Setting the linear layer of the network appropriately once mean, std of features have been figured out.
        TODO: This shouldn't go just by names in the future
        """
        for (name, _) in self.model.named_parameters():
            if "path_mod" in name:
                if "weight" in name:
                    rgetattr(self.model, name).data = (1 / self.std) * torch.ones_like(rgetattr(self.model, name).data)
                    _set_pathmod_layer(rgetattr(self.model, name), self.num_paths)

                elif "bias" in name:
                    _make_linear_biases(rgetattr(self.model, name), self.bins)

            if "linear" in name or "classifier" in name:
                if "weight" in name:
                    rgetattr(self.model, name).data = torch.ones_like(rgetattr(self.model, name).data)
                elif "bias" in name:
                    rgetattr(self.model, name).data = torch.zeros_like(rgetattr(self.model, name).data)

    def _feature_distribution(self):
        """Compute the mean and std of the feature layer of the given network."""
        features = dict()

        def named_hook(name):
            def hook_fn(module, input, output):
                features[name] = input[0]

            return hook_fn

        # for name, module in reversed(list(self.model.named_modules())):
        for name, module in list(self.model.named_modules()):
            # if isinstance(module, (torch.nn.Hardtanh)):
            if isinstance(module, (torch.nn.Linear)):
                hook = module.register_forward_hook(named_hook(name))
                feature_layer_name = name
                break
        feats = []
        self.model.train()
        self.model.to(**self.setup)
        print(f"Computing feature distribution before the {feature_layer_name} layer from external data.")
        for i, (inputs, target) in enumerate(self.external_dataloader):
            inputs = inputs.to(**self.setup)
            self.model(inputs)
        self.model.eval()
        for i, (inputs, target) in enumerate(self.external_dataloader):
            inputs = inputs.to(**self.setup)
            self.model(inputs)
            feats.append(features[feature_layer_name].detach().view(inputs.shape[0], -1).clone().cpu())
        tot_sum = torch.sum(torch.cat(feats), dim=-1) / self.num_paths
        std, mu = torch.std_mean(tot_sum)
        print(f"Feature mean is {mu.item()}, feature std is {std.item()}.")
        self.model.eval()
        self.model.cpu()
        hook.remove()

        return std, mu

    def _get_bins(self, mu=0, std=1):
        import numpy as np

        order_stats = [self._get_order_stats(r + 1, self.num_bins, mu, std) for r in range(self.num_bins)]
        diffs = [order_stats[i] - order_stats[i + 1] for i in range(len(order_stats) - 1)]

        bin_val = -np.median(diffs)
        half_dist = (self.num_bins * bin_val) / 2
        bins = (-np.linspace(mu - half_dist, mu + half_dist, self.num_bins, endpoint=False)).tolist()
        return bins, bin_val

    def _get_order_stats(self, r, n, mu=0, sigma=1):
        r"""
        r Order statistics can be computed as follows:
        E(r:n) = \mu + \Phi^{-1}\left( \frac{r-a}{n-2a+1} \sigma \right)
        where a = 0.375
        """
        from scipy.stats import norm

        return mu + norm.ppf((r - self.alpha) / (n - 2 * self.alpha + 1)) * sigma


class StackParameterServer(HonestServer):
    def __init__(
        self, model, loss, cfg_case, setup=dict(dtype=torch.float, device=torch.device("cpu")), external_dataloader=None
    ):
        """Inialize the server settings."""
        super().__init__(model, loss, cfg_case, setup, external_dataloader)
        self.secrets = dict()
        self.num_paths = cfg_case.num_paths
        self.num_bins = cfg_case.num_paths
        self.alpha = self.cfg_server.alpha
        self.bins, self.bin_val = self._get_bins()

    def prepare_model(self):
        """This server is not honest, we sneak in a hardtanh layer."""
        self.model.hardtan = torch.nn.Hardtanh(min_val=0, max_val=self.bin_val)
        return self.model

    def reconfigure_model(self, model_state):
        """Reinitialize, continue training or otherwise modify model parameters."""
        super().reconfigure_model(model_state)  # Load the benign model state first
        self.std, self.mu = self._feature_distribution()
        self.bins = (torch.tensor(self.bins) - (self.mu / self.std)).tolist()
        self._set_linear_layer(self.mu.item(), self.std.item())

    def _set_linear_layer(self, mu, sigma):
        """Setting the linear layer of the network appropriately once mean, std of features have been figured out.
        TODO: This shouldn't go just by names in the future
        """
        for (name, _) in self.model.named_parameters():
            if "path_mod" in name:
                if "weight" in name:
                    rgetattr(self.model, name).data = (1 / self.std) * torch.ones_like(rgetattr(self.model, name).data)
                    _set_pathmod_layer(rgetattr(self.model, name), self.num_paths)

                elif "bias" in name:
                    _make_linear_biases(rgetattr(self.model, name), self.bins)

            if "linear" in name or "classifier" in name:
                if "weight" in name:
                    rgetattr(self.model, name).data = torch.ones_like(rgetattr(self.model, name).data)
                elif "bias" in name:
                    rgetattr(self.model, name).data = torch.zeros_like(rgetattr(self.model, name).data)

    def _feature_distribution(self):
        """Compute the mean and std of the feature layer of the given network."""
        features = dict()

        def named_hook(name):
            def hook_fn(module, input, output):
                features[name] = input[0]

            return hook_fn

        # for name, module in reversed(list(self.model.named_modules())):
        for name, module in list(self.model.named_modules()):
            # if isinstance(module, (torch.nn.Hardtanh)):
            if isinstance(module, (torch.nn.Linear)):
                hook = module.register_forward_hook(named_hook(name))
                feature_layer_name = name
                break
        feats = []
        self.model.train()
        self.model.to(**self.setup)
        print(f"Computing feature distribution before the {feature_layer_name} layer from external data.")
        for i, (inputs, target) in enumerate(self.external_dataloader):
            inputs = inputs.to(**self.setup)
            self.model(inputs)
        self.model.eval()
        for i, (inputs, target) in enumerate(self.external_dataloader):
            inputs = inputs.to(**self.setup)
            self.model(inputs)
            feats.append(features[feature_layer_name].detach().view(inputs.shape[0], -1).clone().cpu())
        tot_sum = torch.sum(torch.cat(feats), dim=-1) / self.num_paths
        std, mu = torch.std_mean(tot_sum)
        print(f"Feature mean is {mu.item()}, feature std is {std.item()}.")
        self.model.eval()
        self.model.cpu()
        hook.remove()

        return std, mu

    def _get_bins(self, mu=0, std=1):
        import numpy as np

        order_stats = [self._get_order_stats(r + 1, self.num_bins, mu, std) for r in range(self.num_bins)]
        diffs = [order_stats[i] - order_stats[i + 1] for i in range(len(order_stats) - 1)]
        bin_val = -np.median(diffs)
        half_dist = (self.num_bins * bin_val) / 2
        bins = (-np.linspace(mu - half_dist, mu + half_dist, self.num_bins, endpoint=False)).tolist()
        return bins, bin_val

    def _get_order_stats(self, r, n, mu=0, sigma=1):
        r"""
        r Order statistics can be computed as follows:
        E(r:n) = \mu + \Phi^{-1}\left( \frac{r-a}{n-2a+1} \sigma \right)
        where a = 0.375
        """
        from scipy.stats import norm

        return mu + norm.ppf((r - self.alpha) / (n - 2 * self.alpha + 1)) * sigma
