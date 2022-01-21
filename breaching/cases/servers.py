"""Implement server code. This will be short, if the server is honest."""

import torch

from .malicious_modifications import ImprintBlock, RecoveryOptimizer, SparseImprintBlock, OneShotBlock
from .malicious_modifications.parameter_utils import introspect_model, replace_module_by_instance
from .malicious_modifications.analytic_transformer_utils import (
    compute_feature_distribution,
    partially_disable_embedding,
    set_MHA,
    set_flow_backward_layer,
    disable_mha_layer,
    partially_norm_position,
    make_imprint_layer,
)

from .aux_training import train_encoder_decoder
from .malicious_modifications.feat_decoders import generate_decoder
from .data import construct_dataloader
import logging

log = logging.getLogger(__name__)


def construct_server(model, loss_fn, cfg_case, setup, external_dataloader=None):
    """Interface function."""
    if external_dataloader is None and cfg_case.server.has_external_data:
        user_split = cfg_case.data.examples_from_split
        cfg_case.data.examples_from_split = "training" if "validation" in user_split else "validation"
        dataloader = construct_dataloader(cfg_case.data, cfg_case.impl, user_idx=None, return_full_dataset=True)
        cfg_case.data.examples_from_split = user_split
    else:
        dataloader = external_dataloader
    if cfg_case.server.name == "honest_but_curious":
        server = HonestServer(model, loss_fn, cfg_case, setup, external_dataloader=dataloader)
    elif cfg_case.server.name == "malicious_model":
        server = MaliciousModelServer(model, loss_fn, cfg_case, setup, external_dataloader=dataloader)
    elif cfg_case.server.name == "malicious_optimized_parameters":
        server = MaliciousParameterOptimizationServer(model, loss_fn, cfg_case, setup, external_dataloader=dataloader)
    elif cfg_case.server.name == "class_malicious_parameters":
        server = ClassParameterServer(model, loss_fn, cfg_case, setup, external_dataloader=dataloader)
    elif cfg_case.server.name == "malicious_transformer_parameters":
        server = MaliciousTransformerServer(model, loss_fn, cfg_case, setup, external_dataloader=dataloader)
    else:
        raise ValueError(f"Invalid server type {cfg_case.server} given.")
    return server


class HonestServer:
    """Implement an honest server protocol.

    This class loads and selects the initial model and then sends this model to the (simulated) user.
    If multiple queries are possible, then these have to loop externally over muliple rounds via .run_protocol

    Central output: self.distribute_payload -> Dict[parameters=parameters, buffers=buffers, metadata=DataHyperparams]
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

        self.num_queries = cfg_case.server.num_queries

        # Data configuration has to be shared across all parties to keep preprocessing consistent:
        self.cfg_data = cfg_case.data
        self.cfg_server = cfg_case.server

        self.external_dataloader = external_dataloader

        self.secrets = dict()  # Should be nothing in here

    def __repr__(self):
        return f"""Server (of type {self.__class__.__name__}) with settings:
    Threat model: {self.THREAT}
    Number of planned queries: {self.num_queries}
    Has external/public data: {self.cfg_server.has_external_data}

    Model:
        model specification: {str(self.model.name)}
        model state: {self.cfg_server.model_state}
        {f'public buffers: {self.cfg_server.provide_public_buffers}' if len(list(self.model.buffers())) > 0 else ''}

    Secrets: {self.secrets}
    """

    def reconfigure_model(self, model_state, query_id=0):
        """Reinitialize, continue training or otherwise modify model parameters in a benign way."""
        for name, module in self.model.named_modules():
            if model_state == "untrained":
                if hasattr(module, "reset_parameters"):
                    module.reset_parameters()
            elif model_state == "trained":
                pass  # model was already loaded as pretrained model
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

    def reset_model(self):
        pass

    def distribute_payload(self, query_id=0):
        """Server payload to send to users. These are only references to simplfiy the simulation."""

        self.reconfigure_model(self.cfg_server.model_state, query_id)
        honest_model_parameters = [p for p in self.model.parameters()]  # do not send only the generators
        if self.cfg_server.provide_public_buffers:
            honest_model_buffers = [b for b in self.model.buffers()]
        else:
            honest_model_buffers = None
        return dict(parameters=honest_model_parameters, buffers=honest_model_buffers, metadata=self.cfg_data)

    def vet_model(self, model):
        """This server is honest."""
        model = self.model  # Re-reference this everywhere
        return self.model

    def queries(self):
        return range(self.num_queries)

    def run_protocol(self, user):
        """Helper function to simulate multiple queries given a user object."""
        # Simulate a simple FL protocol
        shared_user_data = []
        payloads = []
        for query_id in self.queries():
            server_payload = self.distribute_payload(query_id)  # A malicious server can return something "fun" here
            shared_data_per_round, true_user_data = user.compute_local_updates(server_payload)
            # true_data can only be used for analysis
            payloads += [server_payload]
            shared_user_data += [shared_data_per_round]
        return shared_user_data, payloads, true_user_data


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

    def vet_model(self, model):
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
        model = modified_model
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
            log.info(f"Normalizing model throughput with gain {gain}...")
            model.to(**self.setup)
        for round in range(trials):
            if not bn_modeset:
                for name, module in model.named_modules():
                    if isinstance(module, (torch.nn.Conv2d, torch.nn.BatchNorm2d)):
                        if isinstance(module, torch.nn.Conv2d) and module.bias is None:
                            if "downsample.0" in name:
                                module.weight.data.zero_()
                                log.info(f"Reset weight in downsample {name} to zero.")
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
                        log.info(f"Current mean of layer {name} is {mu.item()}, std is {std.item()} in round {round}.")

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
        log.info(encoder)
        log.info(decoder)
        stats = train_encoder_decoder(encoder, decoder, self.external_dataloader, self.setup)
        return modified_model, decoder


class MaliciousTransformerServer(HonestServer):
    """Implement a malicious server protocol.

    This server cannot modify the 'honest' model architecture posed by an analyst,
    but may modify the model parameters freely.
    This variation is designed to leak token information from transformer models for language modelling.
    """

    THREAT = "Malicious (Parameters)"

    def __init__(
        self, model, loss, cfg_case, setup=dict(dtype=torch.float, device=torch.device("cpu")), external_dataloader=None
    ):
        """Inialize the server settings."""
        super().__init__(model, loss, cfg_case, setup, external_dataloader)
        self.secrets = dict()

    def vet_model(self, model):
        """This server is not honest, but the model architecture stays unchanged."""
        model = self.model  # Re-reference this everywhere
        return self.model

    def reconfigure_model(self, model_state, query_id=0):
        """Reinitialize, continue training or otherwise modify model parameters."""
        super().reconfigure_model(model_state)  # Load the benign model state first

        # Figure out the hidden dimension:
        # For now this is non-automated.
        if "transformer" in self.model.name:  # These are our implementations from model/language_models.py
            embedding = self.model.encoder
            pos_encoder = self.model.pos_encoder

            norm_layer = self.model.transformer_encoder.layers[0].norm1
            attention_layer = self.model.transformer_encoder.layers[0].self_attn

        first_layers, second_layers, unused_mha = [], [], []  # collecting all the imprint layers
        for i, layer in enumerate(self.model.transformer_encoder.layers):
            first_layers.append(layer.linear1)
            second_layers.append(layer.linear2)
            if i != 0:
                unused_mha.append(layer.self_attn)

        hidden_dim, embedding_dim = first_layers[0].weight.shape

        # Define "probe" function / measurement vector:
        # Probe Length is embedding_dim minus v_proportion minus skip node
        measurement_scale = self.cfg_server.param_modification.measurement_scale
        v_length = self.cfg_server.param_modification.v_length
        probe_dim = embedding_dim - v_length - 1
        weights = torch.randn(probe_dim, **self.setup)
        std, mu = torch.std_mean(weights)  # correct sample toward perfect mean and std
        probe = (weights - mu) / std / torch.as_tensor(probe_dim, **self.setup).sqrt() * measurement_scale

        measurement = torch.zeros(embedding_dim, **self.setup)
        measurement[v_length:-1] = probe

        # Disable these parts of the embedding:
        partially_disable_embedding(embedding, v_length)
        if hasattr(pos_encoder, "embedding"):
            partially_disable_embedding(self.model.pos_encoder.embedding, v_length)
            partially_norm_position(self.model.pos_encoder.embedding, v_length)

            # Maybe later:
            # self.model.pos_encoder.embedding.weight.data[:, v_length : v_length * 4] = 0
            # embedding.weight.data[:, v_length * 4 :] = 0

        # Modify the first attention mechanism in the model:
        # Set QKV modifications in-place:
        set_MHA(
            attention_layer,
            norm_layer,
            pos_encoder,
            embedding_dim,
            self.cfg_data.shape,
            sequence_token_weight=self.cfg_server.param_modification.sequence_token_weight,
            imprint_sentence_position=self.cfg_server.param_modification.imprint_sentence_position,
            softmax_skew=self.cfg_server.param_modification.softmax_skew,
            v_length=v_length,
        )

        # Take care of second linear layers, and unused mha layers first
        set_flow_backward_layer(second_layers, eps=self.cfg_server.param_modification.eps)
        disable_mha_layer(unused_mha)

        # Evaluate feature distribution of this model
        std, mu = compute_feature_distribution(self.model, self, measurement)
        # And add imprint modification to the first linear layer
        make_imprint_layer(first_layers, measurement, mu, std)
        # This should be all for the attack :>

        # We save secrets for the attack later on:
        tracker = 0
        weight_idx, bias_idx = [], []
        for idx, param in enumerate(self.model.parameters()):
            if tracker < len(first_layers) and param is first_layers[tracker].weight:
                weight_idx.append(idx)
            if tracker < len(first_layers) and param is first_layers[tracker].bias:
                bias_idx.append(idx)
                tracker += 1

        details = dict(
            weight_idx=weight_idx,
            bias_idx=bias_idx,
            data_shape=self.cfg_data.shape,
            structure="cumulative",
            v_length=v_length,
        )
        self.secrets["ImprintBlock"] = details


class MaliciousParameterOptimizationServer(HonestServer):
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

    def vet_model(self, model):
        """This server is not honest, but the model architecture stays normal."""
        model = self.model  # Re-reference this everywhere
        return self.model

    def reconfigure_model(self, model_state, query_id=0):
        """Reinitialize, continue training or otherwise modify model parameters."""
        super().reconfigure_model(model_state)  # Load the benign model state first

        # Then do fun things:
        self.parameter_algorithm.optimize_recovery()


class ClassParameterServer(HonestServer):
    """Modify parameters for the "class attack" which can pick out a subset of image data from a larger batch."""

    THREAT = "Malicious (Parameters)"

    def __init__(
        self, model, loss, cfg_case, setup=dict(dtype=torch.float, device=torch.device("cpu")), external_dataloader=None
    ):
        """Inialize the server settings."""
        super().__init__(model, loss, cfg_case, setup, external_dataloader)
        self.model_state = "custom"  # Do not mess with model parameters no matter what init is agreed upon
        self.secrets = dict()
        import copy

        self.original_model = copy.deepcopy(model)

    def reset_model(self):
        import copy

        self.model = copy.deepcopy(self.original_model)

    def vet_model(self, model):
        """This server is not honest, but the model architecture stays normal."""
        model = self.model  # Re-reference this everywhere
        return self.model

    def wrap_indices(self, indices):
        import numbers

        if isinstance(indices, numbers.Number):
            return [indices]
        else:
            return list(indices)

    def reconfigure_model(self, model_state, extra_info={}):
        super().reconfigure_model(model_state)

        if model_state == "cls_attack" and "cls_to_obtain" in extra_info:
            cls_to_obtain = extra_info["cls_to_obtain"]
            cls_to_obtain = self.wrap_indices(cls_to_obtain)

            with torch.no_grad():
                *_, l_w, l_b = self.model.parameters()

                # linear weight
                masked_param = torch.zeros_like(l_w, dtype=l_w.dtype).to(l_w.device)
                masked_param[cls_to_obtain] = torch.ones_like(l_w[cls_to_obtain], dtype=l_w.dtype).to(l_w.device) * 0.5
                l_w.copy_(masked_param.to(l_w.device))

                # linear bias
                masked_param = torch.ones_like(l_b, dtype=l_b.dtype).to(l_b.device) * 1000
                masked_param[cls_to_obtain] = l_b[cls_to_obtain]
                l_b.copy_(masked_param.to(l_b.device))

        if model_state == "fishing_attack" and "cls_to_obtain" in extra_info:
            cls_to_obtain = extra_info["cls_to_obtain"]
            b_mv = extra_info["b_mv"] if "b_mv" in extra_info else 0
            b_mv_non = extra_info["b_mv_non"] if "b_mv_non" in extra_info else 0
            multiplier = extra_info["multiplier"] if "multiplier" in extra_info else 1
            cls_to_obtain = self.wrap_indices(cls_to_obtain)

            with torch.no_grad():
                *_, l_w, l_b = self.model.parameters()

                # linear weight
                masked_param = torch.zeros_like(l_w, dtype=l_w.dtype).to(l_w.device)
                masked_param[cls_to_obtain] = l_w[cls_to_obtain]
                l_w.copy_(masked_param.to(l_w.device))

                # linear bias
                masked_param = torch.ones_like(l_b, dtype=l_b.dtype).to(l_b.device)
                masked_param[cls_to_obtain] = l_b[cls_to_obtain] + b_mv
                l_b.copy_(masked_param.to(l_b.device))

                *_, l_w, l_b = self.model.parameters()
                *_, l_w_o, l_b_o = self.original_model.parameters()
                cls_to_obtain = int(extra_info["cls_to_obtain"])
                l_w[:cls_to_obtain] = l_w_o[:cls_to_obtain]
                l_w[cls_to_obtain + 1 :] = l_w_o[cls_to_obtain + 1 :]
                l_b[:cls_to_obtain] = l_b_o[:cls_to_obtain] + b_mv_non
                l_b[cls_to_obtain + 1 :] = l_b_o[cls_to_obtain + 1 :] + b_mv_non

                l_w *= multiplier
                l_b *= multiplier

        if model_state == "feature_attack" and "cls_to_obtain" in extra_info and "feat_to_obtain" in extra_info:
            cls_to_obtain = extra_info["cls_to_obtain"]
            feat_to_obtain = extra_info["feat_to_obtain"]
            cls_to_obtain = self.wrap_indices(cls_to_obtain)
            feat_to_obtain = self.wrap_indices(feat_to_obtain)

            with torch.no_grad():
                *_, bn_w, bn_b, l_w, l_b = self.model.parameters()

                # # batch norm part is only for recovering one img from feature attack
                # # batch norm weight
                # masked_param = torch.zeros_like(bn_w, dtype=bn_w.dtype)
                # masked_param[feat_to_obtain] = bn_w[feat_to_obtain]
                # bn_w.copy_(masked_param.to(**self.setup))

                # # batch norm bias
                # masked_param = torch.zeros_like(bn_b, dtype=bn_b.dtype)
                # masked_param[feat_to_obtain] = bn_b[feat_to_obtain]
                # bn_b.copy_(masked_param.to(**self.setup))

                if "feat_value" not in extra_info:
                    # just turn off other features
                    masked_param = torch.zeros_like(l_w, dtype=l_w.dtype).to(l_w.device)
                    masked_param[cls_to_obtain, feat_to_obtain] = torch.ones_like(
                        l_w[cls_to_obtain, feat_to_obtain], dtype=l_w.dtype
                    )
                    l_w.copy_(masked_param.to(l_w.device))
                else:
                    # do gradient amplification
                    multiplier = extra_info["multiplier"]
                    extra_b = extra_info["extra_b"] if "extra_b" in extra_info else 0
                    non_target_logit = extra_info["non_target_logit"] if "non_target_logit" in extra_info else 0
                    db_flip = extra_info["db_flip"] if "db_flip" in extra_info else 1

                    masked_param = torch.zeros_like(l_w, dtype=l_w.dtype).to(l_w.device)
                    masked_param[cls_to_obtain, feat_to_obtain] = (
                        torch.ones_like(l_w[cls_to_obtain, feat_to_obtain], dtype=l_w.dtype).to(l_w.device)
                        * multiplier
                        * db_flip
                    )
                    l_w.copy_(masked_param.to(l_w.device))

                    masked_param = torch.zeros_like(l_b, dtype=l_b.dtype).to(l_b.device) + non_target_logit
                    masked_param[cls_to_obtain] = (
                        torch.zeros_like(l_b[cls_to_obtain], dtype=l_b.dtype).to(l_b.device)
                        - extra_info["feat_value"] * multiplier * db_flip
                        + extra_b
                    )
                    l_b.copy_(masked_param.to(l_b.device))

        if model_state == "db_attack" and "cls_to_obtain" in extra_info:
            cls_to_obtain = extra_info["cls_to_obtain"]
            cls_to_obtain = self.wrap_indices(cls_to_obtain)
            db_multiplier = extra_info["db_multiplier"]
            multiplier = extra_info["multiplier"]
            db_flip = extra_info["db_flip"]

            with torch.no_grad():
                *_, bn_w, bn_b, l_w, l_b = self.model.parameters()

                # batch norm weight
                masked_param = bn_w
                bn_w.copy_(masked_param.to(bn_w.device))

                # batch norm bias
                masked_param = bn_b + l_w[cls_to_obtain[0]] * db_multiplier
                bn_b.copy_(masked_param.to(bn_b.device))

                # linear weight
                masked_param = torch.zeros_like(l_w, dtype=l_w.dtype).to(l_w.device)
                masked_param[cls_to_obtain] = l_w[cls_to_obtain] * multiplier * db_flip
                l_w.copy_(masked_param.to(l_w.device))

                # linear bias
                masked_param = torch.zeros_like(l_b, dtype=l_b.dtype).to(l_b.device)
                masked_param[cls_to_obtain] = l_b[cls_to_obtain] * db_flip
                l_b.copy_(masked_param.to(l_b.device))

    def one_shot_binary_attack(self, user, extra_info):
        cls_to_obtain = extra_info["cls_to_obtain"]
        feat_to_obtain = extra_info["feat_to_obtain"]
        feat_value = extra_info["feat_value"]
        num_target_data = extra_info["num_target_data"]
        num_data_points = extra_info["num_data_points"]
        self.num_target_data = num_target_data
        self.all_feat_value = []

        extra_info["multiplier"] = 1000
        while True:
            self.all_feat_value.append(feat_value)
            extra_info["feat_value"] = feat_value
            self.reset_model()
            self.reconfigure_model("cls_attack", extra_info=extra_info)
            self.reconfigure_model("feature_attack", extra_info=extra_info)
            server_payload = self.distribute_payload()
            shared_data, _ = user.compute_local_updates(server_payload)
            avg_feature = torch.flatten(self.reconstruct_feature(shared_data, cls_to_obtain))
            feat_value = float(avg_feature[feat_to_obtain])

            if self.check_with_tolerance(feat_value, self.all_feat_value):
                curr_grad = list(shared_data["gradients"])
                break

        curr_grad[-1] = curr_grad[-1] * num_data_points
        curr_grad[:-1] = [grad_ii * num_data_points / extra_info["multiplier"] for grad_ii in curr_grad[:-1]]
        self.all_feat_value.sort()

        return [curr_grad]

    def binary_attack(self, user, extra_info):
        import numpy as np

        feat_value = extra_info["feat_value"]
        num_target_data = extra_info["num_target_data"]
        num_data_points = extra_info["num_data_points"]
        self.num_target_data = num_target_data
        self.all_feat_value = []

        # get filter feature points first
        self.all_feat_value = []
        self.feat_grad = []
        self.visited = []
        self.counter = 0
        extra_info["multiplier"] = 1000
        retval = self.binary_attack_helper(user, extra_info, [feat_value])
        if retval == 0:  # Stop early after too many attempts in binary search:
            return None
        self.all_feat_value = np.array(self.all_feat_value)
        sorted_inds = np.argsort(self.all_feat_value)
        sorted_feat_grad = []
        self.all_feat_value = self.all_feat_value[sorted_inds]
        for i in sorted_inds:
            sorted_feat_grad.append(self.feat_grad[i])
        self.feat_grad = sorted_feat_grad

        # recover gradients
        import copy

        curr_grad = copy.deepcopy(list(self.feat_grad[0]))
        curr_grad[-1] = curr_grad[-1] * num_data_points
        curr_grad[:-1] = [grad_ii * num_data_points / extra_info["multiplier"] for grad_ii in curr_grad[:-1]]
        prev_grad = copy.deepcopy(curr_grad)
        single_gradients = [curr_grad]
        for i in range(1, len(self.all_feat_value)):
            curr_grad = copy.deepcopy(list(self.feat_grad[i]))
            curr_grad[-1] = curr_grad[-1] * num_data_points
            curr_grad[:-1] = [grad_ii * num_data_points / extra_info["multiplier"] for grad_ii in curr_grad[:-1]]
            grad_i = [grad_ii - grad_jj for grad_ii, grad_jj in zip(curr_grad, prev_grad)]
            single_gradients.append(grad_i)
            prev_grad = copy.deepcopy(curr_grad)

        return single_gradients

    def binary_attack_helper(self, user, extra_info, feat_01_values):

        if len(self.all_feat_value) >= self.num_target_data:
            return 1
        # if self.counter >= math.log2(self.num_target_data) * self.num_target_data:
        if self.counter >= self.num_target_data ** 2:
            log.info(f"Too many attempts ({self.counter}) on this feature!")
            return 0

        # print('new level')

        new_feat_01_values = []

        # get left and right mid point
        cls_to_obtain = extra_info["cls_to_obtain"]
        feat_to_obtain = extra_info["feat_to_obtain"]

        for feat_01_value in feat_01_values:
            extra_info["feat_value"] = feat_01_value
            extra_info["multiplier"] = 1000
            self.reset_model()
            self.reconfigure_model("cls_attack", extra_info=extra_info)
            self.reconfigure_model("feature_attack", extra_info=extra_info)
            server_payload = self.distribute_payload()
            shared_data, _ = user.compute_local_updates(server_payload)
            feat_0 = torch.flatten(self.reconstruct_feature(shared_data, cls_to_obtain))
            feat_0_value = float(feat_0[feat_to_obtain])  # the middle includes left hand side
            feat_1_value = 2 * feat_01_value - feat_0_value
            self.counter += 1
            # print(feat_01_value, feat_0_value, feat_1_value)

            feat_candidates = [feat_0_value]

            for feat_cand in feat_candidates:
                if self.check_with_tolerance(feat_cand, self.visited):
                    pass
                elif not self.check_with_tolerance(feat_cand, self.visited):
                    if not self.check_with_tolerance(feat_01_value, self.all_feat_value):
                        self.all_feat_value.append(feat_01_value)
                        self.feat_grad.append(list(shared_data["gradients"]))
                    new_feat_01_values.append(feat_cand)
                    self.visited.append(feat_cand)

                if len(self.all_feat_value) >= self.num_target_data:
                    return
                # self.counter >= math.log2(self.num_target_data) * self.num_target_data:
                if self.counter >= self.num_target_data ** 2:
                    log.info(f"Too many attempts ({self.counter}) on this feature!")
                    return 0

            feat_candidates = [feat_1_value, (feat_01_value + feat_1_value) / 2, (feat_01_value + feat_0_value) / 2]

            for feat_cand in feat_candidates:
                if not self.check_with_tolerance(feat_cand, self.visited):
                    new_feat_01_values.append(feat_cand)

        return self.binary_attack_helper(user, extra_info, new_feat_01_values)

    def check_with_tolerance(self, value, list, threshold=0.05):
        for i in list:
            if abs(value - i) < threshold:
                return True

        return False

    def order_gradients(self, recovered_single_gradients, gt_single_gradients):
        from scipy.optimize import linear_sum_assignment

        single_gradients = []
        num_data = len(gt_single_gradients)

        for grad_i in recovered_single_gradients:
            single_gradients.append(torch.cat([torch.flatten(i) for i in grad_i]))

        similarity_matrix = torch.zeros(num_data, num_data, **self.setup)
        for idx, x in enumerate(single_gradients):
            for idy, y in enumerate(gt_single_gradients):
                similarity_matrix[idy, idx] = torch.nn.CosineSimilarity(dim=0)(y, x).detach()

        try:
            _, rec_assignment = linear_sum_assignment(similarity_matrix.cpu().numpy(), maximize=True)
        except ValueError:
            log.info(f"ValueError from similarity matrix {similarity_matrix.cpu().numpy()}")
            log.info("Returning trivial order...")
            rec_assignment = list(range(num_data))

        return [recovered_single_gradients[i] for i in rec_assignment]

    def reconstruct_feature(self, shared_data, cls_to_obtain):
        if type(shared_data) is not list:
            shared_grad = shared_data["gradients"]
        else:
            shared_grad = shared_data

        weights = shared_grad[-2]
        bias = shared_grad[-1]
        grads_fc_debiased = weights / bias[:, None]

        if bias[cls_to_obtain] != 0:
            return grads_fc_debiased[cls_to_obtain]
        else:
            return torch.zeros_like(grads_fc_debiased[0])

    def cal_single_gradients(self, attacker, true_user_data):
        true_data = true_user_data["data"]
        num_data = len(true_data)
        labels = true_user_data["labels"]
        model = self.model.to(**self.setup)

        single_gradients = []
        single_losses = []

        for ii in range(num_data):
            cand_ii = true_data[ii : (ii + 1)]
            label_ii = labels[ii : (ii + 1)]
            model.zero_grad()
            spoofed_loss_ii = attacker.loss_fn(model(cand_ii), label_ii)
            attacker.objective.loss_fn = attacker.loss_fn
            gradient_ii, _ = attacker.objective._grad_fn_single_step(model, cand_ii, label_ii)
            gradient_ii = [g_ii.reshape(-1) for g_ii in gradient_ii]
            gradient_ii = torch.cat(gradient_ii)
            single_gradients.append(gradient_ii)
            single_losses.append(spoofed_loss_ii)

        return single_gradients, single_losses

    def print_gradients_norm(self, single_gradients, single_losses, which_to_recover=-1, return_results=False):
        grad_norm = []
        losses = []

        if not return_results:
            print("grad norm         loss")

        for i, gradient_ii in enumerate(single_gradients):
            if not return_results:
                if i == which_to_recover:
                    print(float(torch.norm(gradient_ii)), float(single_losses[i]), "   target")
                else:
                    print(float(torch.norm(gradient_ii)), float(single_losses[i]))

            grad_norm.append(float(torch.norm(gradient_ii)))
            losses.append(float(single_losses[i]))

        if return_results:
            return torch.stack(grad_norm), torch.stack(losses)

    def random_transoformation(self, img):
        import torchvision.transforms as transforms

        transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(img.shape[-2:], scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(p=1),
                # transforms.RandomVerticalFlip(p=1),
                transforms.GaussianBlur(3),
            ]
        )

        return transform(img)

    class pseudo_dataset:
        def __init__(self, outer_instance, target_img, dataset_len=2000, num_classes=1000, target_cls=0, prob=4):
            self.outer_instance = outer_instance
            self.target_img = target_img
            self.img_size = tuple(target_img.shape[-2:])
            self.dataset_len = dataset_len
            self.num_classes = num_classes
            self.target_cls = target_cls
            self.prob = prob

        def __len__(self):
            return self.dataset_len

        def __getitem__(self, idx):
            prob_label = torch.zeros((self.num_classes))

            if idx % self.prob == 0:
                target_img = self.target_img
                prob_label[self.target_cls] = -10
            else:
                target_img = torch.rand(self.target_img.shape) * 2 - 1
                prob_label[self.target_cls] = 0
                # prob_label += 1
                # prob_label[self.target_cls + 1] = 1

            return target_img.to(**self.outer_instance.setup), prob_label.to(**self.outer_instance.setup)

    def fishing_attack_retrain(self, target_img, target_cls=0, dataset_len=2000):
        import torch.optim as optim
        from tqdm import tqdm
        from torch import nn
        from torch.utils.data import DataLoader

        # only retraining on the last linear layer
        self.model = self.model.to(**self.setup)
        for param in self.model.parameters():
            param.requires_grad = False

        *_, l_w, l_b = self.model.parameters()
        l_w.requires_grad = True
        l_b.requires_grad = True

        # prepare for training
        dataset = self.pseudo_dataset(self, target_img, target_cls=target_cls, dataset_len=dataset_len)
        dataloader = DataLoader(dataset, batch_size=8, num_workers=0)
        # criterion = nn.CrossEntropyLoss()
        criterion = nn.MSELoss()
        optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)

        with tqdm(dataloader, unit="batch") as tepoch:
            running_loss = 0.0

            for i, (inputs, labels) in enumerate(tepoch):
                inputs, labels = inputs.to(**self.setup), labels.to(**self.setup)

                optimizer.zero_grad()

                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                tepoch.set_postfix(loss=running_loss / (i + 1))

        for param in self.model.parameters():
            param.requires_grad = True

    def estimate_feat(self, cfg, extra_info, num_to_est=900):
        import breaching
        from tqdm import tqdm
        import numpy as np

        est_features = []
        sample_sizes = []
        cls_to_obtain = extra_info["cls_to_obtain"]
        model = extra_info["model"]
        loss_fn = extra_info["loss_fn"]
        setup = extra_info["setup"]
        self.reset_model()
        self.reconfigure_model("cls_attack", extra_info=extra_info)

        for i in tqdm(range(num_to_est // cfg.case.user.num_data_points)):
            cfg.case.user.user_idx = i
            user = breaching.cases.construct_user(model, loss_fn, cfg.case, setup)
            server_payload = self.distribute_payload()
            shared_data, _ = user.compute_local_updates(server_payload)
            num_target = int(torch.count_nonzero((shared_data["metadata"]["labels"] == int(cls_to_obtain)).to(int)))
            if num_target != 0:
                est_features.append(
                    torch.flatten(self.reconstruct_feature(shared_data, cls_to_obtain)).detach().cpu().numpy()
                )
                sample_sizes.append(num_target)

        est_features = np.vstack(est_features)
        sample_sizes = np.array(sample_sizes)

        return est_features.T, sample_sizes

    def estimate_gt_stats(self, est_features, sample_sizes, indx=0):
        import numpy as np

        aggreg_data = []
        est_feature = est_features[indx]

        for i in range(len(est_feature)):
            feat_i = est_feature[i]
            size_i = sample_sizes[i]
            aggreg_data.append(feat_i * (size_i ** (1 / 2)))

        return np.mean(est_feature), np.std(aggreg_data)

    def find_best_feat(self, est_features, sample_sizes, method="kstest"):
        import numpy as np
        from scipy import stats

        if "kstest" in method:
            p_values = []
            for i in range(len(est_features)):
                tmp_series = est_features[i]
                tmp_series = (tmp_series - np.mean(tmp_series)) / np.std(tmp_series)
                p_values.append((np.std(tmp_series), stats.kstest(tmp_series, "norm").pvalue)[1])
        elif "most-spread" in method or "most-high-mean" in method:
            means = []
            stds = []
            for i in range(len(est_features)):
                mu, sigma = self.estimate_gt_stats(est_features, sample_sizes, indx=1)
                means.append(mu)
                stds.append(sigma)

            if "most-spread" in method:
                return np.argmax(stds)
            else:
                return np.argmax(means)
        else:
            raise ValueError(f"Method {method} not implemented.")

        return np.argmax(p_values)

    def recover_patch(self, x):
        # retile img
        p_num_2, p_size_2 = x.shape[1:]
        p_num = int(p_num_2 ** (1 / 2))
        p_size = int(p_size_2 ** (1 / 2))
        img_size = int((p_num_2 * p_size_2) ** (1 / 2))
        x = x.reshape((3, p_num, p_num, p_size, p_size))
        new_x = torch.zeros_like(x).reshape((3, img_size, img_size))

        for i in range(p_num):
            for j in range(p_num):
                new_x[:, i * p_size : (i + 1) * p_size, j * p_size : (j + 1) * p_size] = x[:, i, j, :]

        return new_x

    def closed_form_april(self, shared_data):
        # recover patch embeddings first, (APRIL paper)
        qkv_w = self.model.model.blocks[0].attn.qkv.weight.detach().double().cpu()
        q_w, k_w, v_w = qkv_w.reshape(3, -1, qkv_w.shape[-1]).unbind()
        qkv_g = shared_data["gradients"][4].double().cpu()
        q_g, k_g, v_g = qkv_g.reshape(3, -1, qkv_g.shape[-1]).unbind()
        A = shared_data["gradients"][1].detach().squeeze().double().cpu()
        pos_embed = self.model.model.pos_embed.detach().squeeze().double().cpu()

        b = q_w.T @ q_g + k_w.T @ k_g + v_w.T @ v_g
        z = torch.linalg.lstsq(A.T, b, driver="gelsd").solution
        z -= pos_embed
        z = z[1:]

        # recover img
        em_w = self.model.model.patch_embed.proj.weight.detach().double().cpu()
        em_w = em_w.reshape((em_w.shape[0], -1))
        em_b = self.model.model.patch_embed.proj.bias.detach().double().cpu()

        x = z - em_b
        x = torch.linalg.lstsq(em_w, x.T, driver="gelsd").solution
        x = x.reshape((3, -1, x.shape[-1]))
        x = x.transpose(1, 2)
        x = self.recover_patch(x)

        return x.float().to(**self.setup)

    class ModifiedBlock(torch.nn.Module):
        def __init__(self, old_Block):
            super().__init__()
            self.attn = old_Block.attn
            self.drop_path = old_Block.drop_path
            self.norm2 = old_Block.norm2
            self.mlp = old_Block.mlp

        def forward(self, x):
            x = self.attn(x)
            x = self.drop_path(self.mlp((self.norm2(x))))
            return x
