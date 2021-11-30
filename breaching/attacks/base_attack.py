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

    def __repr__(self):
        raise NotImplementedError()


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
            labels = self._recover_label_information(shared_data, rec_models)
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

            new_model = copy.deepcopy(self.model_template)
            new_model.to(**self.setup, memory_format=self.memory_format)

            # Load parameters
            parameters = payload['parameters']
            if user_buffers is not None and idx < len(user_buffers):
                # User sends buffers. These should be used!
                buffers = user_buffers[idx]
                new_model.eval()
            elif payload['buffers'] is not None:
                # The server has public buffers in any case
                buffers = payload['buffers']
                new_model.eval()
            else:
                # The user sends no buffers and there are no public bufers
                # (i.e. the user in in training mode and does not send updates)
                new_model.train()
                for module in new_model.modules():
                    if hasattr(module, 'track_running_stats'):
                        module.track_running_stats = False
                        module.reset_parameters()
                buffers = []

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

    def _recover_label_information(self, user_data, rec_models):
        """Recover label information.

        This method runs under the assumption that the last two entries in the gradient vector
        correpond to the weight and bias of the last layer (mapping to num_classes).
        For non-classification tasks this has to be modified.

        The behavior with respect to multiple queries is work in progress and subject of debate.
        """
        num_data_points = user_data['num_data_points']
        num_classes = user_data['gradients'][0][-1].shape[0]
        num_queries = len(user_data['gradients'])

        # In the simplest case, the label can just be inferred from the last layer
        if self.cfg.label_strategy == 'iDLG':
            # This was popularized in "iDLG" by Zhao et al., 2020
            # assert num_data_points == 1
            label_list = []
            for query_id, shared_grad in enumerate(user_data['gradients']):
                last_weight_min = torch.argmin(torch.sum(shared_grad[-2], dim=-1), dim=-1)
                label_list += [last_weight_min.detach()]
            labels = torch.stack(label_list).unique()
        elif self.cfg.label_strategy == 'analytic':
            # Analytic recovery simply works as long as all labels are unique.
            label_list = []
            for query_id, shared_grad in enumerate(user_data['gradients']):
                valid_classes = (shared_grad[-1] < 0).nonzero()
                label_list += [valid_classes]
            labels = torch.stack(label_list).unique()[:num_data_points]
        elif self.cfg.label_strategy == 'yin':
            # As seen in Yin et al. 2021, "See Through Gradients: Image Batch Recovery via GradInversion"
            # This additionally assumes that there is a nonlinearity with positive output (like ReLU) in front of the
            # last classification layer.
            # This scheme also works best if all labels are unique
            # Otherwise this is an extension of iDLG to multiple labels:
            total_min_vals = 0
            for query_id, shared_grad in enumerate(user_data['gradients']):
                total_min_vals += shared_grad[-2].min(dim=-1)[0]
            labels = total_min_vals.argsort()[:num_data_points]

        elif 'wainakh' in self.cfg.label_strategy:

            if self.cfg.label_strategy == 'wainakh-simple':
                # As seen in Weinakh et al., "User Label Leakage from Gradients in Federated Learning"
                m_impact = 0
                for query_id, shared_grad in enumerate(user_data['gradients']):
                    g_i = shared_grad[-2].sum(dim=1)
                    m_query = torch.where(g_i < 0, g_i, torch.zeros_like(g_i)).sum() * (1 + 1 / num_classes) / num_data_points
                    s_offset = 0
                    m_impact += m_query / num_queries
            elif self.cfg.label_strategy == 'wainakh-whitebox':
                # Augment previous strategy with measurements of label impact for dummy data.
                m_impact = 0
                s_offset = torch.zeros(num_classes, **self.setup)

                print('Starting a white-box search for optimal labels. This will take some time.')
                for query_id, (shared_grad, model) in enumerate(zip(user_data['gradients'], rec_models)):
                    # Estimate m:
                    weight_params = (list(rec_models[0].parameters())[-2],)
                    for class_idx in range(num_classes):
                        fake_data = torch.randn([num_data_points, *self.data_shape], **self.setup)
                        fake_labels = torch.as_tensor([class_idx] * num_data_points, **self.setup)
                        with torch.autocast(self.setup['device'].type, enabled=self.cfg.impl.mixed_precision):
                            loss = self.loss_fn(model(fake_data), fake_labels)
                        W_cls, = torch.autograd.grad(loss, weight_params)
                        g_i = W_cls.sum(dim=1)
                        m_impact += g_i.sum() * (1 + 1 / num_classes) / num_data_points / num_classes / num_queries

                    # Estimate s:
                    T = num_classes - 1
                    for class_idx in range(num_classes):
                        fake_data = torch.randn([T, *self.data_shape], **self.setup)
                        fake_labels = torch.arange(num_classes, **self.setup)
                        fake_labels = fake_labels[fake_labels != class_idx]
                        with torch.autocast(self.setup['device'].type, enabled=self.cfg.impl.mixed_precision):
                            loss = self.loss_fn(model(fake_data), fake_labels)
                        W_cls, = torch.autograd.grad(loss, (weight_params[0][class_idx],))
                        s_offset[class_idx] += W_cls.sum() / T / num_queries

            else:
                raise ValueError(f'Invalid Wainakh strategy {self.cfg.label_strategy}.')

            # After determining impact and offset, run the actual recovery algorithm
            label_list = []
            g_per_query = [shared_grad[-2].sum(dim=1) for shared_grad in user_data['gradients']]
            g_i = torch.stack(g_per_query).mean(dim=0)
            # Stage 1:
            for idx in range(num_classes):
                if g_i[idx] < 0:
                    label_list.append(torch.as_tensor(idx, device=self.setup['device']))
                    g_i[idx] -= m_impact
            # Stage 2:
            g_i = g_i - s_offset
            while len(label_list) < num_data_points:
                selected_idx = g_i.argmin()
                label_list.append(torch.as_tensor(selected_idx, device=self.setup['device']))
                g_i[idx] -= m_impact
            # Finalize labels:
            labels = torch.stack(label_list)

        elif self.cfg.label_strategy == 'bias-corrected':  # WIP
            # This is slightly modified analytic label recovery in the style of Wainakh
            bias_per_query = [shared_grad[-1] for shared_grad in user_data['gradients']]
            label_list = []
            # Stage 1
            average_bias = torch.stack(bias_per_query).mean(dim=0)
            valid_classes = (average_bias < 0).nonzero()
            label_list += [*valid_classes.squeeze()]
            m_impact = average_bias_correct_label = average_bias[valid_classes].sum() / num_data_points

            average_bias[valid_classes] = average_bias[valid_classes] - m_impact
            # Stage 2
            while len(label_list) < num_data_points:
                selected_idx = average_bias.argmin()
                label_list.append(selected_idx)
                average_bias[selected_idx] -= m_impact
            labels = torch.stack(label_list)

        elif self.cfg.label_strategy == 'random':
            # A random baseline
            labels = torch.randint(0, num_classes, (num_data_points,), device=self.setup['device'])
        elif self.cfg.label_strategy == 'exhaustive':
            # Exhaustive search is possible in principle
            combinations = num_classes ** num_data_points
            raise ValueError(f'Exhaustive label searching not implemented. Nothing stops you though from running your'
                             f'attack algorithm for any possible combination of labels, except computational effort.'
                             f'In the given setting, a naive exhaustive strategy would attack {combinations} label vectors.')
            # Although this is arguably a worst-case estimate, you might be able to get "close enough" to the actual
            # label vector in much fewer queries, depending on which notion of close-enough makes sense for a given attack.
        else:
            raise ValueError(f'Invalid label recovery strategy {self.cfg.label_strategy} given.')

        # Pad with random labels if too few were produced:
        if len(labels) < num_data_points:
            labels = torch.cat([labels, torch.randint(0, num_classes, (num_data_points - len(labels),), device=self.setup['device'])])

        # Always sort, order does not matter here:
        labels = labels.sort()[0]
        print(f'Recovered labels {labels.tolist()} through strategy {self.cfg.label_strategy}.')
        return labels
