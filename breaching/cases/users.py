"""Implement user code."""

import torch
import copy

from .data import construct_dataloader


def construct_user(model, loss_fn, cfg_case, setup):
    """Interface function."""
    if cfg_case.user.user_type == "local_gradient":
        dataloader = construct_dataloader(cfg_case.data, cfg_case.impl, user_idx=cfg.case.user.user_idx)
        # The user will deepcopy this model template to have their own
        user = UserSingleStep(model, loss, dataloader, setup, cfg_case.user)
    elif cfg_case.user.user_type == "local_update":
        dataloader = construct_dataloader(cfg_case.data, cfg_case.impl, user_idx=cfg.case.user.user_idx)
        user = UserMultiStep(model, loss, dataloader, setup, cfg_case.user)
    elif cfg_case.user.user_type == "multiuser_aggregate":
        dataloaders = []
        for idx in range(*cfg.case.user.user_range):
            dataloaders += [construct_dataloader(cfg_case.data, cfg_case.impl, user_idx=idx)]
        user = MultiUserAggregate(model, loss, dataloaders, setup, cfg_case.user)


class UserSingleStep(torch.nn.Module):
    """A user who computes a single local update step."""

    def __init__(self, model, loss, dataloader, setup, cfg_user):
        """Initialize from cfg_user dict which contains atleast all keys in the matching .yaml :>"""
        super().__init__()
        self.num_data_points = cfg_user.num_data_points

        self.provide_labels = cfg_user.provide_labels
        self.provide_num_data_points = cfg_user.provide_num_data_points
        self.provide_buffers = cfg_user.provide_buffers

        self.user_idx = cfg_user.user_idx
        self.setup = setup

        self.model = copy.deepcopy(model)
        self.model.to(**setup)

        self.defense_repr = []
        self._initialize_local_privacy_measures(cfg_user.local_diff_privacy)

        self.dataloader = dataloader
        self.loss = copy.deepcopy(loss)  # Just in case the loss contains state

    def __repr__(self):
        n = "\n"
        return f"""User (of type {self.__class__.__name__}) with settings:
    Number of data points: {self.num_data_points}

    Threat model:
    User provides labels: {self.provide_labels}
    User provides buffers: {self.provide_buffers}
    User provides number of data points: {self.provide_num_data_points}

    Data:
    Dataset: {self.dataloader.dataset.__class__.__name__}
    user: {self.user_idx}
    {n.join(self.defense_repr)}
        """

    def _initialize_local_privacy_measures(self, local_diff_privacy):
        """Initialize generators for noise in either gradient or input."""
        if local_diff_privacy["gradient_noise"] > 0.0:
            loc = torch.as_tensor(0.0, **self.setup)
            scale = torch.as_tensor(local_diff_privacy["gradient_noise"], **self.setup)
            if local_diff_privacy["distribution"] == "gaussian":
                self.generator = torch.distributions.normal.Normal(loc=loc, scale=scale)
            elif local_diff_privacy["distribution"] == "laplacian":
                self.generator = torch.distributions.laplace.Laplace(loc=loc, scale=scale)
            else:
                raise ValueError(f'Invalid distribution {local_diff_privacy["distribution"]} given.')
            self.defense_repr.append(
                f'Defense: Local {local_diff_privacy["distribution"]} gradient noise with strength {scale.item()}.'
            )
        else:
            self.generator = None
        if local_diff_privacy["input_noise"] > 0.0:
            loc = torch.as_tensor(0.0, **self.setup)
            scale = torch.as_tensor(local_diff_privacy["input_noise"], **self.setup)
            if local_diff_privacy["distribution"] == "gaussian":
                self.generator_input = torch.distributions.normal.Normal(loc=loc, scale=scale)
            elif local_diff_privacy["distribution"] == "laplacian":
                self.generator_input = torch.distributions.laplace.Laplace(loc=loc, scale=scale)
            else:
                raise ValueError(f'Invalid distribution {local_diff_privacy["distribution"]} given.')
            self.defense_repr.append(
                f'Defense: Local {local_diff_privacy["distribution"]} input noise with strength {scale.item()}.'
            )
        else:
            self.generator_input = None
        self.clip_value = local_diff_privacy.get("per_example_clipping", 0.0)
        if self.clip_value > 0:
            self.defense_repr.append(f"Defense: Gradient clipping to maximum of {self.clip_value}.")

    def compute_local_updates(self, server_payload):
        """Compute local updates to the given model based on server payload.

        Batchnorm behavior:
        If public buffers are sent by the server, then the user will be set into evaluation mode
        Otherwise the user is in training mode and sends back buffer based on .provide_buffers.

        Shared labels are canonically sorted for simplicity."""

        data, labels = self._load_data()
        # Compute local updates
        shared_grads = []
        shared_buffers = []

        payload = server_payload["queries"][query]
        parameters = payload["parameters"]
        buffers = payload["buffers"]

        with torch.no_grad():
            for param, server_state in zip(self.model.parameters(), parameters):
                param.copy_(server_state.to(**self.setup))
            if buffers is not None:
                for buffer, server_state in zip(self.model.buffers(), buffers):
                    buffer.copy_(server_state.to(**self.setup))
                self.model.eval()
            else:
                for module in self.model.modules():
                    if hasattr(module, "momentum"):
                        module.momentum = None  # Force recovery without division
                self.model.train()

        def _compute_batch_gradient(data, labels):
            data_input = data + self.generator_input.sample(data.shape) if self.generator_input is not None else data
            outputs = self.model(data_input)
            loss = self.loss(outputs, labels)
            return torch.autograd.grad(loss, self.model.parameters())

        if self.clip_value > 0:  # Compute per-example gradients and clip them in this case
            shared_grads = [torch.zeros_like(p) for p in self.model.parameters()]
            for data_point, data_label in zip(data, labels):
                per_example_grads = _compute_batch_gradient(data_point[None, ...], data_label[None, ...])
                self._clip_list_of_grad_(per_example_grads)
                torch._foreach_add_(shared_grads, per_example_grads)
            torch._foreach_div_(shared_grads, len(data))
        else:
            # Compute the forward pass
            shared_grads = _compute_batch_gradient(data, labels)
        self._apply_differential_noise(shared_grads)

        if buffers is not None:
            shared_buffers = None
        else:
            shared_buffers = [b.clone().detach() for b in self.model.buffers()]

        metadata = dict(
            num_data_points=self.num_data_points if self.provide_num_data_points else None,
            labels=labels.sort()[0] if self.provide_labels else None,
            local_hyperparams=None,
        )
        shared_data = dict(
            gradients=shared_grads, buffers=shared_buffers if self.provide_buffers else None, metadata=metadata
        )
        true_user_data = dict(data=data, labels=labels, buffers=shared_buffers)

        return shared_data, true_user_data

    def _clip_list_of_grad_(self, grads):
        """Apply differential privacy component per-example clipping."""
        grad_norm = torch.norm(torch.stack([torch.norm(g, 2) for g in grads]), 2)
        if grad_norm > self.clip_value:
            [g.mul_(self.clip_value / (grad_norm + 1e-6)) for g in grads]

    def _apply_differential_noise(self, grads):
        """Apply differential privacy component gradient noise."""
        if self.generator is not None:
            for grad in grads:
                grad += self.generator.sample(grad.shape)

    def _load_data(self):
        """Generate data from dataloader, truncated by self.num_data_points"""
        # Select data
        data_blocks = []
        label_blocks = []
        num_samples = 0

        for idx, (data, labels) in enumerate(self.dataloader):
            data_blocks += [data]
            label_blocks += [labels]
            num_samples += labels.shape[0]
            if num_samples > self.num_data_points:
                break

        data = torch.cat(data_blocks, dim=0)[: self.num_data_points].to(**self.setup)
        labels = torch.cat(label_blocks, dim=0)[: self.num_data_points].to(device=self.setup["device"])
        return data, labels

    def plot(self, user_data, scale=False, print_labels=False):
        """Plot user data to output. Probably best called from a jupyter notebook."""
        import matplotlib.pyplot as plt  # lazily import this here

        dm = torch.as_tensor(self.dataloader.dataset.mean, **self.setup)[None, :, None, None]
        ds = torch.as_tensor(self.dataloader.dataset.std, **self.setup)[None, :, None, None]
        classes = self.dataloader.dataset.classes

        data = user_data["data"].clone().detach()
        labels = user_data["labels"].clone().detach() if user_data["labels"] is not None else None
        if labels is None:
            print_labels = False

        if scale:
            min_val, max_val = data.amin(dim=[2, 3], keepdim=True), data.amax(dim=[2, 3], keepdim=True)
            # print(f'min_val: {min_val} | max_val: {max_val}')
            data = (data - min_val) / (max_val - min_val)
        else:
            data.mul_(ds).add_(dm).clamp_(0, 1)
        data = data.to(dtype=torch.float32)

        if data.shape[0] == 1:
            plt.axis("off")
            plt.imshow(data[0].permute(1, 2, 0).cpu())
            if print_labels:
                plt.title(f"Data with label {classes[labels]}")
        else:
            grid_shape = int(torch.as_tensor(data.shape[0]).sqrt().ceil())
            s = 24 if data.shape[3] > 150 else 6
            fig, axes = plt.subplots(grid_shape, grid_shape, figsize=(s, s))
            label_classes = []
            for i, (im, axis) in enumerate(zip(data, axes.flatten())):
                axis.imshow(im.permute(1, 2, 0).cpu())
                if labels is not None and print_labels:
                    label_classes.append(classes[labels[i]])
                axis.axis("off")
            if print_labels:
                print(label_classes)


class UserMultiStep(UserSingleStep):
    """A user who computes multiple local update steps as in a FedAVG scenario."""

    def __init__(self, model, loss, dataloader, setup, cfg_user, num_queries=1):
        """Initialize but do not propagate the cfg_case.user dict further."""
        super().__init__(model, loss, dataloader, setup, cfg_user, num_queries)

        self.num_local_updates = cfg_user.num_local_updates
        self.num_data_per_local_update_step = cfg_user.num_data_per_local_update_step
        self.local_learning_rate = cfg_user.local_learning_rate
        self.provide_local_hyperparams = cfg_user.provide_local_hyperparams

    def __repr__(self):
        n = "\n"
        return (
            super().__repr__()
            + n
            + f"""    Local FL Setup:
        Number of local update steps: {self.num_local_updates}
        Data per local update step: {self.num_data_per_local_update_step}
        Local learning rate: {self.local_learning_rate}

        Threat model:
        Share these hyperparams to server: {self.provide_local_hyperparams}

        """
        )

    def compute_local_updates(self, server_payload):
        """Compute local updates to the given model based on server payload."""

        user_data, user_labels = self._load_data()

        # Compute local updates
        payload = server_payload["queries"][query]
        parameters = payload["parameters"]
        buffers = payload["buffers"]

        with torch.no_grad():
            for param, server_state in zip(self.model.parameters(), parameters):
                param.copy_(server_state.to(**self.setup))
            if buffers is not None:
                for buffer, server_state in zip(self.model.buffers(), buffers):
                    buffer.copy_(server_state.to(**self.setup))
                self.model.eval()
            else:
                self.model.train()

        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.local_learning_rate)
        seen_data_idx = 0
        label_list = []
        for step in range(self.num_local_updates):
            data = user_data[seen_data_idx : seen_data_idx + self.num_data_per_local_update_step]
            labels = user_labels[seen_data_idx : seen_data_idx + self.num_data_per_local_update_step]
            seen_data_idx += self.num_data_per_local_update_step
            seen_data_idx = seen_data_idx % self.num_data_points
            label_list.append(labels.sort()[0])

            optimizer.zero_grad()
            # Compute the forward pass
            data_input = data + self.generator_input.sample(data.shape) if self.generator_input is not None else data
            outputs = self.model(data_input)
            loss = self.loss(outputs, labels)
            loss.backward()

            grads_ref = [p.grad for p in self.model.parameters()]
            if self.clip_value > 0:
                self._clip_list_of_grad_(grads_ref)
            self._apply_differential_noise(grads_ref)
            optimizer.step()

        # Share differential to server version:
        # This is equivalent to sending the new stuff and letting the server do it, but in line
        # with the gradients sent in UserSingleStep
        shared_grads = [
            (p_local - p_server.to(**self.setup)).clone().detach()
            for (p_local, p_server) in zip(self.model.parameters(), parameters)
        ]

        shared_buffers = [b.clone().detach() for b in self.model.buffers()]
        metadata = dict(
            num_data_points=self.num_data_points if self.provide_num_data_points else None,
            labels=user_labels if self.provide_labels else None,
            local_hyperparams=dict(
                lr=self.local_learning_rate,
                steps=self.num_local_updates,
                data_per_step=self.num_data_per_local_update_step,
                labels=label_list,
            )
            if self.provide_local_hyperparams
            else None,
        )
        shared_data = dict(
            gradients=shared_grads, buffers=shared_buffers if self.provide_buffers else None, metadata=metadata
        )
        true_user_data = dict(data=user_data, labels=user_labels, buffers=shared_buffers)

        return shared_data, true_user_data


class MultiUserAggregate(UserMultiStep):
    """A silo of users who computes multiple local update steps as in a FedAVG scenario and aggregate their results.

    For an unaggregated single silo refer to SingleUser classes as above.
    A simple aggregate over multiple users in the FedSGD setting can better be modelled by the single user model above.
    """

    def __init__(self, model, loss, dataloader, setup, cfg_user, num_queries=1):
        """Initialize but do not propagate the cfg_case.user dict further."""
        super().__init__(model, loss, dataloader, setup, cfg_user, num_queries)

        self.num_users = cfg_user.num_users

    def __repr__(self):
        n = "\n"
        return (
            UserSingleStep.__repr__(self)
            + n
            + f"""    Local FL Setup:
        Number of aggregated users: {self.num_users}
        Number of local update steps: {self.num_local_updates}
        Data per local update step: {self.num_data_per_local_update_step}
        Local learning rate: {self.local_learning_rate}

        Threat model:
        Share these hyperparams to server: {self.provide_local_hyperparams}

        """
        )

    def _load_data(self, user_idx=0):
        """Take care to partition data among users on the fly."""

        raise NotImplementedError("Todo: Update this to the new and sane setup.")
        data_per_user = len(self.dataloader.dataset) // self.num_users
        user_data_subset = torch.utils.data.Subset(
            self.dataloader.dataset, list(range(data_per_user * user_idx, (user_idx + 1) * data_per_user))
        )
        data = []
        labels = []
        pointer = self.data_idx
        for _ in range(self.num_data_points):
            datum, label = self.dataloader.dataset[pointer]
            data += [datum]
            labels += [torch.as_tensor(label)]
            if self.data_with_labels == "unique":
                pointer += len(user_data_subset) // len(user_data_subset.dataset.classes)
            elif self.data_with_labels == "same":
                pointer += 1
            elif self.data_with_labels == "random":  # This will collide with the user thing
                pointer = torch.randint(0, len(user_data_subset), (1,))
            else:
                raise ValueError(f"Unknown {self.data_with_labels}")
            pointer = pointer % len(user_data_subset)  # Make sure not to leave the dataset range
        data = torch.stack(data).to(**self.setup)
        labels = torch.stack(labels).to(device=self.setup["device"])
        return data, labels

    def compute_local_updates(self, server_payload):
        """Compute local updates to the given model based on server payload."""
        # Compute local updates
        shared_grads = []
        shared_buffers = []
        for query in range(self.num_user_queries):
            payload = server_payload["queries"][query]
            server_parameters = payload["parameters"]
            server_buffers = payload["buffers"]

            aggregate_params = [torch.zeros_like(p) for p in self.model.parameters()]
            aggregate_buffers = [torch.zeros_like(b, dtype=torch.float) for b in self.model.buffers()]
            for user_idx in range(self.num_users):
                # Compute single user update
                user_data, user_labels = self._load_data(user_idx)
                with torch.no_grad():
                    for param, server_state in zip(self.model.parameters(), server_parameters):
                        param.copy_(server_state.to(**self.setup))
                    if server_buffers is not None:
                        for buffer, server_state in zip(self.model.buffers(), server_buffers):
                            buffer.copy_(server_state.to(**self.setup))
                        self.model.eval()
                    else:
                        self.model.train()

                optimizer = torch.optim.SGD(self.model.parameters(), lr=self.local_learning_rate)
                seen_data_idx = 0
                label_list = []
                for step in range(self.num_local_updates):
                    data = user_data[seen_data_idx : seen_data_idx + self.num_data_per_local_update_step]
                    labels = user_labels[seen_data_idx : seen_data_idx + self.num_data_per_local_update_step]
                    seen_data_idx += self.num_data_per_local_update_step
                    seen_data_idx = seen_data_idx % self.num_data_points
                    label_list.append(labels)

                    optimizer.zero_grad()
                    # Compute the forward pass
                    data_input = (
                        data + self.generator_input.sample(data.shape) if self.generator_input is not None else data
                    )
                    outputs = self.model(data_input)
                    loss = self.loss(outputs, labels)
                    loss.backward()

                    grads_ref = [p.grad for p in self.model.parameters()]
                    if self.clip_value > 0:
                        self._clip_list_of_grad_(grads_ref)
                    self._apply_differential_noise(grads_ref)
                    optimizer.step()
                # Add to running aggregates:

                # Share differential to server version:
                # This is equivalent to sending the new stuff and letting the server do it, but in line
                # with the gradients sent in UserSingleStep
                param_difference_to_server = torch._foreach_sub(
                    [p.cpu() for p in self.model.parameters()], server_parameters
                )
                torch._foreach_sub_(param_difference_to_server, aggregate_params)
                torch._foreach_add_(aggregate_params, param_difference_to_server, alpha=-1 / self.num_users)

                if len(aggregate_buffers) > 0:
                    buffer_to_server = [
                        b.to(device=torch.device("cpu"), dtype=torch.float) for b in self.model.buffers()
                    ]
                    torch._foreach_sub_(buffer_to_server, aggregate_buffers)
                    torch._foreach_add_(aggregate_buffers, buffer_to_server, alpha=1 / self.num_users)

            shared_grads += [aggregate_params]
            shared_buffers += [aggregate_buffers]

        shared_data = dict(
            gradients=shared_grads,
            buffers=shared_buffers if self.provide_buffers else None,
            num_data_points=self.num_data_points if self.provide_num_data_points else None,
            labels=user_labels if self.provide_labels else None,
            num_users=self.num_users,
            local_hyperparams=dict(
                lr=self.local_learning_rate,
                steps=self.num_local_updates,
                data_per_step=self.num_data_per_local_update_step,
                labels=label_list,
            )
            if self.provide_local_hyperparams
            else None,
        )

        def generate_user_data():
            for user_idx in range(self.num_users):
                yield self._generate_example_data(user_idx)[0]

        true_user_data = dict(data=generate_user_data(), labels=None, buffers=shared_buffers)

        return shared_data, true_user_data
