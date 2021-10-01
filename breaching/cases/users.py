"""Implement user code."""

import torch
import copy


class UserSingleStep(torch.nn.Module):
    """A user who computes a single local update step."""

    def __init__(self, model, loss, dataloader, setup, num_data_points=1, num_user_queries=1, batch_norm_training=False,
                 provide_labels=True, provide_num_data_points=True, data_idx=None, num_local_updates=1,
                 num_data_per_local_update_step=None, local_learning_rate=None):
        """Initialize but do not propagate the cfg_case.user dict further."""
        super().__init__()
        self.num_data_points = num_data_points
        self.num_user_queries = num_user_queries

        self.num_local_updates = num_local_updates
        self.num_data_per_local_update_step = num_data_per_local_update_step
        self.local_learning_rate = local_learning_rate

        self.provide_labels = provide_labels
        self.provide_num_data_points = provide_num_data_points

        if data_idx is None:
            self.data_idx = torch.randint(0, len(dataloader.dataset), (1,))
        else:
            self.data_idx = data_idx

        self.setup = setup

        self.model = copy.deepcopy(model)
        self.model.to(**setup)
        if batch_norm_training:
            self.model.training()
        else:
            self.model.eval()

        self.dataloader = dataloader
        self.loss = copy.deepcopy(loss)  # Just in case the loss contains state

    def __repr__(self):
        return f"""User (of type {self.__class__.__name__} with settings:
            number of local updates: {self.num_local_updates}
            number of data points: {self.num_data_points}
            number of user queries {self.num_user_queries}

            Threat model:
            User provides labels: {self.provide_labels}
            User provides number of data points: {self.provide_num_data_points}

            Model:
            model specification: {str(self.model.__class__.__name__)}
            loss function: {str(self.loss)}

            Data:
            Dataset: {self.dataloader.dataset.__class__.__name__}
            data_idx: {self.data_idx.item() if isinstance(self.data_idx, torch.Tensor) else self.data_idx}
        """

    def compute_local_updates(self, server_payload):
        """Compute local updates to the given model based on server payload."""

        data, labels = self._generate_example_data()
        # Compute local updates
        shared_grads = []
        shared_buffers = []
        for query in range(self.num_user_queries):
            payload = server_payload['queries'][query]
            parameters = payload['parameters']
            buffers = payload['buffers']

            with torch.no_grad():
                for param, server_state in zip(self.model.parameters(), parameters):
                    param.copy_(server_state.to(**self.setup))
                for buffer, server_state in zip(self.model.buffers(), buffers):
                    buffer.copy_(server_state.to(**self.setup))

            # Compute the forward pass
            outputs = self.model(data)
            loss = self.loss(outputs, labels)

            shared_grads += [torch.autograd.grad(loss, self.model.parameters())]
            shared_buffers += [[b.clone().detach() for b in self.model.buffers()]]

        shared_data = dict(gradients=shared_grads, buffers=shared_buffers,
                           num_data_points=self.num_data_points if self.provide_num_data_points else None,
                           labels=labels if self.provide_labels else None)
        true_user_data = dict(data=data, labels=labels)

        return shared_data, true_user_data


    def _generate_example_data(self):
        # Select data
        data = []
        labels = []
        pointer = self.data_idx
        for data_point in range(self.num_data_points):
            datum, label = self.dataloader.dataset[pointer]
            data += [datum]
            labels += [torch.as_tensor(label)]
            pointer += len(self.dataloader.dataset.classes)
            pointer = pointer % len(self.dataloader.dataset)
        data = torch.stack(data).to(**self.setup)
        labels = torch.stack(labels).to(device=self.setup['device'])
        return data, labels


    def plot(self, user_data, scale=False):
        """Plot user data to output. Probably best called from a jupyter notebook."""
        import matplotlib.pyplot as plt  # lazily import this here

        dm = torch.as_tensor(self.dataloader.dataset.mean, **self.setup)[None, :, None, None]
        ds = torch.as_tensor(self.dataloader.dataset.std, **self.setup)[None, :, None, None]
        classes = self.dataloader.dataset.classes

        data = user_data['data'].clone().detach()
        labels = user_data['labels'].clone().detach()

        data.mul_(ds).add_(dm).clamp_(0, 1)

        if scale:
            min_val, max_val = data.amin(dim=[2, 3], keepdim=True), data.amax(dim=[2, 3], keepdim=True)
            data = (data - min_val) / (max_val - min_val)

        if data.shape[0] == 1:
            plt.imshow(data[0].permute(1, 2, 0).cpu())
            plt.title(f'Data with label {classes[labels]}')
        else:
            grid_shape = int(torch.as_tensor(data.shape[0]).sqrt().ceil())
            s = 24 if data.shape[3] > 150 else 6
            fig, axes = plt.subplots(grid_shape, grid_shape, figsize=(s, s))
            label_classes = []
            for i, (im, axis) in enumerate(zip(data, axes.flatten())):
                axis.imshow(im.permute(1, 2, 0).cpu())
                label_classes.append(classes[labels[i]])
                axis.axis('off')
            print(label_classes)


class UserMultiStep(UserSingleStep):
    """A user who computes multiple local update steps as in a FedAVG scenario."""

    def __init__(self, model, loss, dataloader, setup, num_data_points=1, num_user_queries=1, batch_norm_training=False,
                 provide_labels=True, provide_num_data_points=True, data_idx=None, num_local_updates=1,
                 num_data_per_local_update_step=None, local_learning_rate=None):
        """Initialize but do not propagate the cfg_case.user dict further."""
        super().__init__(model, loss, dataloader, setup, num_data_points, num_user_queries, batch_norm_training,
                         provide_labels, provide_num_data_points, data_idx, num_local_updates,
                         num_data_per_local_update_step, local_learning_rate)

        # assert self.num_data_per_local_update_step * self.num_local_updates < self.num_data_points # ?

    def compute_local_updates(self, server_payload):
        """Compute local updates to the given model based on server payload."""

        user_data, user_labels = self._generate_example_data()

        # Compute local updates
        shared_grads = []
        shared_buffers = []
        for query in range(self.num_user_queries):
            payload = server_payload['queries'][query]
            parameters = payload['parameters']
            buffers = payload['buffers']

            with torch.no_grad():
                for param, server_state in zip(self.model.parameters(), parameters):
                    param.copy_(server_state.to(**self.setup))
                for buffer, server_state in zip(self.model.buffers(), buffers):
                    buffer.copy_(server_state.to(**self.setup))

            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.local_learning_rate)
            seen_data_idx = 0

            for step in range(self.num_local_updates):

                data = user_data[seen_data_idx: seen_data_idx + self.num_data_per_local_update_step]
                labels = user_labels[seen_data_idx: seen_data_idx + self.num_data_per_local_update_step]
                seen_data_idx += self.num_data_per_local_update_step
                seen_data_idx = seen_data_idx % self.num_data_points

                optimizer.zero_grad()
                # Compute the forward pass
                outputs = self.model(data)
                loss = self.loss(outputs, labels)
                loss.backward()
                optimizer.step()

            # Share differential to server version:
            # This is equivalent to sending the new stuff and letting the server do it, but in line
            # with the gradients sent in UserSingleStep
            shared_grads += [[p_local - p_server for (p_local, p_server) in zip(self.model.parameters(), parameters)]]
            shared_buffers += [[b.clone().detach() for b in self.model.buffers()]]

        shared_data = dict(gradients=shared_grads, buffers=shared_buffers,
                           num_data_points=self.num_data_points if self.provide_num_data_points else None,
                           labels=user_labels if self.provide_labels else None)
        true_user_data = dict(data=user_data, labels=user_labels)

        return shared_data, true_user_data
