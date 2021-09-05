"""Implement user code."""

import torch
import copy


class UserSingleStep(torch.nn.Module):
    """A user who computes a single local update step."""

    def __init__(model, loss, dataloader, setup, num_data_points=1, num_user_queries=1, batch_norm_training=False,
                 provide_labels=True, provide_num_data_points=True, data_idx=None):
        """Initialize but do not propagate the cfg_case.user dict further."""
        super().__init__()

        self.num_local_updates = 1
        self.num_data_points = num_data_points
        self.num_user_queries = num_user_queries

        self.provide_labels = provide_labels
        self.provide_num_data_points = provide_num_data_points

        if self.data_idx is None:
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

    def compute_local_updates(self, server_payload):
        """Compute local updates to the given model based on server payload."""

        # Select data
        data, labels = self.dataloader[self.data_idx // self.dataloader.batch_size]
        data, labels = data[0:self.num_data_points], labels[0:self.num_data_points]
        data = data.to(**setup)
        labels = labels.to(device=setup['device'])

        # Compute local updates
        shared_grads = []
        for query in range(self.num_queries):
            payload = server_payload[query]
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

        shared_data = dict(gradients=shared_grads, buffers=model.buffers(),
                           num_data_points=self.num_data_points if self.provide_num_data_points else None,
                           labels=labels if self.provide_labels else None)
        true_user_data = dict(data=data, labels=labels)

        return shared_data, true_user_data
