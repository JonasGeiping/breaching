"""Implement server code. This will be short, if the server is honest."""

"""Payload template:

payload should be a list of dicts. The length of this list is num_queries.
Each dict contains at least the keys "parameters" and "buffers".
"""


class HonestServer():
    """Implement an honest server protocol."""

    def __init__(self, model, loss, model_state='untrained', num_queries=1, cfg_data=None):
        """Inialize the server settings."""
        self.model = model
        self.loss = loss

        self.num_queries = num_queries
        self.model_state = model_state

        self.cfg_data = cfg_data  # Data configuration has to be shared across all parties to keep preprocessing consistent

    def reconfigure_model(self, model_state):
        """Reinitialize, continue training or otherwise modify model parameters in a benign way."""
        for name, module in self.model.named_modules:
            if model_state == 'untrained':
                module.reset_weights()
            elif model_state == 'trained':
                pass  # model was already loaded as pretrained model
            elif model_state == 'orthogonal':
                # reinit model with orthogonal parameters:
                if 'conv' in name or 'linear' in name:
                    torch.nn.init.orthogonal_(module.weight, gain=1)

    def distribute_payload(self):
        """Server payload to send to users. These are only references, to simplfiy the simulation."""

        payload = []
        for round in range(self.num_queries):
            self.reconfigure_model(self.model_state)

            honest_model_parameters = self.model.parameters()
            honest_model_buffers = self.model.buffers()
            payload.append(dict(parameters=honest_model_parameters, buffers=honest_model_buffers, data=self.cfg_data))
        return payload
