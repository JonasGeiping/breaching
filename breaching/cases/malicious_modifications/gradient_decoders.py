"""Model architectures that decode gradient information."""
import torch


class AmygdalaDecoder(torch.nn.Module):
    """Recover inputs with a learned decoder of gradient information.

    The entire gradient vector is used.
    """

    def __init__(self, model, loss, target_shape=(64, 3, 32, 32)):
        """Enforce that input data matches this gradient vector."""
        super().__init__()
        self.loss = loss

        model_parameter_vector = torch.cat([p.view(-1) for p in model.parameters()])
        self.target_block_size = torch.prod(torch.as_tensor(target_shape))
        self.target_shape = list(target_shape)

        self.decoder = self._construct_amygdala(model)

    def _construct_amygdala(self, model):
        """Parse the model backwards and construct a decoder module."""
        param_shapes = [p.shape for p in model.parameters()]
        feature_shapes = self._introspect_model(model)
        return Amygdala(param_shapes, feature_shapes)

    def _introspect_model(self, model):
        """Compute model feature shapes."""
        feature_shapes = dict()

        def named_hook(name):
            def hook_fn(module, input, output):
                feature_shapes[name] = dict(ref=module, shape=input[0].shape, info=str(module))

            return hook_fn

        hooks_list = []
        for name, module in model.named_modules():
            if isinstance(
                module,
                (
                    torch.nn.Conv2d,
                    torch.nn.Linear,
                    torch.nn.BatchNorm2d,
                    torch.nn.MaxPool2d,
                    torch.nn.AvgPool2d,
                    torch.nn.Flatten,
                ),
            ):
                hooks_list.append(module.register_forward_hook(named_hook(name)))

        test_param = next(model.parameters())
        throughput = torch.randn(self.target_shape, dtype=test_param.dtype, device=test_param.device)
        model(throughput)
        # True user data is returned only for analysis
        [h.remove() for h in hooks_list]
        return feature_shapes

    def forward(self, model, inputs, labels):
        default_loss = self.loss(model(inputs), labels)
        grads = torch.autograd.grad(default_loss, model.parameters(), create_graph=True)

        outputs = self.decoder(grads)
        final_loss = (inputs[: self.target_shape[0]] - outputs).pow(2).mean()

        return outputs, final_loss, default_loss


class Amygdala(torch.nn.Module):
    """Model mapping from parameter-vector / gradient-vector sized input to model input sized output."""

    def __init__(self, parameter_shapes, feature_shapes):
        """Instantiate only from shapes."""
        super().__init__()
        modules = []
        for key in reversed(feature_shapes):
            print(key, feature_shapes[key])
            module = feature_shapes[key]["ref"]
            if isinstance(module, torch.nn.Linear):
                print("yes")

    def forward(self, grads):
        return None


class Linearizer(torch.nn.Module):
    """Convert from linear layer to its input."""

    def __init__(self, param_shape, input_shape):
        super().__init__()
        self.layers = []
        for param in param_shape:
            self.layers += torch.nn.Linear(2 * param_shape, torch.as_tensor(input_shape).prod())

    def forward(self, parameters, grads):
        state = 0
        for param, grad, layer in zip(parameters, grads, self.layers):
            state += layer(torch.cat(param, grad))
            # ?need to figure out batching
        return input_sized
