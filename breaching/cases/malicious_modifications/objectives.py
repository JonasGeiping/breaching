"""Implement objective functions."""

import torch
import scipy.fft

class DeepLayerRatioMatching(torch.nn.Module):
    """Recover inputs from the ratio of the given layers.

    Generally layers[0] should be a conv/linear layer, and layers[1] the next bias after it.
    Dividing both quantities cancels the gradient contributions of further layers by the chain rule.
    """

    def __init__(self, model, loss, target_shape=(64, 3, 32, 32), layers=['layer2.0.conv2', 'layer2.0.bn2']):
        """Enforce that input data matches this gradient vector."""
        super().__init__()
        self.model = model
        self.loss = loss

        for idx, (name, val) in enumerate(model.named_parameters()):
            if name == layers[0] + '.weight':
                self.weight_idx = idx
                weight_shape = val.shape
            if name == layers[1] + '.bias':
                self.bias_idx = idx
                bias_shape = val.shape

        maximal_batch_size = torch.div(torch.prod(torch.as_tensor(weight_shape)),
                                       torch.prod(torch.as_tensor(target_shape[1:])),
                                       rounding_mode='floor')
        maximal_batch_size = torch.as_tensor(4)
        if maximal_batch_size < target_shape[0]:
            print(f'Reducing targeted batch size to {maximal_batch_size} due to layer shapes.')
            target_shape[0] = maximal_batch_size.item()
        self.target_block_size = torch.prod(torch.as_tensor(target_shape))
        self.target_shape = list(target_shape)


    def forward(self, inputs, labels, eps=1e-4):
        default_loss = self.loss(self.model(inputs), labels)
        grads = torch.autograd.grad(default_loss, self.model.parameters(), create_graph=True)
        # maybe need to guard against zero here in a smart way:
        weight = grads[self.weight_idx]
        bias = grads[self.bias_idx][:, None, None, None]
        differentiable_ratio = weight / (bias**2 + eps**2).sqrt() * bias.sign()
        inputs_prototype = differentiable_ratio.view(-1)[:self.target_block_size].reshape(self.target_shape)


        input_frequencies = torch.as_tensor(scipy.fft.dctn(inputs.cpu().numpy(), axes=[2,3], norm='ortho'), device=inputs.device, dtype=inputs.dtype)
        final_loss = (input_frequencies[:self.target_shape[0]] - inputs_prototype).pow(2).mean()

        outputs = torch.as_tensor(scipy.fft.idctn(inputs_prototype.detach().cpu().numpy(), axes=[2,3], norm='ortho'), device=inputs.device, dtype=inputs.dtype)

        return outputs, final_loss, default_loss
