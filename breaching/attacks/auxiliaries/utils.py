"""Various utility operations."""

import torch


class AntiAlias(torch.nn.Module):
    """Simple anti-aliasing. Based pretty much on the implementation from "Making Convolutional Networks Shift-Invariant Again"
    at https://github.com/adobe/antialiased-cnns/blob/master/antialiased_cnns/blurpool.py
    """

    filter_bank = {
        1: [
            1.0,
        ],
        2: [1.0, 1.0],
        3: [1.0, 2.0, 1.0],
        4: [1.0, 3.0, 3.0, 1.0],
        5: [1.0, 4.0, 6.0, 4.0, 1.0],
        6: [1.0, 5.0, 10.0, 10.0, 5.0, 1.0],
        7: [1.0, 6.0, 15.0, 20.0, 15.0, 6.0, 1.0],
    }

    def __init__(self, channels, width=5, stride=1):
        super().__init__()
        self.width = int(width)
        self.padding = width // 2 + 1
        self.stride = stride
        self.channels = channels

        filter_base = torch.as_tensor(self.filter_bank[self.width])
        antialias = filter_base[:, None] * filter_base[None, :]
        antialias = antialias / antialias.sum()
        self.register_buffer("antialias", antialias[None, None, :, :].repeat((self.channels, 1, 1, 1)))

    def forward(self, inputs):
        return torch.nn.functional.conv2d(
            inputs, self.antialias, padding=self.padding, stride=self.stride, groups=inputs.shape[1]
        )
