"""Implements a malicious block that can be inserted at the front on normal models to break them."""
from statistics import NormalDist
import torch



class ImprintBlock(torch.nn.Module):
    def __init__(self, data_size, num_bins, connection='linear', gain=1e-3):
        """
        data_size is the length of the input data
        num_bins is how many "paths" to include in the model
        connection is how this block should return back to the input shape (optional)
        """
        super().__init__()
        self.data_size = data_size
        self.num_bins = num_bins
        self.linear0 = torch.nn.Linear(data_size, num_bins)

        self.bins = self._get_bins()
        with torch.no_grad():
            self.linear0.weight.data[:, :] = self._make_average_layer() * gain
            self.linear0.bias.data[:] = self._make_biases() * gain

        self.connection = connection
        if connection == 'linear':
            self.linear2 = torch.nn.Linear(num_bins, data_size)
            with torch.no_grad():
                self.linear2.weight.data = torch.ones_like(self.linear2.weight.data)
                self.linear2.bias.data -= torch.as_tensor(self.bins).mean()

        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x_in = x
        x = self.linear0(x)
        x = self.relu(x)
        if self.connection == 'linear':
            output = self.linear2(x)
        elif self.connection == 'cat':
            output = torch.cat([x, x_in[:, self.num_bins:]], dim=1)
        elif self.connection == 'softmax':
            s = torch.softmax(x, dim=1)[:, :, None]
            output = (x_in[:, None, :] * s).sum(dim=1)
        else:
            output = x_in + x.mean(dim=1, keepdim=True)
        return output

    def _get_bins(self):
        left_bins = []
        bins = []
        mass_per_bin = 1 / (self.num_bins)
        bins.append(-10)  # -Inf is not great here, but NormalDist(mu=0, sigma=1).cdf(10) approx 1
        for i in range(1, self.num_bins):
            bins.append(NormalDist().inv_cdf(i * mass_per_bin))
        return bins

    def _make_average_layer(self):
        new_data = 1 / self.linear0.weight.data.shape[-1] * torch.ones_like(self.linear0.weight.data)
        return new_data

    def _make_biases(self):
        new_biases = torch.zeros_like(self.linear0.bias.data)
        for i in range(new_biases.shape[0]):
            new_biases[i] = -self.bins[i]
        return new_biases


class SparseImprintBlock(torch.nn.Module):
    def __init__(self, data_size, num_bins, connection='linear'):
        """
        data_size is the size of the input images
        num_bins is how many "paths" to include in the model

        This block uses a single hardtanh for simplicity of presentation.
        This not at all necessary and can be replaced with
        another simple linear+relu layer which replicates the same computation.
        """
        super().__init__()
        self.data_size = data_size
        self.num_bins = num_bins
        self.linear0 = torch.nn.Linear(data_size, num_bins)

        self.bins, self.bin_sizes = self._get_bins(num_bins)
        with torch.no_grad():
            self.linear0.weight.data[:, :] = self._make_scaled_average_layer()
            self.linear0.bias.data[:] = self._make_biases()
        self.hardtanh = torch.nn.Hardtanh(min_val=0, max_val=1)

        if connection == 'linear':
            self.linear2 = torch.nn.Linear(num_bins, data_size)
            with torch.no_grad():
                self.linear2.weight.data = torch.ones_like(self.linear2.weight.data)  # / data_size / num_bins

    def forward(self, x):
        x_in = x
        x = self.linear0(x)
        x = (self.hardtanh(x) + 1) / 2
        if self.connection == 'linear':
            output = self.linear2(x)
        elif self.connection == 'cat':
            output = torch.cat([x, x_in[:, self.num_bins:]], dim=1)
        elif self.connection == 'softmax':
            s = torch.softmax(x, dim=1)[:, :, None]
            output = (x_in[:, None, :] * s).sum(dim=1)
        else:
            output = x_in + x.mean(dim=1, keepdim=True)
        return output

    def _get_bins(self, num_bins, mu=0, sigma=1):
        bins = []
        mass = 0
        for path in range(num_bins + 1):
            mass += 1 / (num_bins + 2)
            bins += [NormalDist(mu=mu, sigma=sigma).inv_cdf(mass)]
        bin_sizes = [bins[i + 1] - bins[i] for i in range(len(bins) - 1)]
        return bins, bin_sizes

    def _make_scaled_average_layer(self):
        new_data = 1 / self.linear0.weight.data.shape[-1] * torch.ones_like(self.linear0.weight.data)
        for i, row in enumerate(new_data):
            row /= torch.as_tensor(self.bin_sizes[i], device=new_data.device)
        return new_data

    def _make_biases(self):
        new_biases = torch.zeros_like(self.linear0.bias.data)
        for i, (bin_val, bin_width) in enumerate(zip(self.bins[1:-1], self.bin_sizes[1:-1])):
            new_biases[i + 1] = -bin_val / bin_width
        return new_biases


class EquispacedImprintBlock(torch.nn.Module):
    """The old implementation."""

    def __init__(self, data_size, num_bins, connection='linear', alpha=0.375):
        """
        data_size is the size of the input images
        num_bins is how many "paths" to include in the model
        Does not accept other connections
        """
        super().__init__()
        self.data_size = data_size
        self.num_bins = num_bins
        self.linear0 = torch.nn.Linear(data_size, num_bins)
        self.linear1 = torch.nn.Linear(num_bins, data_size)

        self.alpha = alpha
        self.bins, self.bin_val = self._get_bins()
        with torch.no_grad():
            self.linear0.weight.data[:, :] = self._make_average_layer()
            self.linear0.bias.data[:] = self._make_biases()
        self.hardtanh = torch.nn.Hardtanh(min_val=0, max_val=self.bin_val)

    def forward(self, x):
        x = self.linear0(x)
        x = (self.hardtanh(x) + 1) / 2
        output = self.linear1(x)
        return output

    def _get_bins(self):
        order_stats = [self._get_order_stats(r + 1, self.num_bins) for r in range(self.num_bins)]
        diffs = [order_stats[i] - order_stats[i + 1] for i in range(len(order_stats) - 1)]
        bin_val = -sum(diffs) / len(diffs)
        left_bins = [-i * bin_val for i in range(self.num_bins // 2 + 1)]
        right_bins = [i * bin_val for i in range(1, self.num_bins // 2)]
        left_bins.reverse()
        bins = left_bins + right_bins
        return bins, bin_val

    def _get_order_stats(self, r, n):
        r"""
        r Order statistics can be computed as follows:
        E(r:n) = \mu + \Phi^{-1}\left( \frac{r-a}{n-2a+1} \sigma \right)
        where a = 0.375
        """
        return 0 + NormalDist().inv_cdf((r - self.alpha) / (n - 2 * self.alpha + 1)) * 1

    def _make_average_layer(self):
        new_data = 1 / self.linear0.weight.data.shape[-1] * torch.ones_like(self.linear0.weight.data)
        return new_data

    def _make_biases(self):
        new_biases = torch.zeros_like(self.linear0.bias.data)
        for i in range(len(new_biases)):
            new_biases[i] = self.bins[i]

        return new_biases


class OneShotBlock(SparseImprintBlock):
    def __init__(self, data_size, num_bins, connection='linear'):
        """
        data_size is the size of the input images
        num_bins is how many "paths" to include in the model
        """
        super().__init__(data_size, num_bins=1, connection=connection)
        self.data_size = data_size
        self.num_bins = num_bins

    def forward(self, x):
        x = self.linear0(x)
        x = self.hardtanh(x)
        output = self.linear2(x)
        return output

    def _get_bins(self):
        # Here we just build bins of uniform mass

        left_bins = []
        bins = []
        mass_per_bin = 1 / self.num_bins
        bins = [-NormalDist().inv_cdf(0.5), -NormalDist().inv_cdf(0.5 + mass_per_bin)]
        bin_sizes = [bins[i + 1] - bins[i] for i in range(len(bins) - 1)]
        bins = bins[:-1]  # here we need to throw away one on the right
        return bins, bin_sizes

    def _make_scaled_identity(self):
        new_data = torch.diag(1 / torch.tensor(self.bin_sizes))
        return new_data

    def _make_average_layer(self):
        new_data = 1 / self.linear0.weight.data.shape[-1] * torch.ones_like(self.linear0.weight.data)
        for i, row in enumerate(new_data):
            row *= 1 / torch.tensor(self.bin_sizes[i])
        return new_data

    def _make_biases(self):
        new_biases = torch.zeros_like(self.linear0.bias.data)
        new_biases[0] = -self.bins[0] / self.bin_sizes[0]
        return new_biases
