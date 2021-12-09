"""Implements a malicious block that can be inserted at the front on normal models to break them."""
from statistics import NormalDist
import torch
import math
from scipy.stats import laplace


class ImprintBlock(torch.nn.Module):
    structure = "cumulative"

    def __init__(self, data_size, num_bins, connection="linear", gain=1e-3, linfunc="fourier", mode=0):
        """
        data_size is the length of the input data
        num_bins is how many "paths" to include in the model
        connection is how this block should coonect back to the input shape (optional)

        linfunc is the choice of linear query function ('avg', 'fourier', 'randn', 'rand').
        If linfunc is fourier, then the mode parameter determines the mode of the DCT-2 that is used as linear query.
        """
        super().__init__()
        self.data_size = data_size
        self.num_bins = num_bins
        self.linear0 = torch.nn.Linear(data_size, num_bins)

        self.bins = self._get_bins(linfunc)
        with torch.no_grad():
            self.linear0.weight.data = self._init_linear_function(linfunc, mode) * gain
            self.linear0.bias.data = self._make_biases() * gain

        self.connection = connection
        if connection == "linear":
            self.linear2 = torch.nn.Linear(num_bins, data_size)
            with torch.no_grad():
                self.linear2.weight.data = torch.ones_like(self.linear2.weight.data) / gain
                self.linear2.bias.data -= torch.as_tensor(self.bins).mean()

        self.nonlin = torch.nn.ReLU()

    @torch.no_grad()
    def _init_linear_function(self, linfunc="fourier", mode=0):
        K, N = self.num_bins, self.data_size
        if linfunc == "avg":
            weights = torch.ones_like(self.linear0.weight.data) / N
        elif linfunc == "fourier":
            weights = torch.cos(math.pi / N * (torch.arange(0, N) + 0.5) * mode).repeat(K, 1) / N * max(mode, 0.33) * 4
            # dont ask about the 4, this is WIP
            # nonstandard normalization
        elif linfunc == "randn":
            weights = torch.randn(N).repeat(K, 1)
            std, mu = torch.std_mean(weights[0])  # Enforce mean=0, std=1 with higher precision
            weights = (weights - mu) / std / math.sqrt(N)  # Move to std=1 in output dist
        elif linfunc == "rand":
            weights = torch.rand(N).repeat(K, 1)  # This might be a terrible idea haven't done the math
            std, mu = torch.std_mean(weights[0])  # Enforce mean=0, std=1
            weights = (weights - mu) / std / math.sqrt(N)  # Move to std=1 in output dist
        else:
            raise ValueError(f"Invalid linear function choice {linfunc}.")

        return weights

    def _get_bins(self, linfunc="avg"):
        bins = []
        mass_per_bin = 1 / (self.num_bins)
        bins.append(-10)  # -Inf is not great here, but NormalDist(mu=0, sigma=1).cdf(10) approx 1
        for i in range(1, self.num_bins):
            if "fourier" in linfunc:
                bins.append(laplace(loc=0.0, scale=1 / math.sqrt(2)).ppf(i * mass_per_bin))
            else:
                bins.append(NormalDist().inv_cdf(i * mass_per_bin))
        return bins

    def _make_biases(self):
        new_biases = torch.zeros_like(self.linear0.bias.data)
        for i in range(new_biases.shape[0]):
            new_biases[i] = -self.bins[i]
        return new_biases

    def forward(self, x):
        x_in = x
        x = self.linear0(x)
        x = self.nonlin(x)
        if self.connection == "linear":
            output = self.linear2(x)
        elif self.connection == "cat":
            output = torch.cat([x, x_in[:, self.num_bins :]], dim=1)
        elif self.connection == "softmax":
            s = torch.softmax(x, dim=1)[:, :, None]
            output = (x_in[:, None, :] * s).sum(dim=1)
        else:
            output = x_in + x.mean(dim=1, keepdim=True)
        return output


class SparseImprintBlock(ImprintBlock):
    structure = "sparse"

    """This block is sparse instead of cumulative which is more efficient in noise/param tradeoffs but requires
    two ReLUs that construct the hard-tanh nonlinearity."""

    def _get_bins(self, mu=0, sigma=1, linfunc="avg"):
        bins = []
        mass = 0
        for path in range(self.num_bins + 1):
            mass += 1 / (self.num_bins + 2)
            if "fourier" in linfunc:
                bins.append(laplace(loc=mu, scale=sigma / math.sqrt(2)).ppf(mass))
            else:
                bins += [NormalDist(mu=mu, sigma=sigma).inv_cdf(mass)]
        bin_sizes = [bins[i + 1] - bins[i] for i in range(len(bins) - 1)]
        self.bin_sizes = bin_sizes
        return bins[1:]

    @torch.no_grad()
    def _init_linear_function(self, linfunc="fourier", mode=0):
        weights = super()._init_linear_function(linfunc, mode)
        for i, row in enumerate(weights):
            row /= torch.as_tensor(self.bin_sizes[i], device=new_data.device)
        return weights

    def _make_biases(self):
        new_biases = torch.zeros_like(self.linear0.bias.data)
        for i, (bin_val, bin_width) in enumerate(zip(self.bins, self.bin_sizes)):
            new_biases[i] = -bin_val / bin_width
        return new_biases


class OneShotBlock(ImprintBlock):
    structure = "cumulative"

    """One-shot attack with minimal additional parameters. Can target a specific data point if its target_val is known."""

    def __init__(self, data_size, num_bins, connection="linear", gain=1e-3, linfunc="fourier", mode=0, target_val=0):
        self.virtual_bins = num_bins
        self.target_val = target_val
        num_bins = 2
        super().__init__(data_size, num_bins, connection, gain, linfunc, mode)

    def _get_bins(self, linfunc="avg"):
        bins = []
        mass_per_bin = 1 / (self.virtual_bins)
        bins.append(-10)  # -Inf is not great here, but NormalDist(mu=0, sigma=1).cdf(10) approx 1
        for i in range(1, self.virtual_bins):
            if "fourier" in linfunc:
                bins.append(laplace(loc=0.0, scale=1 / math.sqrt(2)).ppf(i * mass_per_bin))
            else:
                bins.append(NormalDist().inv_cdf(i * mass_per_bin))
            if self.target_val < bins[-1]:
                break
        return bins[-2:]


class OneShotBlockSparse(SparseImprintBlock):
    structure = "sparse"

    def __init__(self, data_size, num_bins, connection="linear"):
        """
        data_size is the size of the input images
        num_bins is how many "paths" to include in the model
        """
        super().__init__(data_size, num_bins=1, connection=connection)
        self.data_size = data_size
        self.num_bins = num_bins

    def _get_bins(self):
        # Here we just build bins of uniform mass
        left_bins = []
        bins = []
        mass_per_bin = 1 / self.num_bins
        bins = [-NormalDist().inv_cdf(0.5), -NormalDist().inv_cdf(0.5 + mass_per_bin)]
        self.bin_sizes = [bins[i + 1] - bins[i] for i in range(len(bins) - 1)]
        bins = bins[:-1]  # here we need to throw away one on the right
        return bins
