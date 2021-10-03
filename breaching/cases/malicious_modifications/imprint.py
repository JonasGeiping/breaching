"""Implements a malicious block that can be inserted at the front on normal models to break them."""
import torch

from scipy.stats import norm


class ImprintBlock(torch.nn.Module):
    def __init__(self, image_size, num_bins, alpha=0):
        """
        TODO: Get rid of annoying alpha argument
        image_size is the size of the input images
        num_bins is how many "paths" to include in the model
        """
        super().__init__()
        self.image_size = image_size
        self.num_bins = 2 * (num_bins // 2)  # make sure there's even num
        self.linear0 = torch.nn.Linear(image_size, num_bins)
        self.linear1 = torch.nn.Linear(num_bins, num_bins)
        self.linear2 = torch.nn.Linear(num_bins, image_size)
        self.bins, self.bin_sizes = self._get_bins()
        with torch.no_grad():
            self.linear0.weight.data[:, :] = self._make_average_layer()
            self.linear0.bias.data[:] = self._make_biases()
            self.linear1.weight.data[:, :] = self._make_scaled_identity()
            self.linear1.bias.data = torch.zeros_like(self.linear1.bias.data)
        self.hardtanh = torch.nn.Hardtanh(min_val=0, max_val=1)

    def forward(self, x):
        # x_input = x
        x = self.linear0(x)
        x = self.linear1(x)
        x = (self.hardtanh(x) + 1) / 2
        # output = x_input.clone()
        # output[:, :x.shape[1]] = x
        output = self.linear2(x)
        return output

    def _get_bins(self):
        # Here we just build bins of uniform mass

        left_bins = []
        mass_per_bin = 1 / self.num_bins
        for i in range(1, self.num_bins // 2 + 1):
            left_bins.append(norm.ppf(i * mass_per_bin))
        left_bins.append(0)
        right_bins = [-bin_val for bin_val in left_bins[:-1]]
        right_bins.reverse()
        bins = left_bins + right_bins
        bin_sizes = [bins[i + 1] - bins[i] for i in range(len(bins) - 1)]
        bins = bins[:-1]  # here we need to throw away one on the right
        return bins, bin_sizes

    def _make_scaled_identity(self):
        new_data = torch.diag(1 / torch.tensor(self.bin_sizes))
        return new_data

    def _make_average_layer(self):
        new_data = 1 / self.linear0.weight.data.shape[-1] * torch.ones_like(self.linear0.weight.data)
        return new_data

    def _make_biases(self):
        new_biases = torch.zeros_like(self.linear0.bias.data)
        for i in range(len(new_biases)):
            new_biases[i] = self.bins[i]

        return new_biases


from statistics import NormalDist

class DifferentialBlock(torch.nn.Module):
    """Recover data in v-notation instead of u, i.e. from differences in gradients instead of 1-hot."""
    def __init__(self, input_length, num_bins, alpha=None):
        super().__init__()
        self.linear = torch.nn.Linear(input_length, num_bins)
        self.nonlin = torch.nn.ReLU()

        # self.scaler = torch.nn.Linear(num_bins, num_bins, bias=False)  # sanity check
        # the linear_out layer is just to plug-and-play in any location. This is not strictly necessary.
        # You could just as well just connect from num_bins to the next layer
        self.linear_out = torch.nn.Linear(num_bins, input_length)
        self.bins, self.bin_sizes = self.get_bins_by_mass(num_bins)
        # Initialize:
        self.reset_weights()

    def reset_weights(self):
        with torch.no_grad():
            setup = dict(device=self.linear.weight.device, dtype=self.linear.weight.dtype)
            
            self.linear.weight.data = torch.ones_like(self.linear.weight) # / torch.as_tensor(self.bin_sizes, **setup)[:, None]
            self.linear.weight.data /= self.linear.in_features
            self.linear.bias.data = -torch.as_tensor(self.bins, **setup)

            # self.scaler.weight.data = torch.eye(self.linear.out_features, **setup) #torch.diag(1 / torch.as_tensor(self.bin_sizes, **setup))

            torch.nn.init.orthogonal_(self.linear_out.weight, gain=1.0)

    def get_bins_by_mass(self, num_bins, mu=0, sigma=1):
        bins = []
        mass = 0
        for path in range(num_bins + 1):
            mass += 1 / (num_bins + 2)
            bins += [NormalDist(mu=mu, sigma=sigma).inv_cdf(mass)]
        bin_sizes = [bins[i + 1] - bins[i] for i in range(len(bins) - 1)]
        bins = torch.linspace(-1, 1, num_bins)
        return bins, bin_sizes


    def forward(self, x):
        x = self.nonlin(self.linear(x))
        print(x)
        x = self.linear_out(x)
        return x


class EquispacedImprintBlock(torch.nn.Module):
    """The old implementation."""
    def __init__(self, image_size, num_bins, alpha=0.375):
        """
        image_size is the size of the input images
        num_bins is how many "paths" to include in the model
        """
        super().__init__()
        self.image_size = image_size
        self.num_bins = num_bins
        self.linear0 = torch.nn.Linear(image_size, num_bins)
        self.linear1 = torch.nn.Linear(num_bins, image_size)

        self.alpha = alpha
        self.bins, self.bin_val = self._get_bins()
        with torch.no_grad():
            self.linear0.weight.data[:, :] = self._make_average_layer()
            self.linear0.bias.data[:] = self._make_biases()
        self.hardtanh = torch.nn.Hardtanh(min_val=0, max_val=self.bin_val)

    def forward(self, x):
        # x_input = x
        x = self.linear0(x)
        x = (self.hardtanh(x) + 1) / 2
        # output = x_input.clone()
        # output[:, :x.shape[1]] = x
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
        return 0 + norm.ppf((r - self.alpha) / (n - 2 * self.alpha + 1)) * 1

    def _make_average_layer(self):
        new_data = 1 / self.linear0.weight.data.shape[-1] * torch.ones_like(self.linear0.weight.data)
        return new_data

    def _make_biases(self):
        new_biases = torch.zeros_like(self.linear0.bias.data)
        for i in range(len(new_biases)):
            new_biases[i] = self.bins[i]

        return new_biases
