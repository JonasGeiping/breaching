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
        self.num_bins = 2*(num_bins // 2) # make sure there's even num
        self.linear0 = torch.nn.Linear(image_size, num_bins)
        #self.linear1 = torch.nn.Linear(num_bins, num_bins)
        self.linear2 = torch.nn.Linear(num_bins, image_size)
        self.bins, self.bin_sizes = self._get_bins()
        with torch.no_grad():
            self.linear0.weight.data[:, :] = self._make_average_layer()
            self.linear0.bias.data[:] = self._make_biases()
            self.linear2.weight.data = torch.ones_like(self.linear2.weight.data)
            #self.linear1.weight.data[:, :] = self._make_scaled_identity()
            #self.linear1.bias.data = torch.zeros_like(self.linear1.bias.data)
        #self.hardtanh = torch.nn.Hardtanh(min_val=0, max_val=1)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        # x_input = x
        x = self.linear0(x)
        x = self.relu(x) 
        #x = self.hardtanh(x)
        #x = (self.hardtanh(x) + 1) / 2
        # output = x_input.clone()
        # output[:, :x.shape[1]] = x
        output = self.linear2(x)
        return output
    
    def feats(self, x):
        # x_input = x
        x = self.linear0(x)
        #x = self.linear1(x)
        x = self.hardtanh(x)
        return x

    def _get_bins(self):
        # Here we just build bins of uniform mass
        
        left_bins = []
        bins = []
        mass_per_bin = 1 / self.num_bins
        for i in range(self.num_bins+1):
            bins.append(norm.ppf(i*mass_per_bin))
        '''
        for i in range(1, self.num_bins // 2 + 1):
            left_bins.append(norm.ppf(i*mass_per_bin))
        left_bins.append(0)
        right_bins = [-bin_val for bin_val in left_bins[:-1]]
        right_bins.reverse()
        bins = left_bins + right_bins
        '''
        bin_sizes = [bins[i+1] - bins[i] for i in range(len(bins) - 1)]
        bins = bins[1:] # here we need to throw away one on the right
        return bins, bin_sizes

    def _make_scaled_identity(self):
        new_data = torch.diag(1/torch.tensor(self.bin_sizes))
        return new_data

    def _make_average_layer(self):
        new_data = 1 / self.linear0.weight.data.shape[-1] * torch.ones_like(self.linear0.weight.data)
        '''
        for i, row in enumerate(new_data):
            row *= 1/torch.tensor(self.bin_sizes[i])
        '''
        return new_data

    def _make_biases(self):
        new_biases = torch.zeros_like(self.linear0.bias.data)
        for i in range(new_biases.shape[0]):
            new_biases[i] = -self.bins[i]
        #new_biases[0] = 1e20
        '''
        for i, (bin_val, bin_width) in enumerate(zip(self.bins[1:-1], self.bin_sizes[1:-1])):
            #for i in range(len(new_biases)):
            new_biases[i+1] = -bin_val / bin_width
            #new_biases[i+1] = -self.bins[i] / self.bin_sizes[i]
        '''
        return new_biases

    
    ''' Old imprint block. Sleep here for now... we may need you later 
    
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
        """
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
        
    '''

