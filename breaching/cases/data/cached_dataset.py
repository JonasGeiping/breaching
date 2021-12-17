"""Write a PyTorch dataset into RAM."""

import torch
import logging

log = logging.getLogger(__name__)


class CachedDataset(torch.utils.data.Dataset):
    """Cache a given dataset into RAM or SDRAM (GPU memory).

    This is only a good idea if you have enough RAM, especially if mapping into SDRAM.
    """

    def __init__(
        self, dataset, num_workers=0, setup=dict(device=torch.device("cpu"), dtype=torch.float), pin_memory=True
    ):
        """Initialize with a given pytorch dataset. The setup dictionary determines cache location and storage type."""
        self.dataset = dataset
        self.cache = []
        log.info("Caching started ...")
        batch_size = min(len(dataset) // max(num_workers, 1), 8192)
        cacheloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=False
        )

        # Allocate memory:
        pin = pin_memory and (setup["device"] == torch.device("cpu"))
        self.input_cache = torch.empty((len(self.dataset), *self.dataset[0][0].shape), pin_memory=pin, **setup)
        self.label_cache = torch.empty((len(self.dataset)), pin_memory=pin, dtype=torch.long, device=setup["device"])
        pointer = 0
        for data in cacheloader:
            batch_length = data[0].shape[0]
            self.input_cache[pointer : pointer + batch_length] = data[
                0
            ]  # assuming the first return value of data is the image sample!
            self.label_cache[pointer : pointer + batch_length] = data[1]
            pointer += batch_length

        log.info(f'Dataset sucessfully cached into {"RAM" if setup["device"] == torch.device("cpu") else "SDRAM"}.')

    def __getitem__(self, index):
        """Get sample, target from cache."""
        sample = self.input_cache[index]
        label = self.label_cache[index]
        return sample, label

    def __len__(self):
        """Length is length of self.dataset."""
        return len(self.dataset)

    def __getattr__(self, name):
        """This is only called if all attributes of Subset are exhausted."""
        return getattr(self.dataset, name)
