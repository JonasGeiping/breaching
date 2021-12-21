"""Repeatable code parts concerning data loading.
Data Config Structure (cfg_data): See config/data
"""


import torch
import torchvision
from .datasets import TinyImageNet, Birdsnap
from .cached_dataset import CachedDataset

import os

# Block ImageNet corrupt EXIF warnings
import warnings

warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)


def construct_dataloader(cfg_data, cfg_impl, user_idx=0, return_full_dataset=False):
    """Return a dataloader with given dataset for the given user_idx.

    Use return_full_dataset=True to return the full dataset instead (for example for analysis).
    """
    dataset = _build_dataset(cfg_data, cfg_data.examples_from_split, can_download=True)

    if not return_full_dataset:
        if user_idx is None:
            user_idx = torch.randint(0, cfg_data.default_clients, (1,))
        else:
            if user_idx > cfg_data.default_clients:
                raise ValueError("This user index exceeds the maximal number of clients.")

        # Create a synthetic split of the dataset over all possible users if no natural split is given
        if cfg_data.partition == "balanced":
            data_per_class_per_user = len(dataset) // len(dataset.classes) // cfg_data.default_clients
            if data_per_class_per_user < 1:
                raise ValueError("Too many clients for a balanced dataset.")
            data_ids = []
            for class_idx, _ in enumerate(dataset.classes):
                data_with_class = [idx for (idx, label) in dataset.lookup.items() if label == class_idx]
                data_ids += data_with_class[
                    user_idx * data_per_class_per_user : data_per_class_per_user * (user_idx + 1)
                ]
            dataset = Subset(dataset, data_ids)
        elif cfg_data.partition == "unique-class":
            data_ids = [idx for (idx, label) in dataset.lookup.items() if label == user_idx]
            dataset = Subset(dataset, data_ids)
        elif cfg_data.partition == "given":
            pass
        else:
            raise ValueError(f"Partition scheme {cfg_data.partition} not implemented.")

    if cfg_data.db.name == "LMDB":
        from .lmdb_datasets import LMDBDataset  # this also depends on py-lmdb, that's why it's a lazy import

        dataset = LMDBDataset(dataset, cfg_data, cfg_data.examples_from_split, can_create=True)

    if cfg_data.caching:
        dataset = CachedDataset(dataset, num_workers=cfg_impl.threads, pin_memory=cfg_impl.pin_memory)

    if cfg_impl.threads > 0:
        num_workers = (
            min(torch.get_num_threads(), cfg_impl.threads * max(1, torch.cuda.device_count()))
            if torch.get_num_threads() > 1
            else 0
        )
    else:
        num_workers = 0

    if cfg_impl.shuffle:
        data_sampler = torch.utils.data.RandomSampler(dataset, replacement=cfg_impl.sample_with_replacement)
    else:
        data_sampler = torch.utils.data.SequentialSampler(dataset)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=min(cfg_data.batch_size, len(dataset)),
        sampler=data_sampler,
        drop_last=True,  # just throw these images away :> :>
        num_workers=num_workers,
        pin_memory=cfg_impl.pin_memory,
        persistent_workers=cfg_impl.persistent_workers if num_workers > 0 else False,
    )

    return dataloader


def _build_dataset(cfg_data, split, can_download=True):
    _default_t = torchvision.transforms.ToTensor()
    cfg_data.path = os.path.expanduser(cfg_data.path)
    if cfg_data.name == "CIFAR10":
        dataset = torchvision.datasets.CIFAR10(
            root=cfg_data.path, train=split == "training", download=can_download, transform=_default_t,
        )
        dataset.lookup = dict(zip(list(range(len(dataset))), dataset.targets))
    elif cfg_data.name == "CIFAR100":
        dataset = torchvision.datasets.CIFAR100(
            root=cfg_data.path, train=split == "training", download=can_download, transform=_default_t,
        )
        dataset.lookup = dict(zip(list(range(len(dataset))), dataset.targets))
    elif cfg_data.name == "ImageNet":
        dataset = torchvision.datasets.ImageNet(
            root=cfg_data.path, split="train" if "train" in split else "val", transform=_default_t,
        )
        dataset.lookup = dict(zip(list(range(len(dataset))), [label for (_, label) in dataset.samples]))
    elif cfg_data.name == "ImageNetAnimals":
        dataset = torchvision.datasets.ImageNet(
            root=cfg_data.path, split="train" if "train" in split else "val", transform=_default_t,
        )
        dataset.lookup = dict(zip(list(range(len(dataset))), [label for (_, label) in dataset.samples]))
        indices = [idx for (idx, label) in dataset.lookup.items() if label < 398]
        dataset.classes = dataset.classes[:397]
        dataset.samples = [dataset.samples[i] for i in indices]  # Manually remove samples instead of using a Subset
        dataset.lookup = dict(zip(list(range(len(dataset))), [label for (_, label) in dataset.samples]))
    elif cfg_data.name == "TinyImageNet":
        dataset = TinyImageNet(
            root=cfg_data.path, split=split, download=can_download, transform=_default_t, cached=True,
        )
        dataset.lookup = dict(zip(list(range(len(dataset))), dataset.targets))
    elif cfg_data.name == "Birdsnap":
        dataset = Birdsnap(root=cfg_data.path, split=split, download=can_download, transform=_default_t)
        dataset.lookup = dict(zip(list(range(len(dataset))), dataset.labels))
    else:
        raise ValueError(f"Invalid dataset {cfg_data.name} provided.")

    if cfg_data.mean is None and cfg_data.normalize:
        data_mean, data_std = _get_meanstd(dataset)
        cfg_data.mean = data_mean
        cfg_data.std = data_std

    transforms = _parse_data_augmentations(cfg_data, split)

    # Apply transformations
    dataset.transform = transforms if transforms is not None else None

    # Save data mean and data std for easy access:
    if cfg_data.normalize:
        dataset.mean = cfg_data.mean
        dataset.std = cfg_data.std

    # Reduce train dataset according to cfg_data.size:
    if cfg_data.size < len(dataset):
        dataset = Subset(dataset, torch.arange(0, cfg_data.size))

    return dataset


class Subset(torch.utils.data.Subset):
    """Overwrite subset class to provide class methods of main class."""

    def __getattr__(self, name):
        """Call this only if all attributes of Subset are exhausted."""
        return getattr(self.dataset, name)


def _get_meanstd(dataset):
    print("Computing dataset mean and std manually ... ")
    # Run parallelized Wellford:
    current_mean = 0
    current_M2 = 0
    n = 0
    for data, _ in dataset:
        datapoint = data.view(3, -1)
        ds, dm = torch.std_mean(datapoint, dim=1)
        n_a, n_b = n, datapoint.shape[1]
        n += n_b
        delta = dm.to(dtype=torch.double) - current_mean
        current_mean += delta * n_b / n
        current_M2 += ds.to(dtype=torch.double) / (n_b - 1) + delta ** 2 * n_a * n_b / n
        # print(current_mean, (current_M2 / (n - 1)).sqrt())

    data_mean = current_mean.tolist()
    data_std = (current_M2 / (n - 1)).sqrt().tolist()
    print(f"Mean: {data_mean}. Standard deviation: {data_std}")
    return data_mean, data_std


def _parse_data_augmentations(cfg_data, split, PIL_only=False):
    def _parse_cfg_dict(cfg_dict):
        list_of_transforms = []
        if hasattr(cfg_dict, "keys"):
            for key in cfg_dict.keys():
                try:  # ducktype iterable
                    transform = getattr(torchvision.transforms, key)(*cfg_dict[key])
                except TypeError:
                    transform = getattr(torchvision.transforms, key)(cfg_dict[key])
                list_of_transforms.append(transform)
        return list_of_transforms

    if split == "train":
        transforms = _parse_cfg_dict(cfg_data.augmentations_train)
    else:
        transforms = _parse_cfg_dict(cfg_data.augmentations_val)

    if not PIL_only:
        transforms.append(torchvision.transforms.ToTensor())
        if cfg_data.normalize:
            transforms.append(torchvision.transforms.Normalize(cfg_data.mean, cfg_data.std))
    return torchvision.transforms.Compose(transforms)
