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


def construct_dataloader(cfg_data, cfg_impl, split, dryrun=False):
    """Return a dataloader with given dataset. Choose number of workers and their settings."""

    dataset = _build_dataset(cfg_data, split, can_download=True)
    if cfg_data.db.name == "LMDB":
        from .lmdb_datasets import LMDBDataset  # this also depends on py-lmdb, that's why it's a lazy import

        dataset = LMDBDataset(dataset, cfg_data, split, can_create=True)

    if cfg_data.caching:
        dataset = CachedDataset(dataset, num_workers=cfg_impl.threads, pin_memory=cfg_impl.pin_memory)

    if dryrun:
        dataset = torch.utils.data.Subset(dataset, torch.arange(0, cfg_data.batch_size))

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
    cfg_data.path = os.path.expanduser(cfg_data.path)
    if cfg_data.name == "CIFAR10":
        dataset = torchvision.datasets.CIFAR10(
            root=cfg_data.path,
            train=split == "training",
            download=can_download,
            transform=torchvision.transforms.ToTensor(),
        )
        dataset.lookup = dict(zip(list(range(len(dataset))), dataset.targets))
    elif cfg_data.name == "CIFAR100":
        dataset = torchvision.datasets.CIFAR100(
            root=cfg_data.path,
            train=split == "training",
            download=can_download,
            transform=torchvision.transforms.ToTensor(),
        )
        dataset.lookup = dict(zip(list(range(len(dataset))), dataset.targets))
    elif cfg_data.name == "ImageNet":
        dataset = torchvision.datasets.ImageNet(
            root=cfg_data.path,
            split="train" if "train" in split else "val",
            transform=torchvision.transforms.ToTensor(),
        )
        dataset.lookup = dict(zip(list(range(len(dataset))), [label for (_, label) in dataset.samples]))
    elif cfg_data.name == "TinyImageNet":
        dataset = TinyImageNet(
            root=cfg_data.path,
            split=split,
            download=can_download,
            transform=torchvision.transforms.ToTensor(),
            cached=True,
        )
        dataset.lookup = dict(zip(list(range(len(dataset))), dataset.targets))
    elif cfg_data.name == "Birdsnap":
        dataset = Birdsnap(
            root=cfg_data.path, split=split, download=can_download, transform=torchvision.transforms.ToTensor()
        )
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
        dataset = torch.utils.data.Subset(dataset, torch.arange(0, cfg_data.size))

    return dataset


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
