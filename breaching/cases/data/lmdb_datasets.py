"""LMBD dataset wrap an existing dataset and create a database if necessary."""

import os
import io
import pickle
import copy

import platform
import lmdb

import torch
import torchvision
import numpy as np
from PIL import Image

from .data_preparation import _parse_data_augmentations
import logging
log = logging.getLogger(__name__)

class LMDBDataset(torch.utils.data.Dataset):
    """Implement LMDB caching and access.

    Based on https://github.com/pytorch/vision/blob/master/torchvision/datasets/lsun.py
    and
    https://github.com/Lyken17/Efficient-PyTorch/blob/master/tools/folder2lmdb.py
    """

    def __init__(self, dataset, cfg_db, name='train', can_create=True):
        """Initialize with a given pytorch dataset."""
        if os.path.isfile(os.path.expanduser(cfg_db.path)):
            raise ValueError('LMDB path must lead to a folder containing the databases, not a file.')
        self.dataset = dataset
        self.img_shape = self.dataset[0][0].shape

        self.live_transform = copy.deepcopy(dataset.transform)
        if self.live_transform is not None:
            if isinstance(self.live_transform.transforms[0], torchvision.transforms.ToTensor):
                self.skip_pillow = True
                self.live_transform.transforms.pop(0)
            else:
                self.skip_pillow = False
        else:
            self.skip_pillow = True

        shuffled = 'shuffled' if cfg_db.shuffle_while_writing else ''
        full_name = name + ''.join([l for l in repr(cfg_db.augmentations_train) if l.isalnum()]) + f'R{cfg_db.rounds}' + shuffled

        self.path = os.path.join(os.path.expanduser(cfg_db.path), f'{type(dataset).__name__}_{full_name}.lmdb')



        if cfg_db.rebuild_existing_database:
            if os.path.isfile(self.path):
                os.remove(self.path)
                os.remove(self.path + '-lock')

        # Load or create database
        if os.path.isfile(self.path):
            log.info(f'Reusing cached database at {self.path}.')
        else:
            if not can_create:
                raise ValueError(f'No database found at {self.path}. Database creation forbidden in this setting.')
            os.makedirs(os.path.expanduser(cfg_db.path), exist_ok=True)
            log.info(f'Creating database at {self.path}. This may take some time ...')

            if name == 'train':
                repeat_dataset = True
            else:
                repeat_dataset = False
            checksum = create_database(self.dataset, self.path, cfg_db, repeat_dataset)

        # Setup database
        self.cfg = cfg_db
        self.db = lmdb.open(self.path, subdir=False, max_readers=cfg_db.max_readers, readonly=True, lock=False,
                            readahead=cfg_db.readahead, meminit=cfg_db.meminit, max_spare_txns=cfg_db.max_spare_txns)
        self.access = cfg_db.access

        with self.db.begin(write=False) as txn:
            try:
                self.length = pickle.loads(txn.get(b'__len__'))
                self.keys = pickle.loads(txn.get(b'__keys__'))
                self.labels = pickle.loads(txn.get(b'__labels__'))
            except TypeError:
                raise ValueError(f'The provided LMDB dataset at {self.path} is unfinished or damaged.')

        if self.access == 'cursor':
            self._init_cursor()

    def __getstate__(self):
        state = self.__dict__
        state["db"] = None
        if self.access == 'cursor':
            state['_txn'] = None
            state['cursor'] = None
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        # Regenerate db handle after pickling:
        self.db = lmdb.open(self.path, subdir=False, max_readers=self.cfg.max_readers, readonly=True, lock=False,
                            readahead=self.cfg.readahead, meminit=self.cfg.meminit, max_spare_txns=self.cfg.max_spare_txns)
        if self.access == 'cursor':
            self._init_cursor()

    def _init_cursor(self):
        """Initialize cursor position."""
        self._txn = self.db.begin(write=False)
        self.cursor = self._txn.cursor()
        self.cursor.first()
        self.internal_index = 0


    def __getattr__(self, name):
        """Call this only if all attributes of Subset are exhausted."""
        return getattr(self.dataset, name)

    def __len__(self):
        """Draw length from target dataset."""
        return self.length

    def __getitem__(self, index):
        """Get from database. This is either unordered or cursor access for now.

        Future: Write this class as a proper https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset
        """
        img, target = None, None

        if self.access == 'cursor':
            index_key = u'{}'.format(index).encode('ascii')
            if index_key != self.cursor.key():
                self.cursor.set_key(index_key)

            byteflow = self.cursor.value()
            self.cursor.next()

        else:
            with self.db.begin(write=False) as txn:
                byteflow = txn.get(self.keys[index])

        # buffer magic
        buffer = io.BytesIO()
        buffer.write(byteflow)
        buffer.seek(0)

        img = np.reshape(np.frombuffer(buffer.read(), dtype=np.uint8), self.img_shape)
        if not self.skip_pillow:
            img = Image.fromarray(img.transpose(1, 2, 0))
        else:
            img = torch.from_numpy(img.astype('float')) / 255
        if self.live_transform is not None:
            img = self.live_transform(img)

        # load label
        label = self.labels[index]

        return img, label


def create_database(dataset, database_path, cfg_db, repeat_dataset):
    """Create an LMDB database from the given pytorch dataset.

    https://github.com/Lyken17/Efficient-PyTorch/blob/master/tools/folder2lmdb.py

    Removed pyarrow dependency
    """
    sample_transforms, _ = _parse_data_augmentations(cfg_db, PIL_only=True)
    data_transform = dataset.transform
    dataset.transform = sample_transforms

    if platform.system() == 'Linux':
        map_size = 1099511627776 * 2  # Linux can grow memory as needed.
    else:
        raise ValueError('Provide a reasonable default map_size for your operating system.')
    db = lmdb.open(database_path, subdir=False,
                   map_size=map_size, readonly=False,
                   meminit=cfg_db.meminit, map_async=True)
    txn = db.begin(write=True)

    labels = []


    iterations = cfg_db.rounds if repeat_dataset else 1
    idx = 0
    for round in range(iterations):
        if cfg_db.shuffle_while_writing:
            order = torch.randperm(len(dataset)).tolist()
        else:
            order = torch.arange(0, len(dataset))
        for indexing in order:
            img, label = dataset[indexing]
            labels.append(label)
            # serialize
            byteflow = np.asarray(img, dtype=np.uint8).transpose(2, 0, 1).tobytes()
            txn.put(u'{}'.format(idx).encode('ascii'), byteflow)
            idx += 1

            if idx % cfg_db.write_frequency == 0:
                log.info(f"[{idx} / {len(dataset) * iterations}]")
                txn.commit()
                txn = db.begin(write=True)

    # finalize dataset
    txn.commit()
    keys = [u'{}'.format(k).encode('ascii') for k in range(idx)]
    with db.begin(write=True) as txn:
        txn.put(b'__keys__', pickle.dumps(keys))
        txn.put(b'__labels__', pickle.dumps(labels))
        txn.put(b'__len__', pickle.dumps(len(keys)))
    log.info(f'Database written successfully with {len(keys)} entries.')
    dataset.transform = data_transform
