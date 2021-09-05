"""Additional torchvision-like datasets."""

import torch
import os
import glob
from PIL import Image

from torchvision.datasets.utils import download_and_extract_archive
import hashlib


class TinyImageNet(torch.utils.data.Dataset):
    """Tiny ImageNet data set available from `http://cs231n.stanford.edu/tiny-imagenet-200.zip`.

    This is a TinyImageNet variant to the code of Meng Lee, mnicnc404 / Date: 2018/06/04
    References:
        - https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel.html
    Parameters
    ----------
    root: string
        Root directory including `train`, `test` and `val` subdirectories.
    split: string
        Indicating which split to return as a data set.
        Valid option: [`train`, `test`, `val`]
    transform: torchvision.transforms
        A (series) of valid transformation(s).
    cached: bool
        Set to True if there is enough memory (about 5G) and want to minimize disk IO overhead.
    download: bool
        Set to true to automatically download the dataset in to the root folder.
    """

    EXTENSION = 'JPEG'
    NUM_IMAGES_PER_CLASS = 500
    CLASS_LIST_FILE = 'wnids.txt'
    VAL_ANNOTATION_FILE = 'val_annotations.txt'
    CLASSES = 'words.txt'

    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    archive = "tiny-imagenet-200.zip"
    folder = "tiny-imagenet-200"
    train_md5 = 'c77c61d662a966d2fcae894d82df79e4'
    val_md5 = 'cef44e3f1facea2ea8cd5e5a7a46886c'
    test_md5 = 'bc72ebd5334b12e3a7ba65506c0f8bc0'

    def __init__(self, root, split='train', transform=None, target_transform=None, cached=True, download=True):
        """Init with split, transform, target_transform."""
        self.root = os.path.expanduser(root)
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.cached = cached

        self.split_dir = os.path.join(root, self.folder, self.split)
        self.image_paths = sorted(glob.iglob(os.path.join(self.split_dir, '**', '*.%s' % self.EXTENSION), recursive=True))
        self.labels = {}  # fname - label number mapping

        if download:
            self.download()

        self._parse_labels()

        if self.cached:
            self._build_cache()

    def _check_integrity(self):
        """This only checks if all files are there."""
        string_rep = ''.join(self.image_paths).encode('utf-8')
        hash = hashlib.md5(string_rep)
        if self.split == 'train':
            return hash.hexdigest() == self.train_md5
        elif self.split == 'val':
            return hash.hexdigest() == self.val_md5
        else:
            return hash.hexdigest() == self.test_md5

    def download(self):
        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        download_and_extract_archive(self.url, self.root, filename=self.archive)

    def _parse_labels(self):
        with open(os.path.join(self.root, self.folder, self.CLASS_LIST_FILE), 'r') as fp:
            self.label_texts = sorted([text.strip() for text in fp.readlines()])
        self.label_text_to_number = {text: i for i, text in enumerate(self.label_texts)}

        if self.split == 'train':
            for label_text, i in self.label_text_to_number.items():
                for cnt in range(self.NUM_IMAGES_PER_CLASS):
                    self.labels['%s_%d.%s' % (label_text, cnt, self.EXTENSION)] = i
        elif self.split == 'val':
            with open(os.path.join(self.split_dir, self.VAL_ANNOTATION_FILE), 'r') as fp:
                for line in fp.readlines():
                    terms = line.split('\t')
                    file_name, label_text = terms[0], terms[1]
                    self.labels[file_name] = self.label_text_to_number[label_text]

        # Build class names
        label_text_to_word = dict()
        with open(os.path.join(self.root, self.folder, self.CLASSES), 'r') as file:
            for line in file:
                label_text, word = line.split('\t')
                label_text_to_word[label_text] = word.split(',')[0].rstrip('\n')
        self.classes = [label_text_to_word[label] for label in self.label_texts]

        # Prepare index - label mapping
        self.targets = [self.labels[os.path.basename(file_path)] for file_path in self.image_paths]

    def _build_cache(self):
        """Cache images in RAM."""
        self.cache = []
        for index in range(len(self)):
            img = Image.open(self.image_paths[index])
            img = img.convert("RGB")
            self.cache.append(img)

    def __len__(self):
        """Return length via image paths."""
        return len(self.image_paths)

    def __getitem__(self, index):
        """Return image, label."""
        if self.cached:
            img = self.cache[index]
        else:
            img = Image.open(self.image_paths[index])
            img = img.convert("RGB")
        target = self.targets[index]

        img = self.transform(img) if self.transform else img
        target = self.target_transform(target) if self.target_transform else target
        if self.split == 'test':
            return img, None
        else:
            return img, target
