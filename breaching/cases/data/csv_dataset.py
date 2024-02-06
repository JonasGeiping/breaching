import torch
import logging
import csv
import os
from PIL import Image
from torchvision.datasets.folder import accimage_loader, pil_loader, default_loader

log = logging.getLogger(__name__)

class CsvDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, dir, transform=None):
        self.data = []
        self.fields = []
        self.classes = set()
        self.class_to_idx = {}
        self.samples = []
        with open(csv_path, mode='r') as file:
            csvreader = csv.reader(file)
            self.fields = next(csvreader)
            for row in csvreader:
                if len(row) != 2:
                    raise TypeError(f"Wrong number of fields: Expected 2 fields found {len(row)}")
                img_path, label = row
                if label not in self.classes:
                    self.class_to_idx[label] = len(self.classes)
                    self.classes.add(label)
                self.data.append(row)
                self.samples.append((img_path, self.class_to_idx[label]))
        self.classes = sorted(list(self.classes))
        self.dir = dir
        self.transform = transform
        self.loader = default_loader

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_name = os.path.join(self.dir, self.data[idx][0])
        image = self.loader(img_name)
        #image = Image.open(img_name)
        label = self.data[idx][1]
        if self.transform:
            image = self.transform(image)

        return (image, self.class_to_idx[label])