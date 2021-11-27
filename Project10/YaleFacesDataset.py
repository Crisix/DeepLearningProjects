import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


# subject01.glasses.gif deleted, because it seems to be a duplicate of subject01.glasses and does not match the naming scheme.
# subject01.gif renamed to subject01.centerlight, because it does not match the naming scheme and subject01 does not have a centerlight shot.
class YaleFacesDataset(Dataset):
    def __init__(self, transform=None):
        self.img_dir = "./yalefaces"
        self.transform = transform
        self.imgs = []
        self.labels = []
        self.subjects = []
        img_paths = os.listdir(self.img_dir)
        for img in img_paths:
            if img == "Readme.txt" or img == ".DS_Store":
                continue
            split = img.split(".")
            self.subjects.append(split[0])
            self.labels.append(split[1])
            self.imgs.append(np.asarray(Image.open(os.path.join(self.img_dir, img))))
        print(f"{len(self.imgs)} images")
        self.onehot_idx = list(set(self.labels))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        return self.transform(self.imgs[idx]), self.onehot_idx.index(self.labels[idx])  # self.labels[idx]
        # torch.nn.functional.one_hot(torch.tensor(self.onehot_idx.index(self.labels[idx])), len(self.onehot_idx))
