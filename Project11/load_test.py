import csv
import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co


class TestGTSRBData(Dataset):

    def __init__(self, file_label_pairs, transform=None):
        super().__init__()
        self.image_labels = []
        self.transform = transform
        for (file, label) in file_label_pairs:
            with Image.open(file) as img:
                if self.transform:
                    img = self.transform(img)
                label = torch.tensor(int(label))
                self.image_labels.append((img, label))

    def __len__(self):
        return len(self.image_labels)

    def __getitem__(self, index) -> T_co:
        return self.image_labels[index]


def load_test_data(transform=None):
    with open(os.path.abspath("test/GT-final_test.csv")) as infos:
        annotations = [(os.path.abspath(f"test/{line[0]}"), line[-1])
                       for line in csv.reader(infos, delimiter=";")][1:]
    return TestGTSRBData(annotations, transform=transform)


load_test_data()
