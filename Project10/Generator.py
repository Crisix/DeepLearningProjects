import os
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from pytorch_lightning import LightningModule, Trainer
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage


class MyGenerator(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super().__init__()
        self.img_shape = img_shape

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        # self.model = nn.Sequential(
        #         *block(latent_dim, 128, normalize=False),
        #         *block(128, 256),
        #         *block(256, 512),
        #         *block(512, 1024),
        #         nn.Linear(1024, int(np.prod(img_shape))),
        #         nn.Tanh(),
        # )
        self.model = nn.Sequential(
                *block(latent_dim, 256, normalize=False),
                *block(256, 512),
                *block(512, 2048),
                *block(2048, 32 * 15 * 20),  # 4800
                nn.Unflatten(1, (32, 15, 20)),

                nn.ConvTranspose2d(32, 32, kernel_size=4, padding=1, stride=2),
                nn.BatchNorm2d(32, 0.8),
                nn.LeakyReLU(0.2, inplace=True),

                nn.ConvTranspose2d(32, 16, kernel_size=4, padding=1, stride=2),
                nn.BatchNorm2d(16, 0.8),
                nn.LeakyReLU(0.2, inplace=True),

                nn.ConvTranspose2d(16, 8, kernel_size=4, padding=1, stride=2),
                nn.BatchNorm2d(8, 0.8),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(8, 1, kernel_size=3, padding=1, stride=1),
                nn.Tanh(),
        )

    def forward(self, z):
        return self.model(z)


class Generator(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super().__init__()
        self.img_shape = img_shape

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
                *block(latent_dim, 128, normalize=False),
                *block(128, 256),
                *block(256, 512),
                *block(512, 1024),
                nn.Linear(1024, int(np.prod(img_shape))),
                nn.Tanh(),
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img
