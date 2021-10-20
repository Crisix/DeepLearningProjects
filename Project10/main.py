# CODE IS FROM/BASED FROM:
# https://pytorch-lightning.readthedocs.io/en/latest/notebooks/lightning_examples/basic-gan.html

from collections import OrderedDict

"""
Differences:

eriklindernoren/PyTorch-GAN: 
    3 optimizers: D, G, Q

Natsu6767/InfoGAN-PyTorch:
    2 optimizers: D, (G+Q)

"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
from pytorch_lightning import LightningModule, Trainer
from torch.distributions import OneHotCategorical
from torch.nn import CrossEntropyLoss
from torchvision.transforms import ToPILImage
import torchvision.datasets as datasets
from torchvision.datasets import MNIST
import os
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST

from Project10.Discriminator import MyDiscriminator, Discriminator
from Project10.Generator import MyGenerator, Generator
# from Project10.YaleFacesDataset import YaleFacesDataset
from Project10.MnistDataset import MNISTDataModule
from Project10.YaleFacesDataset import YaleFacesDataset

DATASET = "mnist"
# DATASET = "faces"

AVAIL_GPUS = min(1, torch.cuda.device_count())


class GAN(LightningModule):
    def __init__(
            self,
            channels,
            width,
            height,
            cat_ctrl_vars=None,
            lmbda=1.0,
            latent_dim: int = 100,
            lr: float = 0.0002,
            b1: float = 0.5,
            b2: float = 0.999,
            batch_size: int = 1,
            **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()

        # networks
        data_shape = (channels, width, height)

        self.categorical_distribution = OneHotCategorical(torch.tensor([1] * cat_ctrl_vars))

        latent = self.hparams.latent_dim + cat_ctrl_vars
        # self.generator = MyGenerator(latent_dim=latent, img_shape=data_shape)  # 20.9 M
        # self.discriminator = MyDiscriminator(img_shape=data_shape, cat_ctrl_vars=cat_ctrl_vars)  # 5.2 M

        self.generator = Generator(latent_dim=latent, img_shape=data_shape)  # 20.4 M
        self.discriminator = Discriminator(img_shape=data_shape, cat_ctrl_vars=cat_ctrl_vars)  # 10.0 M

        # self.validation_z = torch.randn(8, self.hparams.latent_dim)
        rows = 8
        self.num_classes = cat_ctrl_vars
        z = self.generate_latents(rows)[0]

        self.validation_z = torch.hstack([z.repeat_interleave(cat_ctrl_vars, 0),
                                          torch.eye(cat_ctrl_vars).repeat(rows, 1)])

        # self.example_input_array = torch.zeros(2, self.hparams.latent_dim)

    def forward(self, z):
        return self.generator(z)

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def categorical_loss(self, y_hat, y):
        return F.cross_entropy(y_hat, y)

    def generate_latents(self, batch_size):
        z = torch.randn(batch_size, self.hparams.latent_dim)  # latent space: sample noise
        # z = z.type_as(real_imgs)
        c = self.categorical_distribution.sample([batch_size])  # latent space: sample categorical
        c_idx = torch.argmax(c, dim=1)
        return z.to(self.device), c.to(self.device), c_idx.to(self.device)

    def training_step(self, batch, batch_idx, optimizer_idx):
        real_imgs, _ = batch

        if optimizer_idx == 0:  # train generator
            return self.train_generator(real_imgs)
        if optimizer_idx == 1:  # train discriminator
            return self.train_discriminator(real_imgs)
        if optimizer_idx == 2:
            return self.train_all(real_imgs)

    def train_all(self, real_imgs):

        z, c, c_idx = self.generate_latents(real_imgs.shape[0])
        cz = torch.hstack([c, z])

        fake_imgs = self.generator(cz)  # .detach()  KEIN DETACH HIER
        _, cat_pred = self.discriminator(fake_imgs)

        q_loss = self.categorical_loss(cat_pred, c_idx)
        self.log("q_loss", q_loss, prog_bar=True, on_step=True)
        return q_loss

    def train_discriminator(self, real_imgs):

        z, c, c_idx = self.generate_latents(real_imgs.shape[0])
        cz = torch.hstack([c, z])

        # Measure discriminator's ability to classify real from generated samples
        valid = torch.ones(real_imgs.size(0), 1).type_as(real_imgs)
        d_pred, _ = self.discriminator(real_imgs)
        real_loss = self.adversarial_loss(d_pred, valid)

        fake = torch.zeros(real_imgs.size(0), 1).type_as(real_imgs)
        fake_imgs = self.generator(cz).detach()  # dont train generator -> detach
        d_pred, cat_pred = self.discriminator(fake_imgs)
        fake_loss = self.adversarial_loss(d_pred, fake)

        d_loss = (real_loss + fake_loss) / 2
        q_loss = self.categorical_loss(cat_pred, c_idx)  # TODO IS THIS RIGHT?!?!
        # loss = d_loss + self.hparams.lmbda * q_loss
        loss = d_loss  # TODO -> so müsste InfoGAN selber entscheiden, was auf welchen Index kommt

        self.log("d_loss", d_loss, prog_bar=True, on_step=True)
        return loss

    def train_generator(self, real_imgs):

        z, c, c_idx = self.generate_latents(real_imgs.shape[0])
        cz = torch.hstack([c, z])

        # ground truth result (ie: all fake)
        valid = torch.ones(real_imgs.size(0), 1).type_as(real_imgs)
        fake_images = self.generator(cz)
        # TODO sollte der discriminator hier nicht fixed sein?
        d_pred, cat_pred = self.discriminator(fake_images)

        g_loss = self.adversarial_loss(d_pred, valid)
        # q_loss = self.categorical_loss(cat_pred, c_idx)
        loss = g_loss  # + self.hparams.lmbda * q_loss

        self.log("g_loss", g_loss, prog_bar=True, on_step=True)
        return loss

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        # opt_g = torch.optim.Adam([{'params': self.generator.parameters()},
        #                           {'params': self.discriminator.Q_net.parameters()}],  # min G, Q
        #                          lr=lr, betas=(b1, b2))
        #
        # opt_d = torch.optim.Adam([{'params': self.discriminator.model.parameters()},
        #                           {'params': self.discriminator.head.parameters()}],  # max D
        #                          lr=lr, betas=(b1, b2))

        opt_g = torch.optim.Adam([{'params': self.generator.parameters()}],
                                 lr=lr, betas=(b1, b2))

        opt_d = torch.optim.Adam([{'params': self.discriminator.parameters()}],
                                 lr=lr, betas=(b1, b2))

        opt_all = torch.optim.Adam([{'params': self.discriminator.parameters()},
                                    {'params': self.generator.parameters()}],
                                   lr=lr, betas=(b1, b2))

        return [opt_g, opt_d, opt_all], []

    def on_epoch_end(self):
        if self.current_epoch % (50 if DATASET == "faces" else 15) == 0:
            z = self.validation_z.type_as(self.generator.model[0].weight)

            # log sampled images
            sample_imgs = self(z)
            grid = torchvision.utils.make_grid(sample_imgs, range=(-1, 1), nrow=self.num_classes)
            self.logger.experiment.add_image("generated_images", grid, self.current_epoch)

            # Copied from https://www.programcreek.com/python/?code=lukasruff%2FDeep-SAD-PyTorch%2FDeep-SAD-PyTorch-master%2Fsrc%2Futils%2Fvisualization%2Fplot_images_grid.py
            npgrid = grid.cpu().detach().numpy()
            plt.figure(figsize=(25, 25))
            plt.imshow(np.transpose(npgrid, (1, 2, 0)), interpolation='nearest')
            ax = plt.gca()
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)

            plt.savefig(f"generated_images/{self.current_epoch}.png", bbox_inches='tight', pad_inches=0.1)
            plt.clf()


if DATASET == "faces":
    transform = transforms.Compose([
            ToPILImage(),
            # transforms.Resize((243 // 4, 320 // 4)),  # -> (60, 80)
            # transforms.Resize((120, 160)),  # /2
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
    ])
    dataset = YaleFacesDataset(transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    image_dim = dataset[0][0].shape
    cat_ctrl_vars = dataset[0][1].shape[0]

    model = GAN(*image_dim, cat_ctrl_vars=cat_ctrl_vars)
    trainer = Trainer(gpus=AVAIL_GPUS, max_epochs=20000)
    trainer.fit(model, dataloader)

elif DATASET == "mnist":
    dm = MNISTDataModule()
    model = GAN(*dm.size(), cat_ctrl_vars=10, lmbda=0)
    trainer = Trainer(gpus=AVAIL_GPUS, max_epochs=20000)
    trainer.fit(model, dm)
