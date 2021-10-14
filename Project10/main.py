# CODE IS FROM/BASED FROM:
# https://pytorch-lightning.readthedocs.io/en/latest/notebooks/lightning_examples/basic-gan.html

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

dataroot = "yalefaces"

AVAIL_GPUS = min(1, torch.cuda.device_count())


# subject01.glasses.gif deleted, because it seems to be a duplicate of subject01.glasses and does not match the naming scheme.
# subject01.gif renamed to subject01.centerlight, because it does not match the naming scheme and subject01 does not have a centerlight shot.
class YaleFacesDataset(Dataset):
    def __init__(self, transform=None):
        self.img_dir = "Project10/yalefaces"
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

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        return self.transform(self.imgs[idx]).unsqueeze(0), self.labels[idx]


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


class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super().__init__()

        self.model = nn.Sequential(
                nn.Linear(int(np.prod(img_shape)), 512),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(512, 256),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(256, 1),
                nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity


class GAN(LightningModule):
    def __init__(
            self,
            channels,
            width,
            height,
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
        self.generator = Generator(latent_dim=self.hparams.latent_dim, img_shape=data_shape)
        self.discriminator = Discriminator(img_shape=data_shape)

        self.validation_z = torch.randn(8, self.hparams.latent_dim)

        self.example_input_array = torch.zeros(2, self.hparams.latent_dim)

    def forward(self, z):
        return self.generator(z)

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def training_step(self, batch, batch_idx, optimizer_idx):
        imgs, _ = batch

        # sample noise
        z = torch.randn(imgs.shape[0], self.hparams.latent_dim)
        z = z.type_as(imgs)

        # train generator
        if optimizer_idx == 0:
            # generate images
            self.generated_imgs = self(z)

            # log sampled images
            sample_imgs = self.generated_imgs[:6]
            grid = torchvision.utils.make_grid(sample_imgs)
            self.logger.experiment.add_image("generated_images", grid, 0)

            # ground truth result (ie: all fake)
            # put on GPU because we created this tensor inside training_loop
            valid = torch.ones(imgs.size(0), 1)
            valid = valid.type_as(imgs)

            # adversarial loss is binary cross-entropy
            g_loss = self.adversarial_loss(self.discriminator(self(z)), valid)
            tqdm_dict = {"g_loss": g_loss}
            output = OrderedDict({"loss": g_loss, "progress_bar": tqdm_dict, "log": tqdm_dict})
            return output

        # train discriminator
        if optimizer_idx == 1:
            # Measure discriminator's ability to classify real from generated samples

            # how well can it label as real?
            valid = torch.ones(imgs.size(0), 1)
            valid = valid.type_as(imgs)

            real_loss = self.adversarial_loss(self.discriminator(imgs), valid)

            # how well can it label as fake?
            fake = torch.zeros(imgs.size(0), 1)
            fake = fake.type_as(imgs)

            fake_loss = self.adversarial_loss(self.discriminator(self(z).detach()), fake)

            # discriminator loss is the average of these
            d_loss = (real_loss + fake_loss) / 2
            tqdm_dict = {"d_loss": d_loss}
            output = OrderedDict({"loss": d_loss, "progress_bar": tqdm_dict, "log": tqdm_dict})
            return output

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return [opt_g, opt_d], []

    def on_epoch_end(self):
        z = self.validation_z.type_as(self.generator.model[0].weight)

        # log sampled images
        sample_imgs = self(z)
        grid = torchvision.utils.make_grid(sample_imgs)
        self.logger.experiment.add_image("generated_images", grid, self.current_epoch)

        # Copied from https://www.programcreek.com/python/?code=lukasruff%2FDeep-SAD-PyTorch%2FDeep-SAD-PyTorch-master%2Fsrc%2Futils%2Fvisualization%2Fplot_images_grid.py
        npgrid = grid.cpu().detach().numpy()
        plt.imshow(np.transpose(npgrid, (1, 2, 0)), interpolation='nearest')
        ax = plt.gca()
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

        plt.savefig(f"Project10/generated_images/{str(self.current_epoch)}.jpg", bbox_inches='tight', pad_inches=0.1)
        plt.clf()


dataset = YaleFacesDataset(transform=transforms.Compose([
        # transforms.Resize(image_size),
        # transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.Normalize((0.5,), (0.5,)),
]))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=55, shuffle=True)
image_dim = dataset[0][0][0].shape
print(f"image_dim={image_dim}")
model = GAN(*image_dim)
trainer = Trainer(gpus=AVAIL_GPUS, max_epochs=20000, progress_bar_refresh_rate=20, )
trainer.fit(model, dataloader)
