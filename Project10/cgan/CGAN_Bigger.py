# CODE FROM:
# https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cgan/cgan.py
import argparse
import os
import numpy as np
import math
from torchinfo import summary

import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import os

from PIL import Image
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--n_classes", type=int, default=10, help="number of classes for dataset")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=100, help="interval between image sampling")
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)

# OURS
opt.n_classes = 11
opt.n_epochs = 200000

CHANNELS_G = 256
CHANNELS_D = 64

LR_G = opt.lr
LR_D = opt.lr / 10

w = 320
h = 243

w = 160
h = 120

# w = 80
# h = 60

# w = 40
# h = 30

# w = 20
# h = 15


w, h = h, w

img_shape = (opt.channels, w, h)
opt.img_size = (w, h)
# /OURS

cuda = True if torch.cuda.is_available() else False
num_workers = 3 if cuda else 0


class ConvGenerator(nn.Module):
    def __init__(self):
        super(ConvGenerator, self).__init__()

        self.label_emb = nn.Embedding(opt.n_classes, 50)
        self.label_part = nn.Linear(50, 20 * 15)

        self.latent_part = nn.Linear(opt.latent_dim, 20 * 15 * 16)
        self.upscale_features = nn.Conv2d(16, CHANNELS_G, kernel_size=1, stride=1)

        self.model = nn.Sequential(

                nn.ConvTranspose2d(CHANNELS_G + 1, CHANNELS_G, kernel_size=(4, 4), stride=(2, 2), padding=1),  # 40, 30
                nn.LeakyReLU(),

                nn.ConvTranspose2d(CHANNELS_G, CHANNELS_G, kernel_size=(4, 4), stride=(2, 2), padding=1),  # 80, 60
                nn.LeakyReLU(),

                nn.ConvTranspose2d(CHANNELS_G, CHANNELS_G, kernel_size=(4, 4), stride=(2, 2), padding=1),  # 160, 120
                nn.LeakyReLU(),

                nn.Conv2d(CHANNELS_G, 1, kernel_size=(7, 7), padding=3),
                nn.Tanh()
        )

    def process_latent(self, noise):
        small_featuremap = F.leaky_relu(self.latent_part(noise)).view(-1, 16, 15, 20)
        return self.upscale_features(small_featuremap)

    def forward(self, noise, labels):
        lbl_2d = self.label_part(self.label_emb(labels)).view(-1, 1, 15, 20)
        latent_2d = self.process_latent(noise)
        concat = torch.hstack([lbl_2d, latent_2d])
        return self.model(concat)


class ConvDiscriminator(nn.Module):
    def __init__(self):
        super(ConvDiscriminator, self).__init__()

        self.label_embedding = nn.Embedding(opt.n_classes, 50)
        self.label_part = nn.Linear(50, w * h)  # TODO nur hochskalieren?

        self.model = nn.Sequential(
                nn.Conv2d(2, CHANNELS_D, kernel_size=(4, 4), stride=(2, 2), padding=1),  # 80, 60
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(CHANNELS_D, CHANNELS_D, kernel_size=(4, 4), stride=(2, 2), padding=1),  # 40, 30
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(CHANNELS_D, CHANNELS_D, kernel_size=(4, 4), stride=(2, 2), padding=1),  # 20, 15
                nn.LeakyReLU(0.2, inplace=True),

                nn.Flatten(),  # 20 * 15 * 64
                nn.Dropout(),

                nn.Linear(20 * 15 * CHANNELS_D, 1),
                nn.Sigmoid()
        )

    def forward(self, img, labels):
        lbl_2d = self.label_part(self.label_embedding(labels)).view(-1, 1, w, h)
        concat = torch.hstack([lbl_2d, img])  # 2, 60, 80
        return self.model(concat)


# Loss functions
adversarial_loss = torch.nn.MSELoss()

# Initialize generator and discriminator
generator = ConvGenerator()
discriminator = ConvDiscriminator()

print("Generator:")
summary(generator)
print("Discriminator:")
summary(discriminator)

# generator = Generator()
# discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

DATASET = "faces"
# DATASET = "mnist"

if DATASET == "faces":

    # subject01.glasses.gif deleted, because it seems to be a duplicate of subject01.glasses and does not match the naming scheme.
    # subject01.gif renamed to subject01.centerlight, because it does not match the naming scheme and subject01 does not have a centerlight shot.
    class YaleFacesDataset(Dataset):
        def __init__(self, transform=None):
            self.img_dir = "yalefaces"
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


    dataset = YaleFacesDataset(transform=transforms.Compose(
            [
                    ToPILImage(),
                    transforms.Resize(opt.img_size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5])
            ]
    ), )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=num_workers)

elif DATASET == "mnist":
    # Configure data loader
    os.makedirs("../../data/mnist", exist_ok=True)
    dataloader = torch.utils.data.DataLoader(
            datasets.MNIST(
                    "../../data/mnist",
                    train=True,
                    download=True,
                    transform=transforms.Compose(
                            [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
                    ),
            ),
            batch_size=opt.batch_size,
            shuffle=True,
    )

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=LR_G, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=LR_D, betas=(opt.b1, opt.b2))

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


def sample_image(n_row, batches_done):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Sample noise
    z = Variable(FloatTensor(np.random.normal(0, 1, (n_row * opt.n_classes, opt.latent_dim))))
    # Get labels ranging from 0 to n_classes for n rows
    labels = np.array([num for num in range(n_row) for _ in range(opt.n_classes)])
    labels = Variable(LongTensor(labels))
    gen_imgs = generator(z, labels)
    save_image(gen_imgs.data, "images/%d.png" % batches_done, nrow=opt.n_classes, normalize=True)


# ----------
#  Training
# ----------

for epoch in range(opt.n_epochs):
    for i, (imgs, labels) in enumerate(dataloader):

        batch_size = imgs.shape[0]

        # Adversarial ground truths
        valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
        fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(FloatTensor))
        labels = Variable(labels.type(LongTensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise and labels as generator input
        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
        gen_labels = Variable(LongTensor(np.random.randint(0, opt.n_classes, batch_size)))

        # Generate a batch of images
        gen_imgs = generator(z, gen_labels)

        # Loss measures generator's ability to fool the discriminator
        validity = discriminator(gen_imgs, gen_labels)
        g_loss = adversarial_loss(validity, valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Loss for real images
        validity_real = discriminator(real_imgs, labels)
        d_real_loss = adversarial_loss(validity_real, valid)

        # Loss for fake images
        validity_fake = discriminator(gen_imgs.detach(), gen_labels)
        d_fake_loss = adversarial_loss(validity_fake, fake)

        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            sample_image(n_row=10, batches_done=batches_done)
