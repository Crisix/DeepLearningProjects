import numpy as np
import torch.nn as nn


class MyDiscriminator(nn.Module):
    def __init__(self, img_shape, cat_ctrl_vars):
        super().__init__()

        self.model = nn.Sequential(

                nn.Conv2d(1, 32, kernel_size=3, padding=1, stride=1),
                nn.MaxPool2d(kernel_size=2),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1),
                nn.MaxPool2d(kernel_size=2),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(32, 16, kernel_size=3, padding=1, stride=1),
                nn.MaxPool2d(kernel_size=2),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Flatten(),

                nn.Linear(16 * (img_shape[1] // 8) * (img_shape[2] // 8), 1024),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Linear(1024, 256),
                nn.LeakyReLU(0.2, inplace=True),
        )

        self.head = nn.Sequential(
                nn.Linear(256, 1),
                nn.Sigmoid(),
        )

        self.Q_net = nn.Sequential(
                nn.Linear(256, cat_ctrl_vars),
                nn.Softmax(),
        )

    # def forward(self, img):
    #     return self.head(self.model(img))

    def forward(self, img):
        intermediate = self.model(img)
        return self.head(intermediate), self.Q_net(intermediate)


class Discriminator(nn.Module):
    def __init__(self, img_shape, cat_ctrl_vars):
        super().__init__()

        self.model = nn.Sequential(
                nn.Linear(int(np.prod(img_shape)), 512),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(512, 256),
                nn.LeakyReLU(0.2, inplace=True),
        )

        self.head = nn.Sequential(
                nn.Linear(256, 1),
                nn.Sigmoid(),
        )

        self.Q_net = nn.Sequential(
                nn.Linear(256, cat_ctrl_vars),
                nn.Softmax(),
        )

    def forward(self, img):
        intermediate = self.model(img.view(img.size(0), -1))
        return self.head(intermediate), self.Q_net(intermediate)

        # img_flat = img.view(img.size(0), -1)
        # validity = self.model(img_flat)
        # return validity
