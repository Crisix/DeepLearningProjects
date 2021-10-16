import os

import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
from pytorch_lightning import LightningModule, Trainer
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms


class MyModel(LightningModule):
    def __init__(self):
        super().__init__()
        num_target_classes = 43

        backbone = models.resnet18(pretrained=True)
        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        self.classifier = nn.Sequential(
            *layers,
            nn.Linear(num_filters, num_target_classes)
        )

        data_transforms = transforms.Compose([
            transforms.Resize([112, 112]),
            transforms.ToTensor()
        ])
        train_data = torchvision.datasets.ImageFolder(
            root=os.path.abspath("train"),
            transform=data_transforms)

        ratio = 0.8
        n_train_examples = int(len(train_data) * ratio)
        n_val_examples = len(train_data) - n_train_examples

        train_data, val_data = data.random_split(train_data,
                                                 [n_train_examples,
                                                  n_val_examples])

        BATCH_SIZE = 42
        self.train_loader = data.DataLoader(train_data,
                                            batch_size=BATCH_SIZE)
        self.val_loader = data.DataLoader(val_data,
                                          batch_size=BATCH_SIZE)

    def forward(self, x):
        return self.classifier(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        return F.cross_entropy(logits, y)
        # return F.nll_loss(logits, y)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        return F.cross_entropy(logits, y)
        #loss = F.nll_loss(logits, y)
        #return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        y_hat = torch.argmax(logits, dim=1)
        accuracy = torch.sum(y == y_hat).item() / (len(y) * 1.0)
        output = dict({
            'test_loss': loss,
            'test_acc': torch.tensor(accuracy),
        })
        return output

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


model = MyModel()
trainer = Trainer(gpus=1, max_epochs=30)
trainer.fit(model)
