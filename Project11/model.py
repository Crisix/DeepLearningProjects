import os

import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms
from torchmetrics.functional.classification.accuracy import accuracy


class MyModel(LightningModule):
    def __init__(self):
        super().__init__()
        num_target_classes = 43

        backbone = models.resnet18(pretrained=True)
        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        self.classifier = nn.Sequential(
            *layers,
            nn.Flatten(),
            nn.Linear(num_filters, num_target_classes)
        )

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
        loss = F.cross_entropy(logits, y)
        y_hat = torch.argmax(logits, dim=1)
        accuracy = torch.sum(y == y_hat).item() / (len(y) * 1.0)
        output = dict({
            'val_loss': loss,
            'val_acc': torch.tensor(accuracy),
        })
        self.log("val_acc", accuracy, prog_bar=True)
        return output

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        y_hat = torch.argmax(logits, dim=1)
        # accuracy = torch.sum(y == y_hat).item() / (len(y) * 1.0)
        output = {
            'test_loss': loss,
            'accuracy': accuracy(self(x), y),
        }
        self.log("acc", accuracy(self(x), y), prog_bar=True, on_epoch=True)
        return output

    # def train_dataloader(self):
    #     return self.train_loader
    #
    # def val_dataloader(self):
    #     return self.val_loader

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


if __name__ == '__main__':
    data_transforms = transforms.Compose([
        transforms.Resize([112, 112]),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
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

    BATCH_SIZE = 128
    train_loader = data.DataLoader(train_data,
                                   num_workers=0,
                                   batch_size=BATCH_SIZE)
    val_loader = data.DataLoader(val_data,
                                 num_workers=0,
                                 batch_size=BATCH_SIZE)

    model = MyModel()

    checkpoint = ModelCheckpoint(
        dirpath=os.path.abspath("checkpoints"),
        filename='{v_num}--{epoch}-{val_loss:.2f}-{val_acc:.2f}')

    trainer = Trainer(gpus=1, max_epochs=2, callbacks=checkpoint)
    trainer.fit(model, train_dataloaders=train_loader,
                val_dataloaders=val_loader)
