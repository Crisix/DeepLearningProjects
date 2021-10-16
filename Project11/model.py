import torch
import torch.nn as nn
import torchvision.models as models
from pytorch_lightning import LightningModule, Trainer


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

    def forward(self, x):
        return self.classifier(x)


# todo: load dataset, preprocess

model = MyModel()
trainer = Trainer()  # todo
trainer.fit(model)  # dataloader, todo suche template für pytorch (transforms?)
