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
import torchattacks


class MyModel(LightningModule):
    def __init__(self):
        super().__init__()
        num_target_classes = 43
        self.accs = []

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
            'accuracy': accuracy(y_hat, y),
        }
        self.log("acc", accuracy(y_hat, y), prog_bar=True, on_epoch=True)
        self.accs.append(accuracy(y_hat, y))
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


'''
C:\DeepLearningProjects\venv\Scripts\python.exe C:/DeepLearningProjects/Project11/model.py
C:\DeepLearningProjects\venv\lib\site-packages\pytorch_lightning\callbacks\model_checkpoint.py:446: UserWarning: Checkpoint directory C:\DeepLearningProjects\Project11\checkpoints exists and is not empty.
  rank_zero_warn(f"Checkpoint directory {dirpath} exists and is not empty.")
GPU available: True, used: True
TPU available: False, using: 0 TPU cores
IPU available: False, using: 0 IPUs
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]

  | Name       | Type       | Params
------------------------------------------
0 | classifier | Sequential | 11.2 M
------------------------------------------
11.2 M    Trainable params
0         Non-trainable params
11.2 M    Total params
44.794    Total estimated model params size (MB)
C:\DeepLearningProjects\venv\lib\site-packages\pytorch_lightning\trainer\data_loading.py:105: UserWarning: The dataloader, val dataloader 0, does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` (try 12 which is the number of cpus on this machine) in the `DataLoader` init to improve performance.
  rank_zero_warn(
Epoch 0:   0%|          | 0/308 [00:00<?, ?it/s] C:\DeepLearningProjects\venv\lib\site-packages\pytorch_lightning\trainer\data_loading.py:105: UserWarning: The dataloader, train dataloader, does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` (try 12 which is the number of cpus on this machine) in the `DataLoader` init to improve performance.
  rank_zero_warn(
Epoch 0:  80%|███████▉  | 246/308 [03:53<00:58,  1.06it/s, loss=0.0376, v_num=54]
Validating: 0it [00:00, ?it/s]
Validating:   0%|          | 0/62 [00:00<?, ?it/s]
Epoch 0:  81%|████████  | 248/308 [03:53<00:56,  1.06it/s, loss=0.0376, v_num=54]
Validating:   3%|▎         | 2/62 [00:01<00:51,  1.16it/s]
Validating:   5%|▍         | 3/62 [00:02<00:53,  1.11it/s]
Epoch 0:  81%|████████▏ | 251/308 [03:56<00:53,  1.06it/s, loss=0.0376, v_num=54]
Validating:   8%|▊         | 5/62 [00:04<00:52,  1.08it/s]
Validating:  10%|▉         | 6/62 [00:05<00:51,  1.08it/s]
Epoch 0:  82%|████████▏ | 254/308 [03:59<00:50,  1.06it/s, loss=0.0376, v_num=54]
Validating:  13%|█▎        | 8/62 [00:07<00:51,  1.06it/s]
Validating:  15%|█▍        | 9/62 [00:08<00:49,  1.07it/s]
Epoch 0:  83%|████████▎ | 257/308 [04:02<00:47,  1.06it/s, loss=0.0376, v_num=54]
Validating:  18%|█▊        | 11/62 [00:10<00:47,  1.07it/s]
Validating:  19%|█▉        | 12/62 [00:11<00:47,  1.04it/s]
Epoch 0:  84%|████████▍ | 260/308 [04:05<00:45,  1.06it/s, loss=0.0376, v_num=54]
Validating:  23%|██▎       | 14/62 [00:13<00:45,  1.06it/s]
Validating:  24%|██▍       | 15/62 [00:14<00:44,  1.06it/s]
Epoch 0:  85%|████████▌ | 263/308 [04:08<00:42,  1.06it/s, loss=0.0376, v_num=54]
Validating:  27%|██▋       | 17/62 [00:15<00:42,  1.05it/s]
Validating:  29%|██▉       | 18/62 [00:16<00:41,  1.05it/s]
Epoch 0:  86%|████████▋ | 266/308 [04:10<00:39,  1.06it/s, loss=0.0376, v_num=54]
Validating:  32%|███▏      | 20/62 [00:18<00:40,  1.05it/s]
Validating:  34%|███▍      | 21/62 [00:19<00:39,  1.05it/s]
Epoch 0:  87%|████████▋ | 269/308 [04:13<00:36,  1.06it/s, loss=0.0376, v_num=54]
Validating:  37%|███▋      | 23/62 [00:21<00:37,  1.04it/s]
Validating:  39%|███▊      | 24/62 [00:22<00:36,  1.04it/s]
Epoch 0:  88%|████████▊ | 272/308 [04:16<00:33,  1.06it/s, loss=0.0376, v_num=54]
Validating:  42%|████▏     | 26/62 [00:24<00:34,  1.06it/s]
Validating:  44%|████▎     | 27/62 [00:25<00:33,  1.05it/s]
Epoch 0:  89%|████████▉ | 275/308 [04:19<00:31,  1.06it/s, loss=0.0376, v_num=54]
Validating:  47%|████▋     | 29/62 [00:27<00:32,  1.03it/s]
Validating:  48%|████▊     | 30/62 [00:28<00:30,  1.04it/s]
Epoch 0:  90%|█████████ | 278/308 [04:22<00:28,  1.06it/s, loss=0.0376, v_num=54]
Validating:  52%|█████▏    | 32/62 [00:30<00:28,  1.05it/s]
Validating:  53%|█████▎    | 33/62 [00:31<00:27,  1.04it/s]
Epoch 0:  91%|█████████ | 281/308 [04:25<00:25,  1.06it/s, loss=0.0376, v_num=54]
Validating:  56%|█████▋    | 35/62 [00:33<00:25,  1.04it/s]
Validating:  58%|█████▊    | 36/62 [00:34<00:24,  1.04it/s]
Epoch 0:  92%|█████████▏| 284/308 [04:28<00:22,  1.06it/s, loss=0.0376, v_num=54]
Validating:  61%|██████▏   | 38/62 [00:36<00:23,  1.04it/s]
Validating:  63%|██████▎   | 39/62 [00:37<00:22,  1.03it/s]
Epoch 0:  93%|█████████▎| 287/308 [04:31<00:19,  1.06it/s, loss=0.0376, v_num=54]
Validating:  66%|██████▌   | 41/62 [00:38<00:20,  1.04it/s]
Validating:  68%|██████▊   | 42/62 [00:39<00:19,  1.05it/s]
Epoch 0:  94%|█████████▍| 290/308 [04:33<00:16,  1.06it/s, loss=0.0376, v_num=54]
Validating:  71%|███████   | 44/62 [00:41<00:17,  1.06it/s]
Validating:  73%|███████▎  | 45/62 [00:42<00:15,  1.06it/s]
Epoch 0:  95%|█████████▌| 293/308 [04:36<00:14,  1.06it/s, loss=0.0376, v_num=54]
Validating:  76%|███████▌  | 47/62 [00:44<00:14,  1.06it/s]
Validating:  77%|███████▋  | 48/62 [00:45<00:13,  1.06it/s]
Epoch 0:  96%|█████████▌| 296/308 [04:39<00:11,  1.06it/s, loss=0.0376, v_num=54]
Validating:  81%|████████  | 50/62 [00:47<00:11,  1.06it/s]
Validating:  82%|████████▏ | 51/62 [00:48<00:10,  1.06it/s]
Epoch 0:  97%|█████████▋| 299/308 [04:42<00:08,  1.06it/s, loss=0.0376, v_num=54]
Validating:  85%|████████▌ | 53/62 [00:50<00:08,  1.07it/s]
Validating:  87%|████████▋ | 54/62 [00:51<00:07,  1.07it/s]
Epoch 0:  98%|█████████▊| 302/308 [04:45<00:05,  1.06it/s, loss=0.0376, v_num=54]
Validating:  90%|█████████ | 56/62 [00:53<00:05,  1.05it/s]
Validating:  92%|█████████▏| 57/62 [00:54<00:04,  1.04it/s]
Epoch 0:  99%|█████████▉| 305/308 [04:48<00:02,  1.06it/s, loss=0.0376, v_num=54]
Validating:  95%|█████████▌| 59/62 [00:55<00:02,  1.06it/s]
Validating:  97%|█████████▋| 60/62 [00:56<00:01,  1.05it/s]
Epoch 0: 100%|██████████| 308/308 [04:50<00:00,  1.06it/s, loss=0.0376, v_num=54]
Epoch 0: 100%|██████████| 308/308 [04:51<00:00,  1.06it/s, loss=0.0376, v_num=54, val_acc=0.987]
Epoch 1:  80%|███████▉  | 246/308 [03:46<00:56,  1.09it/s, loss=0.0113, v_num=54, val_acc=0.987]
Validating: 0it [00:00, ?it/s]
Validating:   0%|          | 0/62 [00:00<?, ?it/s]
Epoch 1:  81%|████████  | 249/308 [03:47<00:53,  1.10it/s, loss=0.0113, v_num=54, val_acc=0.987]
Epoch 1:  82%|████████▏ | 252/308 [03:47<00:50,  1.11it/s, loss=0.0113, v_num=54, val_acc=0.987]
Validating:  10%|▉         | 6/62 [00:00<00:05, 10.30it/s]
Epoch 1:  83%|████████▎ | 255/308 [03:47<00:47,  1.12it/s, loss=0.0113, v_num=54, val_acc=0.987]
Validating:  16%|█▌        | 10/62 [00:00<00:05,  9.81it/s]
Epoch 1:  84%|████████▍ | 258/308 [03:47<00:44,  1.14it/s, loss=0.0113, v_num=54, val_acc=0.987]
Validating:  19%|█▉        | 12/62 [00:01<00:05,  9.77it/s]
Validating:  21%|██        | 13/62 [00:01<00:04,  9.82it/s]
Epoch 1:  85%|████████▍ | 261/308 [03:48<00:40,  1.15it/s, loss=0.0113, v_num=54, val_acc=0.987]
Validating:  24%|██▍       | 15/62 [00:01<00:04,  9.76it/s]
Validating:  26%|██▌       | 16/62 [00:01<00:04,  9.71it/s]
Epoch 1:  86%|████████▌ | 264/308 [03:48<00:37,  1.16it/s, loss=0.0113, v_num=54, val_acc=0.987]
Validating:  29%|██▉       | 18/62 [00:01<00:04,  9.14it/s]
Validating:  31%|███       | 19/62 [00:01<00:04,  9.21it/s]
Epoch 1:  87%|████████▋ | 267/308 [03:48<00:35,  1.17it/s, loss=0.0113, v_num=54, val_acc=0.987]
Validating:  34%|███▍      | 21/62 [00:02<00:04,  9.46it/s]
Validating:  35%|███▌      | 22/62 [00:02<00:04,  9.48it/s]
Epoch 1:  88%|████████▊ | 270/308 [03:49<00:32,  1.18it/s, loss=0.0113, v_num=54, val_acc=0.987]
Validating:  39%|███▊      | 24/62 [00:02<00:03,  9.65it/s]
Epoch 1:  89%|████████▊ | 273/308 [03:49<00:29,  1.19it/s, loss=0.0113, v_num=54, val_acc=0.987]
Validating:  44%|████▎     | 27/62 [00:02<00:03,  9.70it/s]
Validating:  45%|████▌     | 28/62 [00:02<00:03,  9.64it/s]
Epoch 1:  90%|████████▉ | 276/308 [03:49<00:26,  1.21it/s, loss=0.0113, v_num=54, val_acc=0.987]
Validating:  48%|████▊     | 30/62 [00:03<00:03,  9.66it/s]
Validating:  50%|█████     | 31/62 [00:03<00:03,  9.75it/s]
Epoch 1:  91%|█████████ | 279/308 [03:50<00:23,  1.22it/s, loss=0.0113, v_num=54, val_acc=0.987]
Validating:  53%|█████▎    | 33/62 [00:03<00:03,  9.63it/s]
Validating:  55%|█████▍    | 34/62 [00:03<00:02,  9.71it/s]
Epoch 1:  92%|█████████▏| 282/308 [03:50<00:21,  1.23it/s, loss=0.0113, v_num=54, val_acc=0.987]
Validating:  58%|█████▊    | 36/62 [00:03<00:02,  9.76it/s]
Epoch 1:  93%|█████████▎| 285/308 [03:50<00:18,  1.24it/s, loss=0.0113, v_num=54, val_acc=0.987]
Validating:  63%|██████▎   | 39/62 [00:04<00:02,  9.82it/s]
Validating:  65%|██████▍   | 40/62 [00:04<00:02,  9.77it/s]
Epoch 1:  94%|█████████▎| 288/308 [03:51<00:15,  1.25it/s, loss=0.0113, v_num=54, val_acc=0.987]
Validating:  68%|██████▊   | 42/62 [00:04<00:02,  9.81it/s]
Validating:  69%|██████▉   | 43/62 [00:04<00:01,  9.80it/s]
Epoch 1:  94%|█████████▍| 291/308 [03:51<00:13,  1.26it/s, loss=0.0113, v_num=54, val_acc=0.987]
Validating:  73%|███████▎  | 45/62 [00:04<00:01,  9.79it/s]
Validating:  74%|███████▍  | 46/62 [00:04<00:01,  9.80it/s]
Epoch 1:  95%|█████████▌| 294/308 [03:51<00:10,  1.27it/s, loss=0.0113, v_num=54, val_acc=0.987]
Validating:  79%|███████▉  | 49/62 [00:05<00:01,  9.97it/s]
Epoch 1:  96%|█████████▋| 297/308 [03:51<00:08,  1.28it/s, loss=0.0113, v_num=54, val_acc=0.987]
Validating:  82%|████████▏ | 51/62 [00:05<00:01,  9.92it/s]
Epoch 1:  97%|█████████▋| 300/308 [03:52<00:06,  1.30it/s, loss=0.0113, v_num=54, val_acc=0.987]
Validating:  87%|████████▋ | 54/62 [00:05<00:00, 10.06it/s]
Epoch 1:  98%|█████████▊| 303/308 [03:52<00:03,  1.31it/s, loss=0.0113, v_num=54, val_acc=0.987]
Validating:  92%|█████████▏| 57/62 [00:05<00:00,  9.95it/s]
Validating:  94%|█████████▎| 58/62 [00:05<00:00,  9.90it/s]
Epoch 1:  99%|█████████▉| 306/308 [03:52<00:01,  1.32it/s, loss=0.0113, v_num=54, val_acc=0.987]
Validating:  97%|█████████▋| 60/62 [00:06<00:00,  9.77it/s]
Epoch 1: 100%|██████████| 308/308 [03:53<00:00,  1.33it/s, loss=0.0113, v_num=54, val_acc=0.995]
Epoch 1: 100%|██████████| 308/308 [03:53<00:00,  1.32it/s, loss=0.0113, v_num=54, val_acc=0.995]

Process finished with exit code 0

'''