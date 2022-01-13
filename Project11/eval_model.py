from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from torchvision import transforms

from Project11.load_test import load_test_data
from Project11.model import MyModel

data_transforms = transforms.Compose([
    transforms.Resize([112, 112]),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    transforms.ToTensor()
])
model = MyModel.load_from_checkpoint("model_checkpoint.ckpt")
test_data = load_test_data(transform=data_transforms)
test_loader = DataLoader(test_data)

trainer = Trainer(auto_scale_batch_size='power', gpus=1, deterministic=True,
                     max_epochs=15)

trainer.test(model, test_loader)

'''
C:\DeepLearningProjects\venv\Scripts\python.exe C:/DeepLearningProjects/Project11/eval_model.py
GPU available: True, used: True
TPU available: False, using: 0 TPU cores
IPU available: False, using: 0 IPUs
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
Testing: 0it [00:00, ?it/s]C:\DeepLearningProjects\venv\lib\site-packages\pytorch_lightning\trainer\data_loading.py:105: UserWarning: The dataloader, test dataloader 0, does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` (try 12 which is the number of cpus on this machine) in the `DataLoader` init to improve performance.
  rank_zero_warn(
Testing: 100%|█████████▉| 12629/12630 [03:11<00:00, 63.08it/s]--------------------------------------------------------------------------------
DATALOADER:0 TEST RESULTS
{'acc': 0.9595407843589783}
--------------------------------------------------------------------------------
Testing: 100%|██████████| 12630/12630 [03:11<00:00, 65.91it/s]

Process finished with exit code 0
'''
