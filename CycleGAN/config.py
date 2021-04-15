import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

device = 'cuda' if torch.cuda.is_available() else 'cpu'
train_dir = "dataset/train"
val_dir = "dataset/val"
batch_size = 1
learning_rate = 1e-5
lambda_identity = 0.0
lambda_cycle = 10
num_workers = 4
num_epochs = 50
load_model = True
save_model = True
CHECKPOINT_GEN_I = "geni.pth.tar"
CHECKPOINT_GEN_D = "gend.pth.tar"
CHECKPOINT_CRITIC_I = "critici.pth.tar"
CHECKPOINT_CRITIC_D = "criticd.pth.tar"

transform = A.Compose([
A.Resize(width=256, height=256),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(),
],
additional_targets={"image0": "image"},)