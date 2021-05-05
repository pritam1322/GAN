import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
device = 'cuda' if torch.cuda.is_available() else "cpu"
train_dir = "archive (1)/data/train"
val_dir = "archive (1)/data/val"
learning_rate = 2e-4
batch_size = 16
num_worker = 2
image_size = 256
channels_img = 3
l1_lambda = 100
lambda_gp = 10
num_epochs = 50
Load_model = False
Save_model = False
CHECKPOINT_DISC = "disc.pth.tar"
CHECKPOINT_GEN = "gen.pth.tar"


both_transform = A.Compose(
    [A.Resize(width=256, height=256),], additional_targets={"image0": "image"},
)

transform_only_input = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.ColorJitter(p=0.2),
    A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
    ToTensorV2(),

])

transform_only_mask = A.Compose(
    [
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
        ToTensorV2(),
    ]
)