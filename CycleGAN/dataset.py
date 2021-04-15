import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset

class Color_enhancement(Dataset):
    def __init__(self, orange_root, apple_root, transform=None):
        self.orange_root = orange_root
        self.apple_root = apple_root
        self.transform = transform

        self.orange_images = os.listdir(orange_root)
        self.apple_images = os.listdir(apple_root)

        self.length_dataset = max(len(self.orange_images), len(self.apple_images))
        self.orange_len = len(self.orange_images)
        self.apple_len = len(self.apple_images)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        orange_img = self.orange_images[index % self.orange_len]
        apple_img = self.apple_images[index % self.apple_len]

        orange_path = os.path.join(self.orange_root, orange_img)
        apple_path = os.path.join(self.apple_root, apple_img)

        orange_img = np.array(Image.open(orange_path).convert('RGB'))
        apple_img = np.array(Image.open(apple_path).convert('RGB'))

        if self.transform:
            augmentations = self.transform(image=orange_img, image0=apple_img)
            orange_img = augmentations['image']
            apple_img = augmentations['image0']

        return orange_img, apple_img

