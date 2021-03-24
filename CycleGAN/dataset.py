import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset

class Color_enhancement(Dataset):
    def __init__(self, iphone_root, dsrl_root, transform=None):
        self.iphone_root = iphone_root
        self.dsrl_root = dsrl_root
        self.transform = transform

        self.iphone_images = os.listdir(iphone_root)
        self.dsrl_images = os.listdir(dsrl_root)

        self.length_dataset = max(len(self.iphone_images), len(self.dsrl_images))
        self.iphone_len = len(self.iphone_images)
        self.dsrl_len = len(self.dsrl_images)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        iphone_img = self.iphone_images[index % self.iphone_len]
        dsrl_img = self.dsrl_images[index % self.dsrl_len]

        iphone_path = os.path.join(self.iphone_root, iphone_img)
        dsrl_path = os.path.join(self.dsrl_root, dsrl_img)

        iphone_img = np.array(Image.open(iphone_path).convert('RGB'))
        dsrl_img = np.array(Image.open(dsrl_path).convert('RGB'))

        if self.transform:
            albulmentations = self.transform(image=iphone_img, image0=dsrl_img)
            iphone_img = albulmentations['image']
            dsrl_img = albulmentations['image0']

        return iphone_img, dsrl_img

