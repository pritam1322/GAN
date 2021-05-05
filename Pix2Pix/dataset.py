import torch
import config
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
from PIL import Image

class Anime_Dataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.list_dir = os.listdir(root_dir)


    def __len__(self):
        return len(self.list_dir)

    def __getitem__(self, index):
        img_file = self.list_dir[index]
        img_path = os.path.join(self.root_dir, img_file)
        image = np.array(Image.open(img_path))
        input_image = image[:,:600,:]
        target_image = image[:,600:,:]

        augmentations = config.both_transform(image=input_image, image0=target_image)
        input_image = augmentations["image"]
        target_image = augmentations["image0"]

        input_image = config.transform_only_input(image=input_image)["image"]
        target_image = config.transform_only_mask(image=target_image)["image"]

        return input_image, target_image
