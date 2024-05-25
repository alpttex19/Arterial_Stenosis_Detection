import os

from PIL import Image
import cv2
import torch
from torch.utils.data import Dataset


class Stenosis_Dataset(Dataset):
    def __init__(self, mode="train", transform=None):        
        self.data_root = f"stenosis_data/{mode}"
        self.file_names = os.listdir(f"stenosis_data/{mode}/images")
        if transform != None:
            self.transform = transform(img_size=512)



    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        image = cv2.imread(f"{self.data_root}/images/{self.file_names[idx]}")
        mask = cv2.imread(f"{self.data_root}/masks/{self.file_names[idx]}", cv2.COLOR_BGR2GRAY)
        image, mask = self.transform(image, mask)
        return image.float(), mask.long(), self.file_names[idx]
