import os

from PIL import Image
import cv2
import torch
from torch.utils.data import Dataset
from data_augmentation import SSDAugmentation, SSDBaseTransform


class Stenosis_Dataset(Dataset):
    def __init__(self, mode="train"):
        self.mode = mode
        if mode == "train_1" or mode == "train_2":
            file_mode = "train"
        else:
            file_mode = mode            
        self.data_root = f"stenosis_data/{file_mode}"
        self.file_names = os.listdir(f"stenosis_data/{file_mode}/images")
        self.imgs = [(torch.from_numpy(cv2.cvtColor(cv2.imread(f"{self.data_root}/images/{file_name}"), cv2.COLOR_BGR2RGB)).permute(2,0,1))[0, :, :].unsqueeze(0) / 255 for file_name in self.file_names]
        self.masks = [(torch.from_numpy(cv2.imread(f"{self.data_root}/masks/{file_name}", cv2.COLOR_BGR2GRAY))).squeeze(0) / 255 for file_name in self.file_names]
        if self.mode == "train_1":
            self.transform = SSDAugmentation(img_size=512)
        else:
            self.transform = SSDBaseTransform()


    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        image = cv2.imread(f"{self.data_root}/images/{self.file_names[idx]}")
        mask = cv2.imread(f"{self.data_root}/masks/{self.file_names[idx]}", cv2.COLOR_BGR2GRAY)
        image, mask = self.transform(image, mask)
        return image.float(), mask.long()
