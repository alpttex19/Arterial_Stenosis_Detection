import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
import torchvision

class CustomDataset(Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        root_list = root.split("/")
        self.transforms = transforms
        self.images = list(sorted(os.listdir(os.path.join(root, "images"))))
        ann_path = (os.path.join(root, "annotations", f"{root_list[-1]}.json"))
        with open(ann_path) as f:
            self.ann = json.load(f)

    def __getitem__(self, idx):
        img_name = f"{self.ann['annotations'][idx]['image_id']}.png"
        img_path = os.path.join(self.root, "images", img_name)
        
        img = Image.open(img_path).convert("RGB")

        boxes = []
        labels = []
        obj = self.ann['annotations'][idx]
        xmin = obj['bbox'][0]
        ymin = obj['bbox'][1]
        xmax = xmin + obj['bbox'][2]
        ymax = ymin + obj['bbox'][3]
        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(1)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels
        }

        if self.transforms is not None:
            img = self.transforms(img)
        
        return img, target, img_name

    def __len__(self):
        return len(self.images)

def get_transform():
    return torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])
