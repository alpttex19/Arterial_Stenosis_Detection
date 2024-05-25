import cv2
import numpy as np
import torch
from numpy import random

# ----------------------- Augmentation Functions -----------------------

class Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask = None):
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask


class ConvertColor(object):
    def __init__(self, current='BGR', transform='HSV'):
        self.transform = transform
        self.current = current

    def __call__(self, image, mask):
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        else:
            raise NotImplementedError
        return image, mask


class Resize(object):
    def __init__(self, img_size=512):
        self.img_size = img_size

    def __call__(self, image, mask):
        image = cv2.resize(image, (self.img_size, self.img_size))
        # rescale bbox
        mask = cv2.resize(mask, (self.img_size, self.img_size))

        return image, mask


class RandomSaturation(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, mask=None):
        if random.randint(2):
            image[:, :, 1] *= random.uniform(self.lower, self.upper)

        return image, mask


class RandomHue(object):
    def __init__(self, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, image, mask):
        if random.randint(2):
            image[:, :, 0] += random.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image, mask



class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, image, mask=None):
        if random.randint(2):
            alpha = random.uniform(self.lower, self.upper)
            image *= alpha
        return image, mask


class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image, mask=None):
        if random.randint(2):
            delta = random.uniform(-self.delta, self.delta)
            image += delta
        return image, mask


class RandomSampleCrop(object):
    """Crop
    Arguments:
        img (Image): the image being input during training
        mode (float tuple): the min and max jaccard overlaps
    Return:
        (img, boxes, classes)
            img (Image): the cropped image
    """
    def __call__(self, image, mask=None):
        height, width, _ = image.shape
        # max trails (50)
        for _ in range(5):
            current_image = image
            current_mask = mask

            w = random.uniform(0.3 * width, width)
            h = random.uniform(0.3 * height, height)

            # aspect ratio constraint b/t .5 & 2
            if h / w < 0.5 or h / w > 2:
                continue

            left = random.uniform(width - w)
            top = random.uniform(height - h)

            # convert to integer rect x1,y1,x2,y2
            rect = np.array([int(left), int(top), int(left+w), int(top+h)])

            # cut the crop from the image
            current_image = current_image[rect[1]:rect[3], rect[0]:rect[2],:]
            current_mask = current_mask[rect[1]:rect[3], rect[0]:rect[2]]

            return current_image, current_mask
        return image, mask


class Expand(object):
    def __call__(self, image, mask):
        if random.randint(2):
            return image, mask

        height, width, depth = image.shape
        ratio = random.uniform(1, 2)
        left = random.uniform(0, width*ratio - width)
        top = random.uniform(0, height*ratio - height)

        expand_image = np.zeros(
            (int(height*ratio), int(width*ratio), depth),
            dtype=image.dtype)
        expand_mask = np.zeros(
            (int(height*ratio), int(width*ratio)),
            dtype=mask.dtype)

        expand_image[int(top):int(top + height),
                     int(left):int(left + width)] = image
        expand_mask[int(top):int(top + height),
                     int(left):int(left + width)] = mask
        image = expand_image
        mask = expand_mask

        return image, mask


class RandomHorizontalFlip(object):
    def __call__(self, image, mask):
        if random.randint(2):
            image = image[:, ::-1]
            mask = mask[:, ::-1]

        return image, mask
    

class RandomVerticalFlip(object):
    def __call__(self, image, mask):
        if random.randint(2):
            image = image[::-1, :]
            mask = mask[::-1, :]

        return image, mask
    
class RandomRotate(object):
    def __init__(self, angle=90):
        self.angle = random.uniform(-angle, angle)

    def __call__(self, image, mask):
        angle = random.uniform(-self.angle, self.angle)
        h, w, _ = image.shape
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        image = cv2.warpAffine(image, M, (w, h))
        mask = cv2.warpAffine(mask, M, (w, h))

        return image, mask


class PhotometricDistort(object):
    def __init__(self):
        self.pd = [
            RandomContrast(),
            ConvertColor(transform='HSV'),
            RandomSaturation(),
            RandomHue(),
            ConvertColor(current='HSV', transform='BGR'),
            RandomContrast()
        ]
        self.rand_brightness = RandomBrightness()

    def __call__(self, image, mask):
        im = image.copy()
        im, mask = self.rand_brightness(im, mask)
        if random.randint(2):
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1:])
        im, mask = distort(im, mask)
        return im, mask


# ----------------------- Main Functions -----------------------
## SSD-style Augmentation
class SSDAugmentation(object):
    def __init__(self, img_size=512):
        self.img_size = img_size
        self.augment = Compose([
            # Expand(),                                  # 扩充增强
            RandomSampleCrop(),                        # 随机剪裁
            RandomHorizontalFlip(),                    # 随机水平翻转
            RandomVerticalFlip(),                      # 随机垂直翻转
            RandomRotate(angle=180),                            # 随机旋转
            Resize(self.img_size)                      # resize操作
        ])

    def __call__(self, image, mask):
        # augment
        image, mask = self.augment(image, mask)

        # to tensor
        img_tensor = (torch.from_numpy(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).permute(2,0,1))[0, :, :].unsqueeze(0) / 255
        mask_tensor = torch.from_numpy(mask).squeeze(0)/255
        

        return img_tensor, mask_tensor
    

## SSD-style valTransform
class SSDBaseTransform(object):
    def __init__(self, img_size=512):
        self.img_size = img_size
        self.resize = Compose([
            Resize(self.img_size)                      # resize操作
        ])
    def __call__(self, image, mask=None): 
        # augment
        image, mask = self.resize(image, mask) 
        # to tensor
        img_tensor = (torch.from_numpy(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).permute(2,0,1))[0, :, :].unsqueeze(0) / 255
        mask_tensor = torch.from_numpy(mask).squeeze(0)/255
            
        return img_tensor, mask_tensor
