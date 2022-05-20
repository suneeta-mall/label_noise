# Borrowed from https://github.com/pytorch/vision/blob/main/references/segmentation/transforms.py
import random

import torch
from torchvision.transforms import functional as F

default_mean = [0.485, 0.456, 0.406]
default_std = [0.229, 0.224, 0.225]


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image):
        for t in self.transforms:
            image = t(image)
        return image


class RandomHorizontalFlip:
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
        return image


class PILToTensor:
    def __call__(self, image):
        image = F.pil_to_tensor(image)
        return image


class ConvertDtype:
    def __init__(self, dtype):
        self.dtype = dtype

    def __call__(self, image):
        image = F.convert_image_dtype(image, self.dtype)
        return image


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image


class UnNormalise:
    def __init__(self, from_mean=default_mean, from_std=default_std):
        self.mean = from_mean
        self.std = from_std

    def __call__(self, image):
        image = F.normalize(image, mean=(0.0, 0.0, 0.0), std=(1 / self.std[0], 1 / self.std[1], 1 / self.std[2]))
        image = F.normalize(image, mean=(-self.mean[0], -self.mean[1], --self.mean[2]), std=(1.0, 1.0, 1.0))
        return image


class TrainTransformer:
    def __init__(
        self,
        hflip_prob=0.5,
        mean=default_mean,
        std=default_std,
    ):
        trans = [
            RandomHorizontalFlip(hflip_prob),
            PILToTensor(),
            ConvertDtype(torch.float),
            Normalize(mean=mean, std=std),
        ]
        self.transforms = Compose(trans)

    def __call__(self, img):
        return self.transforms(img)


class EvalTransformer:
    def __init__(
        self,
        mean=default_mean,
        std=default_std,
    ):
        self.transforms = Compose(
            [
                PILToTensor(),
                ConvertDtype(torch.float),
                Normalize(mean=mean, std=std),
            ]
        )

    def __call__(self, img):
        return self.transforms(img)
