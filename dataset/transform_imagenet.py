import math
import numpy as np
import random
from PIL import Image

import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F


class Resize(object):
    def __init__(self, size, interpolation=F.InterpolationMode.BICUBIC):
        self.size = size
        self.interpolation = interpolation
    def __call__(self, image, target=None):
        image = F.resize(image, size=self.size, interpolation=F.InterpolationMode.LANCZOS)
        if target is not None:
            target = F.resize(target, size=self.size, interpolation=F.InterpolationMode.LANCZOS)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, flip_prob=0.5):
        self.flip_prob = flip_prob

    def __call__(self, image, target=None):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            if target is not None:
                target = F.hflip(target)
        return image, target


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target=None,):
        if isinstance(self.size, list):
            size = random.choice(self.size)
        else:
            size = self.size
        crop_params = T.RandomCrop.get_params(image, size)
        image = F.crop(image, *crop_params)
        if target is not None:
            target = F.crop(target, *crop_params)
        return image, target


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target=None):
        image = F.center_crop(image, self.size)
        if target is not None:
            target = F.center_crop(target, self.size)
        return image, target


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None):
        image = F.normalize(image, mean=self.mean, std=self.std)
        target = F.normalize(target, mean=self.mean, std=self.std)
        return image, target


class Pad(object):
    def __init__(self, padding_n, padding_fill_value=0, padding_fill_target_value=0):
        self.padding_n = padding_n
        self.padding_fill_value = padding_fill_value
        self.padding_fill_target_value = padding_fill_target_value

    def __call__(self, image, target=None):
        image = F.pad(image, self.padding_n, self.padding_fill_value)
        if target is not None:
            target = F.pad(target, self.padding_n, self.padding_fill_target_value)
        return image, target


class ToTensor(object):
    def __call__(self, image, target=None):
        image = F.to_tensor(image)
        if target is not None:
            target = F.to_tensor(target)
        return image, target


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, mask=None):
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask


def create_image_mask_transforms(image_size, random_crop=False, mid_res=1.125):
    mid_res = round(mid_res * image_size)
    if random_crop:
        transform = Compose([
            Resize(mid_res, interpolation=F.InterpolationMode.LANCZOS),  # TODO handle mask
            RandomCrop((image_size, image_size)),
            # RandomHorizontalFlip(),
            ToTensor(),
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    else:
        transform = Compose([
            Resize(mid_res, interpolation=F.InterpolationMode.LANCZOS),  # TODO handle mask
            CenterCrop((image_size, image_size)),
            ToTensor(),
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    return transform
