from .vision_data import VisionDataBase
from torch.utils.data import Subset
import torchvision
import os
import numpy as np


class TinyImageNet(VisionDataBase):
    def __init__(self, path, **kwargs):
        transform = [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=(0.480,0.448,0.397),
                std=(0.276,0.269,0.282))]
        augmentation = [
            torchvision.transforms.RandomCrop(64,padding=4),
            torchvision.transforms.RandomHorizontalFlip()]
        aug_transform = torchvision.transforms.Compose(augmentation+transform)
        pln_transform = torchvision.transforms.Compose(transform)
        train_aug_dataset = torchvision.datasets.ImageFolder(
            root=os.path.join(path, 'tiny-imagenet-200/train'),
            transform=aug_transform)
        train_pln_dataset = torchvision.datasets.ImageFolder(
            root=os.path.join(path, 'tiny-imagenet-200/train'),
            transform=pln_transform)
        test_dataset = torchvision.datasets.ImageFolder(
            root=os.path.join(path, 'tiny-imagenet-200/val'),
            transform=pln_transform)
        self.full_datasets = {
            'train': {
                True: train_aug_dataset,
                False: train_pln_dataset},
            'test': test_dataset}
        self.num_classes = 200
        self.in_shape = (3,64,64)
        super().__init__(**kwargs)