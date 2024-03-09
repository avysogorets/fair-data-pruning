from .vision_data import VisionDataBase
import torchvision


class CIFAR10(VisionDataBase):
    def __init__(self, path, **kwargs):
        transform = [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.4914,0.4822,0.4465),
                (0.2470,0.2435,0.2616))]
        augmentation = [
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
            torchvision.transforms.RandomAffine(
                degrees=0,
                translate=(0.15,0.15))]
        aug_transform = torchvision.transforms.Compose(augmentation+transform)
        pln_transform = torchvision.transforms.Compose(transform)
        train_aug_dataset = torchvision.datasets.CIFAR10(
            path,
            train=True,
            download=True,
            transform=aug_transform)
        train_pln_dataset = torchvision.datasets.CIFAR10(
            path,
            train=True,
            download=True,
            transform=pln_transform)
        test_dataset = torchvision.datasets.CIFAR10(
            path,
            train=False,
            download=True,
            transform=pln_transform)
        self.full_datasets = {
            'train': {
                True: train_aug_dataset,
                False: train_pln_dataset},
            'test': test_dataset}
        self.num_classes = 10
        self.in_shape = (3,32,32)
        super().__init__(**kwargs)