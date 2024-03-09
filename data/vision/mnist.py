from .vision_data import VisionDataBase
import torchvision


class MNIST(VisionDataBase):
    def __init__(self, path, **kwargs):
        transform = [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.1307,),
                (0.3081,))]
        augmentation = [torchvision.transforms.RandomRotation(degrees=4)]
        aug_transform = torchvision.transforms.Compose(augmentation+transform)
        pln_transform = torchvision.transforms.Compose(transform)
        train_aug_dataset = torchvision.datasets.MNIST(
            path,
            train=True,
            download=True,
            transform=aug_transform)
        train_pln_dataset = torchvision.datasets.MNIST(
            path,
            train=True,
            download=True,
            transform=pln_transform)
        test_dataset = torchvision.datasets.MNIST(
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
        self.in_shape = (1,28,28)
        super().__init__(**kwargs)