from .vision_data import VisionDataBase
import torchvision

class ImageNet(VisionDataBase):
    def __init__(self, path='/imagenet', **kwargs):
        transform = [
            torchvision.transforms.Resize(256),
            torchvision.transforms.RandomResizedCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])]
        augmentation = [
            torchvision.transforms.RandomHorizontalFlip()]
        aug_transform = torchvision.transforms.Compose(augmentation+transform)
        pln_transform = torchvision.transforms.Compose(transform)
        train_aug_dataset = torchvision.datasets.ImageFolder(
            root=f'{path}/train',
            transform=aug_transform)
        train_pln_dataset = torchvision.datasets.ImageFolder(
            root=f'{path}/train',
            transform=pln_transform)
        test_dataset = torchvision.datasets.ImageFolder(
            root=f'{path}/val',
            transform=pln_transform)
        self.full_datasets = {
            'train': {
                    True: train_aug_dataset,
                    False: train_pln_dataset},
            'test': test_dataset}
        self.num_classes = 1000
        self.in_shape = (3,224,224)
        super().__init__(**kwargs)