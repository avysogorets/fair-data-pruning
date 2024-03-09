from .classification_model_base import ClassificationModelBase
import torch


class VGG16(ClassificationModelBase):
    
    def __init__(self, device, in_shape, num_classes, dropout, **kwargs):
        super().__init__(device)
        self.in_shape = in_shape
        self.num_classes = num_classes
        num_at_flat = int(512*(in_shape[1]/32)**2)
        self.extractor = torch.nn.Sequential(
            torch.nn.Conv2d(in_shape[0], 64, kernel_size=(3,3), padding=1, stride=1, bias=False),
            torch.nn.BatchNorm2d(64, momentum=0.01),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout),
            torch.nn.Conv2d(64, 64, kernel_size=(3,3), padding=1, stride=1, bias=False),
            torch.nn.BatchNorm2d(64, momentum=0.01),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout),
            torch.nn.MaxPool2d(kernel_size=(2,2), stride=(2,2), padding=0),
            torch.nn.Conv2d(64, 128, kernel_size=(3,3), padding=1, stride=1, bias=False),
            torch.nn.BatchNorm2d(128, momentum=0.01),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout),
            torch.nn.Conv2d(128, 128, kernel_size=(3,3), padding=1, stride=1, bias=False),
            torch.nn.BatchNorm2d(128, momentum=0.01),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout),
            torch.nn.MaxPool2d(kernel_size=(2,2), stride=(2,2), padding=0),
            torch.nn.Conv2d(128, 256, kernel_size=(3,3), padding=1, stride=1, bias=False),
            torch.nn.BatchNorm2d(256, momentum=0.01),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout),
            torch.nn.Conv2d(256, 256, kernel_size=(3,3), padding=1, stride=1, bias=False),
            torch.nn.BatchNorm2d(256, momentum=0.01),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout),
            torch.nn.Conv2d(256, 256, kernel_size=(3,3), padding=1, stride=1, bias=False),
            torch.nn.BatchNorm2d(256, momentum=0.01),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout),
            torch.nn.MaxPool2d(kernel_size=(2,2), stride=(2,2), padding=0),
            torch.nn.Conv2d(256, 512, kernel_size=(3,3), padding=1, stride=1, bias=False),
            torch.nn.BatchNorm2d(512, momentum=0.01),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout),
            torch.nn.Conv2d(512, 512, kernel_size=(3,3), padding=1, stride=1, bias=False),
            torch.nn.BatchNorm2d(512, momentum=0.01),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout),
            torch.nn.Conv2d(512, 512, kernel_size=(3,3), padding=1, stride=1, bias=False),
            torch.nn.BatchNorm2d(512, momentum=0.01),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout),
            torch.nn.MaxPool2d(kernel_size=(2,2), stride=(2,2), padding=0),
            torch.nn.Conv2d(512, 512, kernel_size=(3,3), padding=1, stride=1, bias=False),
            torch.nn.BatchNorm2d(512, momentum=0.01),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout),
            torch.nn.Conv2d(512, 512, kernel_size=(3,3), padding=1, stride=1, bias=False),
            torch.nn.BatchNorm2d(512, momentum=0.01),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout),
            torch.nn.Conv2d(512, 512, kernel_size=(3,3), padding=1, stride=1, bias=False),
            torch.nn.BatchNorm2d(512, momentum=0.01),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout),
            torch.nn.MaxPool2d(kernel_size=(2,2), stride=(2,2), padding=0),
            torch.nn.Flatten())
        self.classifier = torch.nn.Linear(num_at_flat, num_classes, bias=True)
        self.initialize()