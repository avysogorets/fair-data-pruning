from .classification_model_base import ClassificationModelBase
import torch


class LeNet5(ClassificationModelBase):

    def __init__(self, device, in_shape, num_classes, dropout, **kwargs):
        super().__init__(device)
        self.in_shape = in_shape
        self.num_classes = num_classes
        in_neurons = 1
        for i in in_shape:
                in_neurons*=i
        self.extractor = torch.nn.Sequential(
            torch.nn.Conv2d(in_shape[0], 6, padding=0, kernel_size=(5,5), stride=1, bias=False),
            torch.nn.BatchNorm2d(6, momentum=0.01),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout),
            torch.nn.AvgPool2d(kernel_size=(2,2), stride=(2,2), padding=0),
            torch.nn.Conv2d(6, 16, padding=0, kernel_size=(5,5), stride=1, bias=False),
            torch.nn.BatchNorm2d(16, momentum=0.01),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout),
            torch.nn.AvgPool2d(kernel_size=(2,2), stride=(2,2), padding=0),
            torch.nn.Flatten(),
            torch.nn.Linear(400, 120, bias=False),
            torch.nn.BatchNorm1d(120, momentum=0.01),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(120, 84, bias=False),
            torch.nn.BatchNorm1d(84, momentum=0.01),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout),
            torch.nn.Flatten())
        self.classifier = torch.nn.Linear(84, num_classes, bias=True)
        self.initialize()