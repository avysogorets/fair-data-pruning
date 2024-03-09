from .classification_model_base import ClassificationModelBase
import torch


class LeNet300100(ClassificationModelBase):

    def __init__(self, device, in_shape, num_classes, dropout, **kwargs):
        super().__init__(device)
        self.in_shape = in_shape
        self.num_classes = num_classes
        in_neurons = 1
        for i in in_shape:
                in_neurons*=i
        self.extractor = torch.nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.Linear(in_neurons, 300, bias=False),
                torch.nn.BatchNorm1d(300, momentum=0.01),
                torch.nn.ReLU(),
                torch.nn.Dropout(p=dropout),
                torch.nn.Linear(300, 100, bias=False),
                torch.nn.BatchNorm1d(100, momentum=0.01),
                torch.nn.ReLU(),
                torch.nn.Dropout(p=dropout))
        self.classifier = torch.nn.Linear(100, num_classes, bias=True)
        self.initialize()