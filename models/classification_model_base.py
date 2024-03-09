from .utils import configure_dropout_layers
from typing import Union, Dict, Tuple
from torch.utils.data import DataLoader, Dataset, Subset
import torch.nn.functional as F
import torch 
import math


class ClassificationModelBase(torch.nn.Module):
    """ Base class for vision classification models.
    """

    def __init__(self, device: torch.device) -> None:
        super().__init__()
        self.device = device             # cpu or cuda;
        self.extractor: torch.nn.Module  # extarctor only;
        self.classifier: torch.nn.Module # linear output head;
        self.in_shape: Tuple             # input shape.

    def initialize(self) -> None:
        self.dropout_layers = configure_dropout_layers(self)
        for m in self.modules():
            if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight, gain=math.sqrt(2))
                if hasattr(m, 'bias') and m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
        self.to(self.device)

    def embedding_of_x(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        assert isinstance(x, torch.Tensor), f"input type {type(x)}, should be Tensor"
        return self.extractor(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        return self.classifier(self.embedding_of_x(x))

    def _forward(self,
            dataset: Union[Dataset, Subset],
            features_flag: bool = False,
            batch_size: int = 128) -> torch.Tensor:

        # Calling even on a small dataset outside of torch.no_grad()
        # environment will likely result in cuda OOM (graph resources
        # are freed with backward() call, which happens after every
        # batch during training but not in this _forward function). If 
        # gradients needed, write a custom validation loop that computes
        # gradients and releases graph resources after every batch. 
        # Applies to features, predict, and predict_proba.

        batch_size = min(len(dataset), batch_size)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        outputs = []
        for batch in dataloader:
            X,_ = batch
            if features_flag:
                output = self.embedding_of_x(X)
            else:
                output = self.classifier(self.embedding_of_x(X))
            outputs.append(output)
        outputs = torch.vstack(outputs)
        return outputs
        
    def embeddings(self, 
            dataset: Union[Dataset, Subset],
            batch_size: int = 128) -> torch.Tensor:
        return self._forward(dataset, features_flag=True, batch_size=batch_size)

    def predict(self,
            dataset: Union[Dataset, Subset],
            batch_size: int = 128) -> torch.Tensor:
        return self._forward(dataset, features_flag=False, batch_size=batch_size)

    def predict_proba(self, dataset: Union[Dataset, Subset], batch_size: int = 128):
        outputs = self.predict(dataset, batch_size=batch_size)
        return F.softmax(outputs, dim=-1)

    def set_train_mode(self) -> None:
        self.train()
        for dropout_layer in self.dropout_layers:
            dropout_layer.train()

    def set_eval_mode(self, enable_dropout: bool = False) -> None:
        self.eval()
        for dropout_layer in self.dropout_layers:
            dropout_layer.eval()
            dropout_layer.enable_inference_dropout = enable_dropout