from ..globals import DATA_PRUNING
from .scorer_base import ScorerBase
import torch.nn.functional as F
import torch


class DynamicUncertainty(ScorerBase):
    def __init__(self, data_length, J, **kwargs):
        super().__init__(**kwargs)
        self.in_train = True
        self.probas = torch.zeros(0, data_length)
        self.stds = torch.zeros(data_length)
        self.data_length = data_length
        self.J = J
        self.epoch = 0
        err_msg = "Dynamic Uncertainty is intended for data pruning only"
        assert self.strategy == DATA_PRUNING, err_msg

    def score(self, idxs, logits, y, iter, **kwargs):
        probas = F.softmax(logits, dim=-1)
        y = y.reshape(-1,1)
        probas = torch.gather(probas, -1, y).squeeze()
        if iter == 0:
            self.epoch += 1
            if self.epoch >= self.J:
                self.stds += torch.std(self.probas, dim=0)
                self.probas = self.probas[1:]
            new_probas = torch.zeros(1, self.data_length)
            self.probas = torch.vstack([self.probas, new_probas])
        self.probas[-1][idxs] = probas.to(self.probas.device)
        return (self.stds/(self.epoch-self.J)).numpy()

