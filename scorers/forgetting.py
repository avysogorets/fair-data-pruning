from ..globals import DATA_PRUNING
from .scorer_base import ScorerBase
import torch


class Forgetting(ScorerBase):
    def __init__(self, data_length, **kwargs):
        super().__init__(**kwargs)
        self.in_train = True
        self.prev_accs = torch.zeros(data_length)
        self.scores = torch.zeros(data_length)
        err_msg = "Forgetting is intended for data pruning only"
        assert self.strategy == DATA_PRUNING, err_msg

    def score(self, idxs, logits, y, coeff=1, **kwargs):
        y_preds = torch.argmax(logits, dim=-1).squeeze()
        y = y.squeeze()
        accs = (y_preds == y).long().to(self.prev_accs.device)
        acc_drop = (accs<self.prev_accs[idxs]).long()
        self.scores[idxs] += acc_drop*coeff
        self.prev_accs[idxs] = accs.float()
        return self.scores.numpy()