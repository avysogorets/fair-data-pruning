from ..globals import ACTIVE_LEARNING
from .scorer_base import ScorerBase
import torch


class VariationRatio(ScorerBase):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.in_train = False
        err_msg = "VariationRatio is intended for active learning only"
        assert self.strategy == ACTIVE_LEARNING, err_msg

    def score(self, model, data, **kwargs):
        model.set_eval_mode()
        dataset = data.get_pool_dataset(self.aug_key)
        with torch.no_grad():
            probas = model.predict_proba(dataset)
        scores = (1-torch.max(probas, dim=-1).values)
        scores = self.scores.cpu().numpy()
        return scores