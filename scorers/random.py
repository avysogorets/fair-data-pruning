from .scorer_base import ScorerBase
import numpy as np


class Random(ScorerBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.in_train = False

    def score(self, data, **kwargs):
        scores = np.random.uniform(0,1, size=len(data.pool_idxs))
        return scores