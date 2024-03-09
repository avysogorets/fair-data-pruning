from .diversifier_base import DiversifierBase
import numpy as np


class Auto(DiversifierBase):
    """ Auto (e.g., no diversifier)
    """
    def __init__(self, **kwargs):
        super().__init__()

    def fit(self, scores, **kwargs):
        self.idxs = sorted(range(len(scores)), key=lambda i: scores[i])
        self.idxs = np.array(self.idxs)

    def __call__(self, select_size):
        assert hasattr(self, 'idxs'), "fit the Auto diversifier first."
        idxs = self.idxs[-select_size:].tolist()
        return idxs

        