from ..scorers.coreset import CoreSet as CS
from .diversifier_base import DiversifierBase
from scipy.spatial import distance_matrix
import numpy as np


class CoreSet(DiversifierBase):
    """ Selects data according to greedy CoreSet. Unlike
        CoreSet the scorer, CoreSet the diversifier can 
        accomodate class-wise selection quotas.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.coreset = CS(strategy=1, aug_key=False)
    
    def fit(self, data_X, **kwargs):
        self.data_X = data_X.numpy()
    
    def __call__(self, select_size):
        if select_size == 0:
            return []
        new_indices = []
        base_rep = np.mean(self.data_X, axis=0)
        base_idx = 0
        base_dist = float('inf')
        for idx in range(len(self.data_X)):
            dist = np.linalg.norm(base_rep-self.data_X[idx])
            if dist < base_dist:
                base_dist = dist
                base_idx = idx
        base_rep = self.data_X[base_idx][None,:]
        new_indices += self.coreset._greedy_k_center(
                base=base_rep,
                pool=self.data_X,
                batch_size=select_size)
        if base_idx not in new_indices and select_size > 0:
            new_indices[-1] = base_idx
        return new_indices

        