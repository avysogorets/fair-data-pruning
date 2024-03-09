# Adapted from https://github.com/dsgissin/DiscriminativeActiveLearning

from ..globals import ACTIVE_LEARNING, DATA_PRUNING
from .scorer_base import ScorerBase
from tqdm.auto import tqdm
import numpy as np
import torch
from scipy.spatial import distance_matrix


class CoreSet(ScorerBase):
    """ Greedy CoreSet (Sener & Savarese, 2017)
        (https://arxiv.org/abs/1708.00489)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.in_train = False

    def _greedy_k_center(self, base, pool, batch_size):
        greedy_indices = []
        min_dist = np.min(distance_matrix(base[0,:].reshape((1, base.shape[1])), pool), axis=0)
        min_dist = min_dist.reshape((1, min_dist.shape[0]))
        for j in tqdm(range(1, base.shape[0], 100)):
            if j + 100 < base.shape[0]:
                dist = distance_matrix(base[j:j+100,:], pool)
            else:
                dist = distance_matrix(base[j:,:], pool)
            min_dist = np.vstack((min_dist, np.min(dist, axis=0).reshape((1, min_dist.shape[1]))))
            min_dist = np.min(min_dist, axis=0)
            min_dist = min_dist.reshape((1, min_dist.shape[0]))
        farthest = np.argmax(min_dist)
        greedy_indices.append(farthest)
        for _ in tqdm(range(batch_size-1)):
            dist = distance_matrix(pool[greedy_indices[-1],:].reshape((1, pool.shape[1])), pool)
            min_dist = np.vstack((min_dist, dist.reshape((1, min_dist.shape[1]))))
            min_dist = np.min(min_dist, axis=0)
            min_dist = min_dist.reshape((1, min_dist.shape[0]))
            farthest = np.argmax(min_dist)
            greedy_indices.append(farthest)
        return np.array(greedy_indices, dtype=int).tolist()
    
    def _query_regular(self, model, data, batch_size):
        if batch_size == 0:
            return []
        new_indices = []
        with torch.no_grad():
            full_dataset = data.full_datasets['train'][self.aug_key]
            embeddings = model.embeddings(full_dataset).cpu().numpy()
        pool_rep = embeddings[data.pool_idxs]
        if self.strategy == ACTIVE_LEARNING:
            base_rep = embeddings[data.selected_idxs]
        elif self.strategy == DATA_PRUNING:
            base_rep = np.mean(embeddings[data.selected_idxs], axis=0)
            base_idx = 0
            base_dist = float('inf')
            for idx in range(len(data.pool_idxs)):
                dist = np.linalg.norm(base_rep-embeddings[idx])
                if dist < base_dist:
                    base_dist = dist
                    base_idx = idx
            base_rep = embeddings[data.pool_idxs][base_idx][None,:]
        else:
            RuntimeError(f"unknown strategy <{self.strategy}>")
        new_indices += self._greedy_k_center(
                base=base_rep,
                pool=pool_rep,
                batch_size=batch_size)
        if self.strategy == DATA_PRUNING:
            if base_idx not in new_indices:
                if batch_size > 0:
                    new_indices[-1] = base_idx
        return new_indices

    def score(self, model, data, batch_size):
        local_idxs = self._query_regular(model, data, batch_size)
        scores = np.zeros(len(data.pool_idxs))
        for idx in local_idxs:
            scores[idx] = 1.
        return scores
        
            
