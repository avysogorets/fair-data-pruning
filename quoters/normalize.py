from .quoter_base import QuoterBase
import torch

class Normalize(QuoterBase):
    """ Class-wise score normalization.
    """
    def __init__(self, **kwargs):
        super().__init__()

    def __call__(self, iter_scores, data, select_size, **kwargs):
        iter_scores = torch.FloatTensor(iter_scores)
        local_y = []
        local_idxs_by_class = {k: [] for k in range(data.num_classes)}
        for i,(_,y) in enumerate(data.get_pool_dataset(False)):
            local_idxs_by_class[y].append(i)
            local_y.append(y)
        for k in range(data.num_classes):
            if len(local_idxs_by_class[k]) == 0:
                continue
            min_k = min(iter_scores[local_idxs_by_class[k]])
            max_k = max(iter_scores[local_idxs_by_class[k]])
            orig = iter_scores[local_idxs_by_class[k]]
            iter_scores[local_idxs_by_class[k]] = (orig-min_k)/(max_k-min_k)
        all_idxs = range(len(iter_scores))
        sorted_idxs = sorted(all_idxs, key=lambda i: iter_scores[i])[-select_size:]
        top_idxs = sorted_idxs[-select_size:]
        quota = [0 for _ in range(data.num_classes)]
        for idx in top_idxs:
            quota[local_y[idx]] += 1./select_size
        return quota