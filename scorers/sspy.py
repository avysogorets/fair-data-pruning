from ..globals import DATA_PRUNING
from .scorer_base import ScorerBase
from torch.utils.data import DataLoader
import torch

class SSPY(ScorerBase):
    """ Label-dependent variant of Self-Supervised Pruning
        (see: https://arxiv.org/abs/2206.14486)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.in_train = False
        err_msg = "SSPY is intended for data pruning only"
        assert self.strategy == DATA_PRUNING, err_msg

    def score(self, model, data, **kwargs):
        model.set_eval_mode(enable_dropout=False)
        dataset = data.full_datasets["train"][self.aug_key]
        extractor = model.extractor
        representations = [[] for _ in range(data.num_classes)]
        class_idxs = [[] for _ in range(data.num_classes)]
        batch_size = 128
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        for i,(X,y) in enumerate(dataloader):
            representation = extractor(X.to(model.device)).detach()
            for j,(rep,k) in enumerate(zip(representation, y)):
                representations[k].append(rep)
                class_idxs[k].append(i*batch_size+j)
        for k in range(data.num_classes):
            representations[k] = torch.vstack(representations[k])
        scores = torch.zeros(len(dataset)).to(model.device)
        for k in range(data.num_classes):
            mu_k = torch.mean(representations[k], dim=0) 
            scores[class_idxs[k]] = torch.linalg.norm(representations[k]-mu_k, dim=-1)
        return scores.cpu().numpy()