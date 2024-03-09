from ..globals import ACTIVE_LEARNING
from ..utils import get_current_gradients
from .scorer_base import ScorerBase
from tqdm.auto import tqdm
from torch.utils.data import Subset
import torch.nn.functional as F
import torch
import numpy as np


class GradientBased(ScorerBase):
    """ Based on expected gradient length.
        For active learning: https://arxiv.org/abs/1612.03226
        For data pruning: https://arxiv.org/abs/2107.07075
    """

    def __init__(self, ly_name, **kwargs):
        super().__init__(**kwargs)
        ly_options = {
            'max': self._max_ly,
            'avg': self._avg_ly,
            'oracle': self._oracle_ly}
        self.ly_aggrerator = ly_options[ly_name]
        self.in_train = False
        err_msg = 'Cannot use label information in active learning'
        assert not (ly_name == 'oracle' and self.strategy == ACTIVE_LEARNING), err_msg

    def _max_ly(self, model, dataset, **kwargs):
        model.zero_grad()
        output = model.predict(dataset)
        y_max = torch.LongTensor([output.argmax().item()])
        y_max = y_max.to(model.device)
        loss = F.cross_entropy(output, y_max)
        loss.backward()
        g = get_current_gradients(model).cpu()
        model.zero_grad()
        return g.norm()

    def _avg_ly(self, model, dataset, K, **kwargs):
        """ Caution: expensive!
        """
        score = 0
        for k in range(K):
            model.zero_grad()
            output = model.predict(dataset)
            y_k = torch.LongTensor([k]).to(model.device)
            loss = F.cross_entropy(output, y_k)
            loss.backward()
            p = F.softmax(output.detach(), dim=-1)[k]
            g = get_current_gradients(model).cpu()
            score += p*g.norm()
        model.zero_grad()
        return score

    def _oracle_ly(self, model, dataset, **kwargs):
        model.zero_grad()
        y_true = torch.LongTensor([dataset[0][1]])
        y_true = y_true.to(model.device)
        output = model.predict(dataset)
        loss = F.cross_entropy(output, y_true)
        loss.backward()
        g = get_current_gradients(model).cpu()
        model.zero_grad()
        return g.norm()

    def score(self, model, data, **kwargs):
        model.set_eval_mode()
        pool_dataset = data.get_pool_dataset(self.aug_key)
        scores = np.zeros(len(data.pool_idxs))
        for idx in tqdm(range(len(data.pool_idxs))):
            dataset = Subset(pool_dataset, [idx])
            scores[idx] = self.ly_aggrerator(
                    model=model,
                    dataset=dataset,
                    K=data.num_classes)
        return scores