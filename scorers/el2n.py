from ..globals import ACTIVE_LEARNING
from .scorer_base import ScorerBase
import torch.nn.functional as F
import torch

class EL2N(ScorerBase):
    def __init__(self, ly_name, **kwargs):
        super().__init__(**kwargs)
        ly_options = {
            'max': self._max_ly,
            'avg': self._avg_ly,
            'oracle': self._oracle_ly}
        self.ly_aggrerator = ly_options[ly_name]
        self.in_train = False
        assert not (ly_name == 'oracle' and self.strategy == ACTIVE_LEARNING)

    def _avg_ly(self, probas, num_classes, **kwargs):
        scores = torch.zeros(len(probas))
        for c in range(num_classes):
            targets = c*torch.ones(len(probas)).long()
            one_hot_targets = F.one_hot(targets, num_classes=num_classes)
            one_hot_targets = one_hot_targets.float()
            errors = probas - one_hot_targets
            errors_norm = torch.linalg.norm(errors, ord=2, dim=-1)
            scores += probas[:,c]*errors_norm
        return scores

    def _max_ly(self, probas, num_classes, **kwargs):
        targets = torch.argmax(probas, dim=-1).long()
        one_hot_targets = F.one_hot(targets, num_classes=num_classes)
        one_hot_targets = one_hot_targets.float()
        errors = probas - one_hot_targets
        scores = torch.linalg.norm(errors, ord=2, dim=-1)
        return scores

    def _oracle_ly(self, probas, targets, **kwargs):
        one_hot_targets = F.one_hot(targets).float()
        errors = probas - one_hot_targets.to(probas.device)
        scores = torch.linalg.norm(errors, ord=2, dim=-1)
        return scores

    def score(self, model, data, **kwargs):
        model.set_eval_mode(enable_dropout=False)
        dataset = data.get_pool_dataset(self.aug_key)
        with torch.no_grad():
            probas = model.predict_proba(dataset)
        labels = []
        for _,y in dataset:
            labels.append(y)
        labels = torch.LongTensor(labels)
        scores = self.ly_aggrerator(
                probas=probas,
                targets=labels,
                num_classes=data.num_classes)
        return scores.cpu().numpy()