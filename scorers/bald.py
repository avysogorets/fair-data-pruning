from .scorer_base import ScorerBase
from torch.distributions import Categorical
from tqdm.auto import tqdm
import torch


class BALD(ScorerBase):
    """ Approximate inference via MC Dropout
        Source: https://arxiv.org/pdf/1703.02910.pdf)
    """
    def __init__(self, k, **kwargs):
        super().__init__(**kwargs)
        self.k = k
        self.in_train = False

    def get_probabilities_and_entropies(self, model, data):
        dataset = data.get_pool_dataset(self.aug_key)
        probabilities = torch.zeros((len(dataset), self.k, data.num_classes))
        entropies = torch.zeros((len(dataset), self.k))
        for w in tqdm(range(self.k)):
            for dropout_layer in model.dropout_layers:
                dropout_layer.resample_inference_mask()
            with torch.no_grad():
                probas = model.predict_proba(dataset)
                probabilities[:, w, :] = probas
                entropy = Categorical(probs=probas).entropy()
                entropies[:, w] = entropy
        return probabilities.double(), entropies.double()
    
    def get_scores_from_pe(self, probabilities, entropies):
        mean_probability = torch.mean(probabilities, dim=1)
        entropy_of_mean = Categorical(probs=mean_probability).entropy().view(-1)
        mean_of_entropy = torch.mean(entropies, dim=1).view(-1)
        scores = entropy_of_mean - mean_of_entropy
        return scores

    def score(self, model, data, **kwargs):
        assert len(model.dropout_layers) > 0, "model must have dropout layers for BALD"
        model.set_eval_mode(enable_dropout=True)
        probabilities, entropies = self.get_probabilities_and_entropies(model, data)
        return self.get_scores_from_pe(probabilities, entropies).cpu().numpy()