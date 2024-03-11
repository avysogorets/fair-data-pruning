from ..globals import ACTIVE_LEARNING
from .scorer_base import ScorerBase
from .bald import BALD
import torch
import numpy as np


class BatchBALD(ScorerBase):
    """ Greedy BatchBALD with sampling (unoptimized)
        (https://arxiv.org/pdf/1906.08158.pdf)
    """
    def __init__(self, k, **kwargs):
        super().__init__(**kwargs)
        self.k = k
        self.in_train = False
        err_msg = "BatchBALD is intended for active learning only"
        self.strategy == ACTIVE_LEARNING, err_msg

    def _gather_expand(self, data, dim, index):
        """ Collect probabilities of sampled classes given in index
        """
        max_shape = [max(dr, ir) for dr,ir in zip(data.shape, index.shape)]
        new_data_shape = list(max_shape)
        new_data_shape[dim] = data.shape[dim]
        new_index_shape = list(max_shape)
        new_index_shape[dim] = index.shape[dim]
        data = data.expand(new_data_shape)
        index = index.expand(new_index_shape)
        return torch.gather(data, dim, index)

    def _batch_multi_choices(self, probs_b_C, M):
        """ Sample M classes from given distributions
        """
        probs_B_C = probs_b_C.reshape((-1, probs_b_C.shape[-1]))
        choices = torch.multinomial(probs_B_C, num_samples=M, replacement=True)
        choices_b_M = choices.reshape(list(probs_b_C.shape[:-1])+[M])
        return choices_b_M

    def _sample_M_K(self, probs_N_K_C, S=1000):
        """ Compute K*S samples from joint p(y1:n|w) distribution
            from probas of N yi's given K w samples from posterior
        """
        probs_N_K_C = probs_N_K_C.double()
        K = probs_N_K_C.shape[1]
        choices_N_K_S = self._batch_multi_choices(probs_N_K_C, S).long()
        expanded_choices_N_K_K_S = choices_N_K_S[:, None, :, :]
        expanded_probs_N_K_K_C = probs_N_K_C[:, :, None, :]
        probs_N_K_K_S = self._gather_expand(expanded_probs_N_K_K_C,
                dim=-1,
                index=expanded_choices_N_K_K_S)
        probs_K_K_S = torch.exp(torch.sum(torch.log(probs_N_K_K_S), dim=0, keepdim=False))
        samples_K_M = probs_K_K_S.reshape((K, -1))
        samples_M_K = samples_K_M.t()
        return samples_M_K

    def score(self, model, data, batch_size, **kwargs):
        assert len(model.dropout_layers)>0, "model must have dropout layers for BALD"
        model.set_eval_mode(enable_dropout=True)
        bald = BALD(strategy=self.strategy, aug_key=self.aug_key, k=self.k)
        probabilities, entropies = bald.get_probabilities_and_entropies(model, data)
        bald_scores = bald.get_scores_from_pe(probabilities, entropies)
        idxs_to_select = [torch.argmax(bald_scores).item()]
        scores = np.zeros((len(bald_scores)))
        while len(idxs_to_select)<batch_size:
            P_hat = self._sample_M_K(probabilities[idxs_to_select, :, :])
            Z = 1./torch.sum(P_hat, dim=1, keepdim=True).T.double()
            a_batchBALD = {}
            for i in range(len(data.pool_idxs)):
                if i not in idxs_to_select:
                    M = torch.matmul(P_hat, probabilities[i, :, :])
                    N = torch.log(M/self.k)
                    P = torch.mul(M, N)
                    joint_entropy = -torch.sum(torch.matmul(Z, P)).item()/P.shape[0]
                    mean_conditional_entropy_added = torch.mean(entropies[i, :])
                    a_batchBALD[i] = joint_entropy - mean_conditional_entropy_added
            idx_to_select = max(a_batchBALD.keys(), key=lambda i: a_batchBALD[i])
            idxs_to_select.append(idx_to_select)
            scores[idx_to_select] = 1.
        return scores