from ..globals import ACTIVE_LEARNING
from ..data.utils import TabularDataset
from ..models.utils import MLP
from .scorer_base import ScorerBase
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch

class DAL(ScorerBase):
    """ Discriminative Active Learning
        Source: https://arxiv.org/abs/1907.06347)
    """
    def __init__(self, dal_bs, dal_lr, **kwargs):
        super().__init__(**kwargs)
        self.dal_bs = dal_bs
        self.dal_lr = dal_lr
        self.in_train = False
        err_msg = "DAL is intended for active learning only"
        assert self.strategy == ACTIVE_LEARNING, err_msg

    def _train(self, discriminator, train_dataset, lr, max_epochs, batch_size, threshold):
        optimizer = torch.optim.SGD(discriminator.parameters(),lr=lr)
        train_dataloader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True)
        curr_acc = 0.
        epoch = 0
        while epoch < max_epochs and curr_acc < threshold:
            ys = []
            for X, y in train_dataloader:
                X = X.to(discriminator.device)
                y = y.to(discriminator.device)
                ys.append(y)
                discriminator.train()
                optimizer.zero_grad()
                out = discriminator(X)
                loss = F.cross_entropy(out,y)
                loss.backward()
                optimizer.step()
            ys = torch.cat(ys)
            discriminator.eval()
            probas = self._get_probs(discriminator, train_dataset)
            discriminator.train()
            this_acc = sum(ys==torch.argmax(probas, dim=-1)).item()
            curr_acc = max(this_acc, curr_acc)
            self.epoch += 1

    def _get_probs(self, discriminator, dataset):
        batch_size = min(len(dataset), 128)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        probs_pool = []
        with torch.no_grad():
            for batch in dataloader:
                X,_ = batch
                X = X.to(discriminator.device)
                out = discriminator(X)
                probs = F.softmax(out, dim=-1)
                probs_pool.append(probs[:,1])
            probs_pool = torch.cat(probs_pool)
        return probs_pool

    def score(self, model, data, **kwargs):
        model.set_eval_mode()
        selected_dataset = data.get_selected_dataset(self.aug_key)
        with torch.no_grad():
            features_selected = model.embeddings(selected_dataset)
            features_pool = model.embeddings(data.get_pool_dataset(self.aug_key))
        labels_selected = torch.zeros(len(selected_dataset)).long()
        labels_pool = torch.ones(len(data.pool_idxs)).long()
        device = model.device
        features = torch.vstack([features_selected, features_pool]).to(device)
        labels = torch.cat([labels_selected, labels_pool]).to(device)
        dataset = TabularDataset(features, labels)
        discriminator = MLP(
                in_shape=features.shape[1],
                hidden_size=256,
                depth=2,
                num_classes=2,
                dropout=0,
                device=device)
        self._train(discriminator=discriminator,
                train_dataset=dataset,
                lr=self.dal_lr,
                max_epochs=50000,
                batch_size=self.dal_bs,
                threshold=0.98)
        features_pool = features_pool.to(device)
        labels_pool = labels_pool.to(device)
        dataset_pool = TabularDataset(features_pool, labels_pool)
        scores = self._get_probs(discriminator, dataset_pool)
        return scores.cpu().numpy()