from .models import ClassificationModelBase
from .data import DataBase
from .scorers import ScorerBase
from .utils.utils import aggregate_classwise_metrics, \
    get_epoch_str, \
    compute_overall_metrics, \
    compute_classwise_metrics, \
    should_early_stop, \
    get_train_information
from .data.utils import IndexedDatasetWrapper
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from typing import List
import torch.nn.functional as F
import numpy as np
import torch
import logging
import os
import time

class Trainer:
    def __init__(self,
            model: ClassificationModelBase,
            data: DataBase,
            aug_key: bool,
            lr: float,
            weight_decay: float,
            batch_size: int,
            early_stopping: bool,
            patience: bool,
            epochs: int,
            lr_drops: List[int],
            init_id: int,
            iter_id: int,
            ckpt_path: str,
            fileid: str,
            scorer: ScorerBase,
            select_size: int,
            verbose: bool = True,
            use_ckpt: bool = False,
            cdbw: bool = False,
            **kwargs):

        self.model = model
        self.data = data
        self.aug_key = aug_key
        self.num_epochs = epochs
        self.iter_id = iter_id
        self.scorer = scorer
        self.select_size = select_size
        self.use_ckpt = use_ckpt
        self.early_stopping = early_stopping
        self.patience = patience
        self.ckpt_path = ckpt_path
        self.fileid = fileid
        self.cdbw = cdbw
        self.optimizer = torch.optim.SGD([
            {'params': self.model.parameters(),
             'lr': lr,
             'weight_decay': weight_decay}],
             momentum=0.9)
        self.lr_scheduler = MultiStepLR(self.optimizer, milestones=lr_drops, gamma=0.2)
        self.batch_size = batch_size
        self.logger = logging.getLogger(__name__)
        self.epochs = epochs
        self.metrics = {'train': {}, 'val': {}, 'test': {}}
        self.verbose = verbose
        self.init_id = init_id
        self.scores = np.zeros(len(data.selected_idxs))
        self.epoch = 0
        self.weights = torch.ones(data.num_classes).to(model.device)
        if self.use_ckpt:
            self._load_checkpoint()
    
    def train(self):
        self.start_time = time.time()
        train_dataset = self.data.get_selected_dataset(self.aug_key)
        train_dataloader = DataLoader(
                IndexedDatasetWrapper(train_dataset),
                batch_size=self.batch_size,
                shuffle=True)
        while self.epoch < self.num_epochs:
            epoch_outs = []
            epoch_y = []
            for idxs,(X,y) in train_dataloader:
                if self.cdbw and self.epoch > 0:
                    class_metrics = aggregate_classwise_metrics(
                        metrics=[{'0': self.metrics['val'][self.epoch-1]}],
                        quoter_metric='recall',
                        num_classes=self.data.num_classes)
                    for k in range(self.data.num_classes):
                        self.weights[k] = 1-class_metrics[k]
                X = X.to(self.model.device)
                y = y.to(self.model.device)
                self.model.set_train_mode()
                self.optimizer.zero_grad()
                outs = self.model(X)
                loss = F.cross_entropy(outs, y, weight=self.weights)
                epoch_outs.append(outs)
                epoch_y.append(y)
                loss.backward()
                self.optimizer.step()
                if self.scorer is not None and self.scorer.in_train:
                    self.scores = self.scorer.score(
                            idxs=idxs,
                            logits=outs.detach(),
                            y=y,
                            iter=len(epoch_outs))
            self.lr_scheduler.step()
            epoch_outs = torch.vstack(epoch_outs)
            epoch_y = torch.cat(epoch_y)
            self.metrics['train'][self.epoch] = {
                'overall': compute_overall_metrics(epoch_y, epoch_outs),
                'classwise': compute_classwise_metrics(epoch_y, epoch_outs, self.data.num_classes)}
            self.metrics['val'][self.epoch] = self.validate(self.data.full_datasets['val'])
            self.metrics['test'][self.epoch] = self.validate(self.data.full_datasets['test'])
            if self.verbose:
                train_information = get_train_information(self)
                self.logger.info(train_information)
            if self.early_stopping and should_early_stop(self.metrics['val'], self.patience):
                self.on_train_end()
                return self.metrics, self.scores
            self.epoch +=1
            if self.use_ckpt:
                self._save_checkpoint()
        self.on_train_end()
        return self.metrics, self.scores

    def on_train_end(self):
        if self.init_id >= 0:
            last_epoch = max(self.metrics['val'].keys()) if self.epoch>0 else -1
            if last_epoch < 0:
                end_msg = '[no training performed]'
            else:
                val_loss = self.metrics['val'][last_epoch]['overall']['loss']
                val_acc = self.metrics['val'][last_epoch]['overall']['accuracy']
                epoch_str = get_epoch_str(last_epoch+1, self.num_epochs)
                end_msg = f'\n[init #{self.init_id+1} completed]'\
                        f'[epochs: {self.num_epochs}]'\
                        f'[{epoch_str}/{self.num_epochs}]'\
                        f'[loss: {val_loss:.3f}][acc: {val_acc:.3f}]'\
                        f'[time: {time.time()-self.start_time:.0f}]\n'
            self.logger.info(end_msg)
        if self.scorer is not None and not self.scorer.in_train:
            self.scores = self.scorer.score(
                    model=self.model,
                    data=self.data,
                    batch_size=self.select_size)

    def validate(self, dataset):
        self.model.set_eval_mode()
        dev_dataloader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False)
        epoch_outs = []
        epoch_y = []
        self.model.eval()
        with torch.no_grad():
            for X, y in dev_dataloader:
                X = X.to(self.model.device)
                y = y.to(self.model.device)
                outs = self.model(X)
                epoch_outs.append(outs)
                epoch_y.append(y)
        epoch_outs = torch.vstack(epoch_outs)
        epoch_y = torch.cat(epoch_y)
        metrics = {
            'overall': compute_overall_metrics(epoch_y, epoch_outs),
            'classwise': compute_classwise_metrics(epoch_y, epoch_outs, self.data.num_classes)}
        return metrics

    def _save_checkpoint(self):
        trainer_ckpt = {
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'lr_scheduler': self.lr_scheduler.state_dict(),
                'epoch': self.epoch,
                'metrics': self.metrics,
                'scores': self.scores,
                'weights': self.weights.cpu(),
                'init_id': self.init_id,
                'iter_id': self.iter_id}
        data_ckpt = self.data.get_checkpoint_dict()
        ckpt = {**trainer_ckpt, **data_ckpt}
        fileid = '_'.join([self.fileid, 'ckpt.pt'])
        torch.save(ckpt, os.path.join(self.ckpt_path, fileid))

    def _load_checkpoint(self):
        fileid = '_'.join([self.fileid, 'ckpt.pt'])
        ckpt_file = os.path.join(self.ckpt_path, fileid)
        if os.path.isfile(ckpt_file):
            ckpt = torch.load(ckpt_file, map_location=self.model.device)
            iter_id_same = self.iter_id == ckpt['iter_id']
            init_id_same = self.init_id == ckpt['init_id']
            if not (iter_id_same and init_id_same):
                return
            self.model.load_state_dict(ckpt['model'])
            self.optimizer.load_state_dict(ckpt['optimizer'])
            self.lr_scheduler.load_state_dict(ckpt['lr_scheduler'])
            self.epoch = ckpt['epoch']
            self.metrics = ckpt['metrics']
            self.scores = ckpt['scores']
            self.weights = ckpt['weights'].to(self.model.device)
            self.data.set_checkpoint_dict(ckpt)
            msg = f'[valid checkpoint found; resuming from epoch {self.epoch}]'
        else:
            msg = '[no valid checkpoint found; starting training from scratch]'
        self.logger.info(msg)