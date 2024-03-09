from ..globals import ACTIVE_LEARNING, DATA_PRUNING
from torch.utils.data import Dataset, Subset
from typing import List, Dict
import numpy as np
import torch


class DataBase:

    def __init__(self):
        self.strategy: int                     # ACTIVE_LEARNING or DATA_PRUNING;
        self.full_length: int                  # full dataset length;
        self.num_classes: int                  # number of classes; 
        self.full_datasets: Dict[str, Dataset] # dict with splits: 'train', 'val', 'test';
        self.selected_idxs: List[int]          # idxs selected for training.

    @property
    def deselected_idxs(self) -> Subset:
        all_idxs = range(self.full_length)
        selected_idxs = set(self.selected_idxs)
        return [i for i in all_idxs if i not in selected_idxs]
    
    @property
    def pool_idxs(self) -> Subset:
        if self.strategy == ACTIVE_LEARNING:
            return self.deselected_idxs
        elif self.strategy == DATA_PRUNING:
            return self.selected_idxs
        else:
            raise RuntimeError(f'unknown strategy <{self.strategy}>')
        
    def get_pool_dataset(self, aug_key: bool) -> Subset:
        return Subset(self.full_datasets['train'][aug_key], self.pool_idxs)
    
    def get_selected_dataset(self, aug_key: bool) -> Subset:
        return Subset(self.full_datasets['train'][aug_key], self.selected_idxs)
    
    def get_checkpoint_dict(self) -> Dict:
        ckpt_dict = {
            'selected_idxs': list(self.selected_idxs),
            'strategy': self.strategy}
        return ckpt_dict

    def set_checkpoint_dict(self, ckpt_dict: Dict) -> None:
        self.selected_idxs = ckpt_dict['selected_idxs']
        self.strategy = ckpt_dict['strategy']

    def register_selected_idxs(self, local_idxs: list) -> None:
        global_idxs = np.array(self.pool_idxs)[local_idxs].tolist()
        if self.strategy == ACTIVE_LEARNING:
            self._add_idxs_to_selected(global_idxs)
        elif self.strategy == DATA_PRUNING:
            self.selected_idxs = global_idxs
        else:
            raise RuntimeError(f'unknown strategy <{self.strategy}>')
    
    def _add_idxs_to_selected(self, idxs: list) -> None:
        intersection = set(self.selected_idxs).intersection(set(idxs))
        assert len(intersection)==0, (
                f"{len(intersection)} of the selected idxs are already selected")
        self.selected_idxs = self.selected_idxs + idxs