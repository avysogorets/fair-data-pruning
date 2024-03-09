from ..data_base import DataBase
from torch.utils.data import Subset
import numpy as np


class VisionDataBase(DataBase):

    def __init__(self,
            strategy,
            start_frac,
            **kwargs):

        self.strategy = strategy
        self.full_length = len(self.full_datasets["train"][False])
        test_idxs_y = {y:[] for y in range(self.num_classes)}
        for i,(_,y) in enumerate(self.full_datasets['test']):
            test_idxs_y[y].append(i)
        test_frac = 0.5 # how much test data to hold-out for validation
        new_test_idxs = []
        new_val_idxs = []
        for y in range(self.num_classes):
            cutoff = int(test_frac*len(test_idxs_y[y]))
            new_test_idxs += test_idxs_y[y][cutoff:]
            new_val_idxs += test_idxs_y[y][:cutoff]
        new_test = Subset(self.full_datasets['test'], new_test_idxs)
        new_val = Subset(self.full_datasets['test'], new_val_idxs)
        self.full_datasets['val'] = new_val
        self.full_datasets['test'] = new_test
        if start_frac < 1:
            self.selected_idxs = np.random.choice(
                range(self.full_length),
                size=int(start_frac*self.full_length),
                replace=False).tolist()
        else:
            self.selected_idxs = list(range(self.full_length))
        super().__init__()