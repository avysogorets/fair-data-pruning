from .quoter_base import QuoterBase
from .utils import get_class_sizes
from torch.utils.data import Subset
import json
import os


class StrategyQ(QuoterBase):
    """ Class-quotas extracted from a result file res_path/res_fileid
        The apprpriate result format is assumed.
    """
    def __init__(self, strategyq_filepath, **kwargs):
        super().__init__()
        self.strategyq_filepath = strategyq_filepath

    def __call__(self, data, **kwargs):
        f = open(self.strategyq_filepath)
        res = json.load(f)
        final_size = str(min([int(k) for k in res.keys()]))
        idxs = res[final_size]['idxs']
        selected_dataset = Subset(data.full_datasets["train"][False], idxs)
        class_sizes = get_class_sizes(selected_dataset, data.num_classes)
        class_quota = []
        for k in range(data.num_classes):
            class_quota.append(class_sizes[k]/sum(class_sizes))
        return class_quota

        


