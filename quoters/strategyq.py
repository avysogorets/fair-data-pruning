from .quoter_base import QuoterBase
from .utils import get_class_sizes
from torch.utils.data import Subset
import json
import os


class StrategyQ(QuoterBase):
    """ Class-quotas extracted from a reslt file
    """
    def __init__(self, **kwargs):
        super().__init__()

    def __call__(self, res_path, res_fileid, data, **kwargs):
        res_filename = os.path.join(res_path, res_fileid)
        f = open(res_filename)
        res = json.load(f)
        final_size = str(min([int(k) for k in res.keys()]))
        idxs = res[final_size]['idxs']
        selected_dataset = Subset(data.full_datasets["train"][False], idxs)
        class_sizes = get_class_sizes(selected_dataset, data.num_classes)
        class_quota = []
        for k in range(data.num_classes):
            class_quota.append(class_sizes[k]/sum(class_sizes))
        return class_quota

        


