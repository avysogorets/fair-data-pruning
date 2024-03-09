from .quoter_base import QuoterBase
from .utils import get_class_sizes


class Uniform(QuoterBase):
    """ Uniform class quota: the selected dataset
        will respect the original class balance.
    """
    def __init__(self, **kwargs):
        super().__init__()

    def __call__(self, data, **kwargs):
        pool_dataset = data.get_pool_dataset(False)
        class_sizes = get_class_sizes(pool_dataset, data.num_classes)
        quota = []
        pool_length = sum(class_sizes)
        for k in range(data.num_classes):
            quota.append(class_sizes[k]/pool_length)
        return quota