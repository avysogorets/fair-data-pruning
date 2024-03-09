from typing import Union
from torch.utils.data import Dataset, Subset


def get_class_sizes(
        dataset: Union[Dataset, Subset],
        num_classes: int):
    class_sizes = [0]*num_classes
    for _,y in dataset:
        class_sizes[y] += 1
    return class_sizes