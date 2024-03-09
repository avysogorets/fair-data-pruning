from typing import Iterable


class ScorerBase:

    def __init__(self,
                strategy: int,
                aug_key: bool,
                **kwargs) -> None:
        self.aug_key = aug_key   # whether to use data augmentation while scoring;
        self.strategy = strategy # ACTIVE_LEARNING or DATA_PRUNING;
        self.in_train: bool      # whether scorer operates during training.

    def score(self, *args, **kwargs) -> Iterable[int]:
        """ Return a list of scores associated with the data.pool_dataset.
            Higher scores usually indicate "harder" samples that are normally
            more favorable for selection.
        """
        self.in_train = False
        raise NotImplementedError("implement score method.")