from ..globals import ACTIVE_LEARNING, DATA_PRUNING
from .scheduler_base import SchedulerBase


class Linear(SchedulerBase):
    def __init__(self, 
            strategy,
            iterations,
            start_frac,
            final_frac,
            full_size):
        super().__init__()
        self.strategy = strategy
        self.iterations = iterations
        self.start_frac = start_frac
        self.final_frac = final_frac
        self.full_size = full_size

    def __call__(self, it):
        increment = abs(self.final_frac-self.start_frac)/self.iterations
        if self.strategy == ACTIVE_LEARNING:
            return int(increment*self.full_size)
        elif self.strategy == DATA_PRUNING:
            return int(self.full_size*(self.start_frac-(it+1)*increment))
        else:
            raise NotImplementedError(f"strategy <{self.strategy}> is unknown")