from ..globals import ACTIVE_LEARNING, DATA_PRUNING
from .scheduler_base import SchedulerBase


class Exponential(SchedulerBase):
    def __init__(self, 
            strategy,
            iterations,
            start_frac,
            final_frac,
            full_size,
            **kwargs):
        super().__init__()
        self.strategy = strategy
        self.iterations = iterations
        self.start_frac = start_frac
        self.final_frac = final_frac
        self.full_size = full_size
        self.dp = {}

    def __call__(self, it):
        if self.strategy == ACTIVE_LEARNING:
            if it in self.dp:
                return self.dp[it]
            r = (self.start_frac/self.final_frac)**(1./self.iterations)
            if it == 0:
                frac = self.final_frac*(r**(self.iterations-1))-self.start_frac
                batch_size = round(self.full_size*(frac)) 
            else:
                frac = self.final_frac*(r**(self.iterations-it-1))-self.start_frac
                batch_size = round(self.full_size*frac)-sum([self(t) for t in range(it)])
            self.dp[it] = batch_size
            return batch_size
        elif self.strategy == DATA_PRUNING:
            r = (self.final_frac/self.start_frac)**(1./self.iterations)
            frac = (r**(it+1))
            return round(self.full_size*frac)
        else:
            raise NotImplementedError(f"strategy <{self.strategy}> is unknown")