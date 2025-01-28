from ..utils import aggregate_classwise_metrics
from .utils import get_class_sizes
from .quoter_base import QuoterBase


class DRoP(QuoterBase):
    def __init__(self, quoter_metric, **kwargs):
        super().__init__()
        self.quoter_metric = quoter_metric
        
    def __call__(self, data, metrics, select_size, **kwargs):
        class_metrics = aggregate_classwise_metrics(
            metrics=metrics,
            quoter_metric=self.quoter_metric,
            num_classes=data.num_classes)
        class_scores = [1-c for c in class_metrics]
        Z = sum(class_scores)
        select_sizes = [int(select_size*c/Z) for c in class_scores]
        pool_dataset = data.get_pool_dataset(False)
        class_sizes = get_class_sizes(pool_dataset, data.num_classes)
        error = 0
        active_classes = []
        for k in range(data.num_classes):
            if class_sizes[k]<select_sizes[k]:
                error += (select_sizes[k]-class_sizes[k])
                select_sizes[k] = class_sizes[k]
            else:
                active_classes.append(k)
        is_ok = False
        while error > 0 and not is_ok:
            Z = sum([class_scores[k] for k in active_classes])
            to_add = {k: int(error*class_scores[k]/Z) for k in active_classes}
            is_ok = True
            for k in active_classes:
                select_sizes[k] += to_add[k]
                error -= to_add[k]
                if class_sizes[k]<select_sizes[k]:
                    is_ok = False
                    error += (select_sizes[k]-class_sizes[k])
                    select_sizes[k] = class_sizes[k]
                    active_classes.remove(k)
        class_quota = [select_sizes[k]/select_size for k in range(data.num_classes)]
        return class_quota










