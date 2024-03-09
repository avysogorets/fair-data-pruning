from ..utils import get_all_subclasses
from .scheduler_base import SchedulerBase
from .linear import Linear
from .exponential import Exponential


def SchedulerFactory(scheduler_name, **kwargs):
    available_schedulers = {}
    for _class_ in get_all_subclasses(SchedulerBase):
        available_schedulers[_class_.__name__] = _class_
    if scheduler_name in available_schedulers:
        return available_schedulers[scheduler_name](**kwargs)
    else:
        raise NotImplementedError(f"undefined scheduler <{scheduler_name}>")