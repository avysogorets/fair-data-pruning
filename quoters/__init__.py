from ..utils import get_all_subclasses
from .quoter_base import QuoterBase
from .auto import Auto
from .uniform import Uniform
from .strategyq import StrategyQ
from .normalize import Normalize
from .drop import DRoP


def QuoterFactory(quoter_name, **kwargs):
    available_quota = {}
    for _class_ in get_all_subclasses(QuoterBase):
        available_quota[_class_.__name__] = _class_
    if quoter_name in available_quota:
        return available_quota[quoter_name](**kwargs)
    else:
        raise NotImplementedError(f"undefined quoter {quoter_name}")