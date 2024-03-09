from ..utils import get_all_subclasses
from .diversifier_base import DiversifierBase
from .auto import Auto
from .cluster import Cluster
from .coreset import CoreSet


def DiversifierFactory(diversifier_name, **kwargs):
    available_diversifiers = {}
    for _class_ in get_all_subclasses(DiversifierBase):
        available_diversifiers[_class_.__name__] = _class_
    if diversifier_name in available_diversifiers:
        return available_diversifiers[diversifier_name](**kwargs)
    else:
        raise NotImplementedError(f"undefined diversifier {diversifier_name}")