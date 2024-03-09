from ..utils import get_all_subclasses
from .scorer_base import ScorerBase
from .random import Random
from .bald import BALD
from .batch_bald import BatchBALD
from .max_entropy import MaxEntropy
from .variation_ratio import VariationRatio
from .gradient import GradientBased
from .el2n import EL2N
from .coreset import CoreSet
from .dal import DAL
from .forgetting import Forgetting
from .dynamic_uncertainty import DynamicUncertainty
from .sspy import SSPY


def ScorerFactory(scorer_name, **kwargs):
    available_data_scorers = {}
    for _class_ in get_all_subclasses(ScorerBase):
        available_data_scorers[_class_.__name__] = _class_
    if scorer_name in available_data_scorers:
        return available_data_scorers[scorer_name](**kwargs)
    else:
        raise NotImplementedError(f"undefined data scorer <{scorer_name}>")