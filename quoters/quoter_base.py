from ..data import DataBase
from typing import List


class QuoterBase:
    """ Base class for class quota.
    """

    def __init__(self):
        pass

    def __call__(self,
            data: DataBase,
            select_size: int,
            metrics: List, 
            res_path: str,
            res_fileid: str) -> List[float]:
        
        """ Returns K valid class-wise quotas for selection.
            Valid means: quotas approx. sum to one and the
            selected amount from each class should not exceed
            the amount available for selection in that class.
        """
        raise NotImplementedError("implement the __call__ method")