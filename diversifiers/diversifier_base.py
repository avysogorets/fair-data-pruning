from typing import List

class DiversifierBase:
    
    """ Base class for a Diversifier. Diversifiers augment
        the selection procedures beyond scores, often to
        diversify the selected data. See implemented
        diversifiers for examples.
    """

    def __init__(self) -> None:
        pass

    def fit(self, *args, **kwargs) -> None:
        """ (Optional) Invoked before the first __call__ 
            invocation. All potentially costly operations
            go here if multiple __call__(s) are expected.
        """
        pass

    def __call__(self, select_size: int) -> List:
        """ Returns a list of local indices to select.
        """
        raise NotImplementedError("implement the __call__ method")