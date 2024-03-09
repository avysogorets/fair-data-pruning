from .quoter_base import QuoterBase


class Auto(QuoterBase):
    """ Auto: globally chosen class quotas from scores
    """
    def __init__(self, **kwargs):
        super().__init__()

    def __call__(self, **kwargs):
        return None