class SchedulerBase:
    """ When data selction spans over multiple iterations
        (e.g., active learning), scheduler determines the
        number of samples selected on each iteration. Linear
        schedule is commonly used in the literature.
    """

    def __init__(self):
        self.strategy: int # ACTIVE LEARNING or DATA_PRUNING;
        self.iterations: int # data selection iterations;
        self.start_frac: float # initial density of the dataset;
        self.final_frac: float # target density of the dataset;
        self.full_size: int # full length of the dataset.
    
    def __call__(self, *args, **kwargs):
        """ Returns the fraction of samples with respect
        to full sizebe selected (with respect to full size).
        """
        raise NotImplementedError("implement the __call__ method")

