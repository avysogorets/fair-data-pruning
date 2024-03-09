from .globals import ACTIVE_LEARNING, DATA_PRUNING


def check_args(args):
    """ Also consult compatibility of scorers with strategies
        (see the assertions in their constructors).
    """
    assert not (args.final_frac < args.start_frac and args.strategy == ACTIVE_LEARNING)
    assert not (args.final_frac > args.start_frac and args.strategy == DATA_PRUNING)
    assert not (args.strategy == DATA_PRUNING and args.start_frac < 1)
    assert not (args.strategy == ACTIVE_LEARNING and args.start_frac == 1)
    assert not (args.iterations > 1 and args.quoter_name == 'StrategyQ')
    assert not (args.strategy == ACTIVE_LEARNING and args.quoter_name == 'StrategyQ')
    assert not (args.num_inits > 1 and args.scorer_name == 'BatchBALD')
    assert not (args.num_inits > 1 and args.scorer_name == 'CoreSet')
    assert not (args.diversifier_name != 'Auto' and args.scorer_name == 'BatchBALD')
    assert not (args.diversifier_name != 'Auto' and args.scorer_name == 'CoreSet')
    assert not (args.dropout == 0 and args.scorer_name == 'BALD')
    assert not (args.dropout == 0 and args.scorer_name == 'BatchBALD')
    assert args.J < args.epochs_query
