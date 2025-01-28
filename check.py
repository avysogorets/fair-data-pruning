from .globals import ACTIVE_LEARNING, DATA_PRUNING
import os


def check_args(args):
    """ Also consult compatibility of scorers with strategies
        (see the assertions in their constructors).
    """

    err_msg = f"start frac {args.start_frac} cannot be larger than final frac {args.final_frac} in active learning."
    assert not (args.final_frac < args.start_frac and args.strategy == ACTIVE_LEARNING), err_msg

    err_msg = f"start frac {args.start_frac} cannot be smaller than final frac {args.final_frac} in data pruning."   
    assert not (args.final_frac > args.start_frac and args.strategy == DATA_PRUNING), err_msg

    err_msg = f"StrategyQ is intended for data pruning only"
    assert not (args.strategy == ACTIVE_LEARNING and args.quoter_name == 'StrategyQ'), err_msg

    err_msg = f"BatchBALD is intended to be used with 1 query model"
    assert not (args.num_inits > 1 and args.scorer_name == 'BatchBALD'), err_msg

    err_msg = f"CoreSet is intended to be used with 1 query model"
    assert not (args.num_inits > 1 and args.scorer_name == 'CoreSet'), err_msg

    err_msg = f"BatchBALD can only be used with Auto diversifier"
    assert not (args.diversifier_name != 'Auto' and args.scorer_name == 'BatchBALD'), err_msg

    err_msg = f"CoreSet can only be used with Auto diversifier; also see CoreSet diversifier."
    assert not (args.diversifier_name != 'Auto' and args.scorer_name == 'CoreSet'), err_msg

    err_msg = "BALD requires non-zero dropout"
    assert not (args.dropout == 0 and args.scorer_name == 'BALD'), err_msg

    err_msg = "BatchBALD requires non-zero dropout"
    assert not (args.dropout == 0 and args.scorer_name == 'BatchBALD'), err_msg

    err_msg = f"Sliding window J ({args.J}) is larger than training length {args.epochs_query} of the query model."
    assert not (args.J > args.epochs_query and args.scorer_name == 'DynamicUncertainty'), err_msg

    err_msg = f"DroP requires a trained query model. Increase the number of query epochs ({args.epochs_query})"
    assert not (args.quoter_name == 'DRoP' and 0 == args.epochs_query), err_msg

    err_msg = f"File passed to StrtegyQ ({args.strategyq_filepath}) does not exist."
    assert args.quoter_name != 'StrategyQ' or os.path.exists(args.strategyq_filepath), err_msg
