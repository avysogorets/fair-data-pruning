from .globals import *
from .models import ModelFactory
from .data import DataFactory
from .scorers import ScorerFactory
from .schedulers import SchedulerFactory
from .quoters import QuoterFactory
from .diversifiers import DiversifierFactory
from .utils import get_idxs_from_scores, \
    get_fileid, \
    set_seeds, \
    announce_iter, \
    get_specs_info, \
    epochs_correction, \
    get_iter_info
from .config import config
from .trainer import Trainer
from .metrics import Metrics
from .check import check_args
from tqdm.auto import tqdm
import os
import numpy as np
import torch
import time
import argparse
import logging.config

parser = argparse.ArgumentParser()

# general parameters
parser.add_argument('--save', default=True, action='store_true', help='save output files')
parser.add_argument('--test', default=False, action='store_true', help='short training cycle')
parser.add_argument('--use_gpu', default=False, action='store_true', help='whether to use GPU (if available)')
parser.add_argument('--use_ckpt', default=False, action='store_true', help='whether to use checkpoints')
parser.add_argument('--auto_config', default=True, action='store_true', help='auto config hyperparameters')
parser.add_argument('--seed', type=int, default=42, help='global seed')
parser.add_argument('--res_path', type=str, default='fair-data-pruning/res', help='path to results/output')
parser.add_argument('--log_path', type=str, default='fair-data-pruning/log', help='path to execution logs')
parser.add_argument('--ckpt_path', type=str, default='fair-data-pruning/ckpt', help='path to checkpoints')

# data parameters
parser.add_argument('--aug_score', default=False, action='store_true', help='augment dataset when scoring')
parser.add_argument('--aug_query', default=False, action='store_true', help='augment dataset when training a query model')
parser.add_argument('--aug_final', default=True, action='store_true', help='augment dataset when training a final model')
parser.add_argument('--dataset_name', type=str, default='MNIST', help='dataset name')
parser.add_argument('--data_path', type=str, default='datasets')

# model parameters
parser.add_argument('--model_name', type=str, default='LeNet300100', help='model name')
parser.add_argument('--dropout', type=float, default=0, help='dropout rate (between 0 and 1)')

# trainer parameters
parser.add_argument('--early_stopping', default=False, action='store_true', help='use validation loss early stopping')
parser.add_argument('--cdbw_query', default=False, action='store_true', help='query robust optimization by CDB-W')
parser.add_argument('--cdbw_final', default=False, action='store_true', help='final robust optimization by CDB-W')
parser.add_argument('--epochs_query', type=int, default=16, help='training epochs for query model')
parser.add_argument('--epochs_final', type=int, default=160, help='training epochs for final model')
parser.add_argument('--batch_size', type=int, default=16, help='training batch size')
parser.add_argument('--lr', type=float, default=1e-6, help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
parser.add_argument('--lr_drops_query', type=int, nargs='+', default=[50,100], help='lr drop epochs for query model')
parser.add_argument('--lr_drops_final', type=int, nargs='+', default=[5,10], help='lr drop epochs for final model')
parser.add_argument('--weight_decay', type=float, default=0, help='L2 weight decay')
parser.add_argument('--patience', type=int, default=5, help='early stpping patience (in validation epochs)')

# quoter parameters
parser.add_argument('--quoter_name', type=str, default='Auto', help='method for class-wise quota (Auto means none)')
parser.add_argument('--quoter_metric', type=str, default='recall', help='performance metric for MetriQ quoter')
parser.add_argument('--strategyq_filepath', type=str, default=None, help='[StrategyQ] filepath where to extract quotas from')

# selection parameters
parser.add_argument('--strategy', type=int, default=DATA_PRUNING, help='selection strategy')
parser.add_argument('--start_frac', type=float, default=1.0, help='start subset size')
parser.add_argument('--final_frac', type=float, default=0.5, help='final subset size')
parser.add_argument('--num_inits', type=int, default=1, help='the number of initializations to average across')
parser.add_argument('--scheduler_name', type=str, default='Linear', help='data selecting scheduler')
parser.add_argument('--iterations', type=int, default=1, help='number of selecting iterations')

# diversifier parameters
parser.add_argument('--diversifier_name', type=str, default='Auto', help='diversifier strategy (Auto = None)')
parser.add_argument('--num_clusters', type=int, default=20, help='[Cluster] number of clusters')
parser.add_argument('--cluster_metric', type=str, default='euclidean', help='[Cluster] metric for clustering')
parser.add_argument('--merge_criterion', type=str, default='aggavg', help='[Cluster] merge criterion from agglomerative clustering')
parser.add_argument('--sampling_method', type=str, default='uniform', help='[Cluster] method for sampling from the clusters')

# scorer parameters
parser.add_argument('--scorer_name', type=str, default='Random', help='subsetter method')
parser.add_argument('--ly_name', type=str, default='oracle', help='aggregator over classes: min / max / avg')
parser.add_argument('--dal_bs', type=int, default=10, help='[DAL] batch size')
parser.add_argument('--dal_lr', type=float, default=1e-2, help='[DAL] discriminator learning rate')
parser.add_argument('--J', type=int, default=10, help='[DynamicUncertainty] sliding window size over epochs')
parser.add_argument('--bald_k', type=int, default=5, help='[BALD] number of samples for approximate inference')

args = parser.parse_args()

def main(args):

    # Configure & initialize
    set_seeds(args.seed)
    if torch.cuda.is_available() and args.use_gpu:
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    if args.auto_config:
        """ Caution: some user-specified cla might be
            overriden. See config.py for further details.
        """
        config(args)
    check_args(args)
    os.makedirs(args.res_path, exist_ok=True)
    os.makedirs(args.log_path, exist_ok=True)
    os.makedirs(args.ckpt_path, exist_ok=True)
    fileid = get_fileid(args)
    res_filename = '.'.join([fileid, 'json'])
    data = DataFactory(
            dataset_name=args.dataset_name,
            strategy=args.strategy,
            path=args.data_path,
            start_frac=args.start_frac,
            device=device)
    scheduler = SchedulerFactory(
            args.scheduler_name,
            strategy=args.strategy,
            full_size=data.full_length,
            iterations=args.iterations,
            start_frac=args.start_frac,
            final_frac=args.final_frac)
    quoter = QuoterFactory(
            quoter_name=args.quoter_name,
            model_name=args.model_name,
            quoter_metric=args.quoter_metric,
            strategyq_filepath=args.strategyq_filepath,
            num_classes=data.num_classes)

    # Check if results already exists
    if res_filename in os.listdir(args.res_path):
        print('[res file for this configuration already exists.]')
        return

    # Load the right data object from checkpoints
    ckpt_fileid = '_'.join([fileid, 'ckpt.pt'])
    ckpt_file = os.path.join(args.ckpt_path, ckpt_fileid)
    if os.path.isfile(ckpt_file) and args.use_ckpt:
        ckpt = torch.load(ckpt_file, map_location=device)
        start_iter = ckpt['iter_id']
        start_init = ckpt['init_id']
        data.set_checkpoint_dict(ckpt)
    else:
        start_iter = 0
        start_init = 0
    
    # Create logger
    log_filename = '.'.join([fileid, 'log'])
    log_filepath = os.path.join(args.log_path, log_filename)
    logging.basicConfig(
            filename=log_filepath,
            encoding='utf-8',
            filemode='w',
            level=logging.INFO)
    print(f'[logs saved to {log_filepath}]')
    logger = logging.getLogger(__name__)
    specs_info = get_specs_info(args)
    logger.info(specs_info)

    # Begin the data selection loop
    metrics = Metrics()
    iter_time = 0
    for it in tqdm(range(start_iter, args.iterations)):
        s = f' ITER #{it+1}/{args.iterations}: TRAINING {args.num_inits} QUERY MODELS '
        announce_iter(logger, s)
        iter_start = time.time()
        select_size = scheduler(it)
        iter_scores = []
        iter_results = {'train': [], 'val': [], 'test': []}
        for init_id in range(start_init, args.num_inits):
            scorer = ScorerFactory(
                    strategy=args.strategy,
                    scorer_name=args.scorer_name,
                    aug_key=args.aug_score,
                    data_length=data.full_length,
                    k=args.bald_k,
                    dal_bs=args.dal_bs,
                    dal_lr=args.dal_lr,
                    J=args.J,
                    ly_name=args.ly_name)
            model_query = ModelFactory(
                    model_name=args.model_name,
                    in_shape=data.in_shape,
                    num_classes=data.num_classes,
                    dropout=args.dropout,
                    device=device)
            epochs, lr_drops = epochs_correction(
                    lr_drops=args.lr_drops_query,
                    epochs=args.epochs_query,
                    data=data)
            trainer = Trainer(
                    model=model_query,
                    data=data,
                    aug_key=args.aug_query,
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                    batch_size=args.batch_size,
                    early_stopping=False,
                    patience=0,
                    epochs=epochs,
                    lr_drops=lr_drops,
                    init_id=init_id,
                    iter_id=it,
                    ckpt_path=args.ckpt_path,
                    fileid=fileid,
                    scorer=scorer,
                    select_size=select_size,
                    verbose=True,
                    use_ckpt=args.use_ckpt,
                    cdbw=args.cdbw_query)
            iter_init_results, scores = trainer.train()
            iter_results['train'].append(iter_init_results['train'])
            iter_results['val'].append(iter_init_results['val'])
            iter_results['test'].append(iter_init_results['test'])
            iter_scores.append(scores)

        # Select data after the iteration
        iter_scores = np.mean(iter_scores, axis=0).tolist()
        class_quotas = quoter(
                data=data,
                select_size=select_size,
                metrics=iter_results['val'],
                iter_scores=iter_scores)
        diversifier = DiversifierFactory(
                diversifier_name=args.diversifier_name,
                num_clusters=args.num_clusters,
                cluster_metric=args.cluster_metric,
                merge_criterion=args.merge_criterion,
                sampling_method=args.sampling_method)
        full_embeddings = model_query.embeddings(data.full_datasets["train"][False])
        local_idxs = get_idxs_from_scores(
                strategy=args.strategy,
                embeddings=full_embeddings,
                data=data,
                diversifier=diversifier,
                class_quotas=class_quotas,
                scores=iter_scores,
                select_size=select_size)
        data.register_selected_idxs(local_idxs)
        iter_time = time.time()-iter_start
        iter_info = get_iter_info(
                curr_iter=it+1, 
                tot_iter=args.iterations,
                data=data,
                val_metrics=iter_init_results['val'],
                iter_time=iter_time)
        logger.info(iter_info)

    # Train the final model on selected data
    s = f' TRAINING FINAL MODEL '
    start_time = time.time()
    announce_iter(logger, s)
    model_final = ModelFactory(
            model_name=args.model_name,
            in_shape=data.in_shape,
            num_classes=data.num_classes,
            dropout=args.dropout,
            device=device)
    epochs, lr_drops = epochs_correction(
            lr_drops=args.lr_drops_final,
            epochs=args.epochs_final,
            data=data)
    trainer = Trainer(
            model=model_final,
            data=data,
            aug_key=args.aug_final,
            lr=args.lr,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            early_stopping=args.early_stopping,
            patience=args.patience,
            epochs=epochs,
            lr_drops=lr_drops,
            init_id=-1,
            iter_id=args.iterations,
            ckpt_path=args.ckpt_path,
            fileid=fileid,
            scorer=None,
            select_size=None,
            verbose=True,
            use_ckpt=args.use_ckpt,
            cdbw=args.cdbw_final)
    results,_ = trainer.train()
    iter_info = get_iter_info(
            curr_iter=args.iterations, 
            tot_iter=args.iterations,
            data=data,
            val_metrics=results['test'],
            iter_time=time.time()-start_time)
    logger.info(iter_info)
    metrics.add(idxs=data.selected_idxs, metrics=results)

    # Save the results
    if args.save:
        metrics.save(args.res_path, res_filename)

    
if __name__=="__main__":
    main(args)