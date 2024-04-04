from .globals import ACTIVE_LEARNING, DATA_PRUNING
from tqdm.auto import tqdm
from datetime import datetime
from copy import deepcopy
from sklearn.metrics import precision_recall_fscore_support
import torch.nn.functional as F
import torch
import numpy as np


def should_early_stop(metrics, patience):
    losses = {epoch: metrics[epoch]['loss'] for epoch in metrics.keys()}
    losses = [losses[i] for i in sorted(losses.keys())]
    if len(losses)<patience:
        return False
    for i in range(2,patience+1):
        if losses[-i]>losses[-1]:
            return False
    return True


def compute_overall_metrics(y_true, outs, aggregation='macro'):
    y_pred = torch.argmax(outs, dim=1)
    metrics = precision_recall_fscore_support(
        y_true.cpu().numpy().reshape(-1), 
        y_pred.cpu().numpy().reshape(-1), 
        average=aggregation,
        zero_division=0)
    loss = F.cross_entropy(outs, y_true).item()
    metrics = {
        'precision': metrics[0],
        'recall': metrics[1],
        'f1-score': metrics[2],
        'loss': loss,
        'accuracy': (y_true == y_pred).float().mean().item()}
    return metrics


def compute_classwise_metrics(y_true, outs, num_classes):
    y_pred = torch.argmax(outs, dim=-1)
    metrics = []
    for k in range(num_classes):
        y_pred_k = deepcopy(y_pred)
        y_true_k = deepcopy(y_true)
        y_pred_k[y_pred!=k] = 0
        y_pred_k[y_pred==k] = 1
        y_true_k[y_true!=k] = 0
        y_true_k[y_true==k] = 1
        prfs = precision_recall_fscore_support(
            y_true_k.cpu().numpy().reshape(-1), 
            y_pred_k.cpu().numpy().reshape(-1),
            zero_division=0)
        loss = F.cross_entropy(outs[y_true==k],y_true[y_true==k]).item()
        # note that this can fail when there exists class such that
        # (1) there are no samples in that class, and 
        # (2) there are no predictions for that class.
        try:
            metrics.append({
                'precision': prfs[0][1],
                'recall': prfs[1][1],
                'f1-score': prfs[2][1],
                'accuracy': (y_true_k==y_pred_k).float().mean().item(),
                'loss': loss})
        except:
            metrics.append({
                'precision': 0,
                'recall': 0,
                'f1-score': 0,
                'accuracy': 0,
                'loss': float('inf')})
    return metrics


def get_train_information(trainer):
    epoch_str = get_epoch_str(trainer.epoch+1, trainer.num_epochs)
    information = (f"[epoch: {epoch_str}/{trainer.num_epochs}]"
            f"[lr: {trainer.optimizer.param_groups[0]['lr']:.4f}]"
            f"[train loss/acc: {trainer.metrics['train'][trainer.epoch]['overall']['loss']:.4f}/"
            f"{trainer.metrics['train'][trainer.epoch]['overall']['accuracy']:.4f}]"
            f"[val loss/acc: {trainer.metrics['val'][trainer.epoch]['overall']['loss']:.4f}/"
            f"{trainer.metrics['val'][trainer.epoch]['overall']['accuracy']:.4f}]")
    return information
        

def epochs_correction(data, epochs, lr_drops):
    """ Increase the number of epochs if the data is too limited
        The number of epochs is E(0.5+dL) where E is the full (max)
        number of epochs, f is current dataset density, and L is
        the full length of the dataset.
    """
    if epochs == 0:
        return 0, []
    original_length = data.full_length
    current_length = len(data.selected_idxs)
    C = 0.5
    coefficient = C+(1-C)*current_length/original_length
    current_epochs = int(coefficient * (epochs*original_length/current_length))
    lr_drops = [int(lr_drop*(current_epochs/epochs)) for lr_drop in lr_drops]
    return current_epochs, lr_drops


def aggregate_overall_metrics(metrics, quoter_metric='accuracy', **kwargs):
    num_inits = len(metrics)
    metric_stack = np.zeros((num_inits))
    for init_id in range(len(metrics)):
        if len(metrics[init_id].keys()) == 0:
            continue
        best_epoch = max(
                metrics[init_id].keys(),
                key=lambda x: metrics[init_id][x]['overall']['accuracy'])
        metric_i = metrics[init_id][best_epoch]['overall'][quoter_metric]
        metric_stack[init_id] = metric_i
    metric = np.mean(metric_stack)
    return metric


def aggregate_classwise_metrics(metrics, num_classes, quoter_metric='recall'):
    num_inits = len(metrics)
    metric_stack = np.zeros((num_inits, num_classes))
    for init_id in range(len(metrics)):
        best_epoch = max(
            metrics[init_id].keys(),
            key=lambda x: metrics[init_id][x]['overall']['accuracy'])
        for k in range(num_classes):
            if len(metrics[init_id].keys()) == 0:
                continue
            metric_i_k = metrics[init_id][best_epoch]['classwise'][k][quoter_metric]
            metric_stack[init_id][k] = metric_i_k
    metric_stack = metric_stack.mean(axis=0)
    return metric_stack


def dosample(local_idxs, all_idxs, select_size):
    dosample_idxs = np.random.choice(
            list(set(all_idxs)-set(local_idxs)),
            size=select_size-len(local_idxs),
            replace=False).tolist()
    new_local_idxs = local_idxs + dosample_idxs
    return new_local_idxs
    

def get_idxs_from_scores(
        strategy,
        diversifier,
        class_quotas,
        data,
        scores,
        select_size,
        embeddings):
    
    curr_length = len(data.selected_idxs)
    if strategy == ACTIVE_LEARNING:
        err_msg = f"curr length ({curr_length}) + select size ({curr_length}) >"\
                  f"full length ({data.full_length})"
        assert curr_length+select_size <= data.full_length, err_msg
    elif strategy == DATA_PRUNING:
        err_msg = f"select size ({curr_length}) > full length ({data.full_length})"
        assert select_size <= data.full_length, err_msg
    else:
        raise NotImplementedError(f"strategy <{strategy}> is unknown")
    ret_idxs = []
    if class_quotas is not None:
        pool_dataset = data.get_pool_dataset(False)
        pool_y = np.array([pool_dataset[i][1] for i in range(len(pool_dataset))])
        for k in tqdm(range(data.num_classes)):
            local_idxs_k = (torch.arange(len(pool_y))[pool_y==k]).numpy()
            embeddings_k = embeddings[local_idxs_k]
            scores_k = np.array(scores)[local_idxs_k].tolist()
            select_size_k = int(class_quotas[k]*select_size)
            diversifier.fit(data_X=embeddings_k, scores=scores_k)
            local_local_idxs_k = diversifier(select_size=select_size_k)
            ret_idxs += local_idxs_k[local_local_idxs_k].tolist()
    else:
        diversifier.fit(data_X=embeddings, scores=scores)
        ret_idxs = diversifier(select_size=select_size)
    new_local_idxs = dosample(ret_idxs, list(range(len(scores))), select_size)
    assert len(new_local_idxs) == len(set(new_local_idxs)), f"duplicates found in new idxs"
    assert len(new_local_idxs) == select_size, f"{len(new_local_idxs)} != {select_size}"
    return new_local_idxs


def get_current_gradients(model):
    gradients = []
    count = 0
    for module in model.modules():
        for named_param in module._parameters.items():
            _,param = named_param
            if param is None:
                continue
            if param.grad is None:
                continue
            curr_layer_grad = param.grad
            curr_layer_grad = curr_layer_grad.reshape(-1)
            gradients.append(curr_layer_grad.detach())
            count += curr_layer_grad.numel()
    gradients = torch.cat(gradients)
    return gradients


def get_specs_info(args):
    now = datetime.now()
    now_str = now.strftime("%d/%m/%Y %H:%M:%S")
    info_msg = (
            f"\n\n[seed          ] {args.seed}\n"
            f"[strategy      ] {args.strategy}\n"
            f"[iterations    ] {args.iterations}\n"
            f"[num inits     ] {args.num_inits}\n"
            f"[scorer name   ] {args.scorer_name}\n"
            f"[quoter name   ] {args.quoter_name}\n"
            f"[model name    ] {args.model_name}\n"
            f"[dataset name  ] {args.dataset_name}\n"
            f"[scheduler     ] {args.scheduler_name}\n"
            f"[diversifer    ] {args.diversifier_name}\n"
            f"[epochs query  ] {args.epochs_query}\n"
            f"[epochs final  ] {args.epochs_final}\n"
            f"[start frac    ] {args.start_frac}\n"
            f"[final frac    ] {args.final_frac}\n"
            f"[decay         ] {args.weight_decay}\n"
            f"[batch size    ] {args.batch_size}\n"
            f"[quoter metric ] {args.quoter_metric}\n"
            f"[timestamp     ] {now_str}\n")
    return info_msg


def announce_iter(logger, s):
    logger.info('='*100)
    logger.info('='*((100-len(s))//2)+s+'='*((100-len(s))-((100-len(s))//2)))
    logger.info('='*100+'\n')


def get_epoch_str(epoch, num_epochs):
    return '0'*(len(str(num_epochs))-len(str(epoch)))+str(epoch)


def get_iter_info(curr_iter, tot_iter, data, val_metrics, iter_time):
    num_train = len(data.selected_idxs)
    frac_train = num_train/data.full_length
    if len(val_metrics.keys()) > 0:
        best_key = min(val_metrics.keys(), key=lambda k: val_metrics[k]["overall"]["loss"])
        best_val_loss = val_metrics[best_key]["overall"]["loss"]
        best_val_acc = val_metrics[best_key]["overall"]["accuracy"]
        last_init_info = f"[last dev loss: {best_val_loss:.3f}][last dev acc: {best_val_acc:.3f}]"
    else:
        last_init_info = "[no training performed]"
    iter = get_epoch_str(curr_iter, tot_iter)
    info_msg = (f"[iter: {iter}/{tot_iter}]"
        f"{last_init_info}"
        f"[train idxs: {num_train} ({100*frac_train:.1f}%)]"
        f"[iter time: {iter_time:.0f}s]\n")
    return info_msg


def get_specs(args=None):
    """ The parameters that determine the name of the
        res fileid. Change per your needs.
    """

    spec_names = [
        "seed",
        "strategy", 
        "iterations",
        "num_inits",
        "scorer_name",
        "quoter_name",
        "model_name",
        "dataset_name",
        "scheduler_name",
        "start_frac",
        "final_frac",
        "diversifier_name"]    
    if args is not None:
        spec_vals = []
        for spec_name in spec_names:
            spec_vals.append(str(getattr(args, spec_name)))
    else:
        spec_vals = None
    return spec_names, spec_vals


def get_fileid(args):
    _, spec_vals = get_specs(args)
    return "_".join(spec_vals)


def get_all_subclasses(cls):
    all_subclasses = []
    for subclass in cls.__subclasses__():
        all_subclasses.append(subclass)
        all_subclasses.extend(get_all_subclasses(subclass))
    return all_subclasses


def set_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)