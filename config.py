from .globals import DATA_PRUNING


def config(args):

    if args.strategy == DATA_PRUNING:
        args.start_frac = 1.0

    if args.model_name=='LeNet300100':
        args.lr = 0.1
        args.weight_decay = 0.0005
        args.batch_size = 128
        args.momentum = 0.9
        args.dataset_name = 'MNIST'
        args.early_stopping = False
        args.epochs_final = 160
        args.epochs_query = 16
        args.lr_drops_final = [40, 80, 120]
        args.lr_drops_query = [4, 8, 12]

    if args.model_name=='LeNet5':
        args.lr = 0.1
        args.weight_decay = 0.0005
        args.batch_size = 128
        args.momentum = 0.9
        args.dataset_name = 'CIFAR10'
        args.early_stopping = False
        args.epochs_final = 300
        args.epochs_query = 30
        args.lr_drops_final = [75, 150, 225]
        args.lr_drops_query = [8, 15, 23]

    if args.model_name=='VGG16':
        args.lr = 0.1
        args.weight_decay = 0.0001
        args.batch_size = 128
        args.momentum = 0.9
        args.dataset_name = 'CIFAR10'
        args.early_stopping = False
        args.epochs_final = 160
        args.epochs_query = 16
        args.lr_drops_final = [80, 120]
        args.lr_drops_query = [8, 12]

    if args.model_name=='VGG19':
        args.lr = 0.1
        args.weight_decay = 0.0005
        args.batch_size = 128
        args.momentum = 0.9
        args.dataset_name = 'CIFAR100'
        args.early_stopping = False
        args.epochs_final = 160
        args.epochs_query = 16
        args.lr_drops_final = [80, 120]
        args.lr_drops_query = [8, 12]

    if args.model_name=='ResNet18':
        args.lr = 0.2
        args.weight_decay = 0.0001
        args.batch_size = 256
        args.momentum = 0.9
        args.dataset_name = 'TinyImageNet'
        args.early_stopping = False
        args.epochs_final = 200
        args.epochs_query = 20
        args.lr_drops_final = [100, 150]
        args.lr_drops_query = [10, 15]

    if args.model_name=='ResNet50':
        args.lr = 0.4
        args.weight_decay = 0.0001
        args.batch_size = 512
        args.momentum = 0.9
        args.dataset_name = 'ImageNet'
        args.early_stopping = False
        args.epochs_final = 90
        args.epochs_query = 9
        args.lr_drops_final = [30, 60, 80]
        args.lr_drops_query = [3, 6, 8]