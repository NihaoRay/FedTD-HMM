'''
Client Selection for Federated Learning
'''
import argparse
import os
import sys
import time
from utils.dataset import load_dataset,load_exist
from utils.util import init_prop, params_tolist, params_tomodel
from models.model import LeNet_mnist,CNN_fmnist,resnet20,CNN_cifar
import utils.min_hash as lsh

AVAILABLE_WANDB = False
try:
    import wandb
except ModuleNotFoundError:
    AVAILABLE_WANDB = False

import torch
import random

from data import *
from model import *
from FL_core.server import Server
from FL_core.client_selection import *
from FL_core.federated_algorithm import *
from utils import utils
# from utils.argparse import get_args

ALL_METHODS = [
    'Random', 'Cluster1', 'Cluster2', 'Pow-d', 'AFL', 'DivFL', 'GradNorm'
]

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=str, default='0', help='gpu cuda index')

    parser.add_argument('--model', type=str, default='CNN', help='model', choices=['BLSTM', 'CNN', 'ResNet'])
    parser.add_argument('--method', type=str, default='Pow-d', help='client selection',
                        choices=ALL_METHODS)
    parser.add_argument('--fed_algo', type=str, default='FedAvg', help='Federated algorithm for aggregation',
                        choices=['FedAvg', 'FedAdam'])

    # dataset
    parser.add_argument('--dataset', type=str, default='FederatedEMNIST', help='dataset',
                        choices=['MNIST', 'FashionMNIST', 'CIFAR10', 'CIFAR100', 'Reddit', 'FederatedEMNIST',
                                 'FedCIFAR100', 'CelebA', 'PartitionedCIFAR10', 'FederatedEMNIST_IID',
                                 'FederatedEMNIST_nonIID'])
    # parser.add_argument('--data_dir', type=str, default='/home/chenrui/workspace/ml_dataset',
    #                     help='dataset directory')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='dataset directory')
    parser.add_argument("--data_dir_path", type=str, default="data_dir/", help="directory of logs")  # 替换

    # data split
    parser.add_argument('--n_shards', type=int, default=5, help='number of shards')
    parser.add_argument('--alpha', type=float, default=1, help='parameter of dirichlet')
    parser.add_argument('--sgm', type=float, default=0.3, help='parameter of unbalance')
    parser.add_argument('--split', type=str, default='noniid', help='split method: iid or non-iid')
    parser.add_argument('--noniid_method', type=str, default='dirichlet',
                        help='noniid method: pathological or dirichlet')

    # client的num
    parser.add_argument('--n_clients', type=int, default=8, metavar='N',
                        help='how many training processes to use (default: 10)')
    parser.add_argument('--isSpars', type=str, default='topk', help='sparsification method: topk or randk or topk')
    # parser.add_argument('--enc', type=bool, default=True,help='enc or not')


    # sparsification
    parser.add_argument('--topk', type=float, default=0.2, help='sparfication fraction')
    # selection
    parser.add_argument('--sim_len', type=int, default=200, help='lsh matrix width')
    parser.add_argument('--quan_bits', type=int, default=16, help='quantification bits')
    parser.add_argument('--enc_batch_size', type=int, default=4096, help='Batch Encryption size')
    parser.add_argument('--randk_seed', type=int, default=12, help='random k packages seed')
    # optimizer
    parser.add_argument('--client_optimizer', type=str, default='sgd', choices=['sgd', 'adam'], help='client optim')
    parser.add_argument('--lr_local', type=float, default=0.1, help='learning rate for client optim')
    parser.add_argument('--lr_global', type=float, default=0.001, help='learning rate for server optim')
    parser.add_argument('--wdecay', type=float, default=0, help='weight decay for optim')
    parser.add_argument('--momentum', type=float, default=0, help='momentum for SGD')

    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for Adam')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam')
    parser.add_argument('--epsilon', type=float, default=1e-8, help='epsilon for Adam')

    parser.add_argument('--alpha1', type=float, default=0.75, help='alpha1 for AFL')
    parser.add_argument('--alpha2', type=float, default=1, help='alpha2 for AFL')
    parser.add_argument('--alpha3', type=float, default=0.1, help='alpha3 for AFL')

    # training setting
    parser.add_argument('-E', '--num_epoch', type=int, default=1, help='number of epochs')
    parser.add_argument('-B', '--batch_size', type=int, default=64, help='batch size of each client data')
    parser.add_argument('-R', '--num_round', type=int, default=400, help='total number of rounds')
    parser.add_argument('-A', '--num_clients_per_round', type=int, default=8, help='number of participated clients')
    parser.add_argument('-K', '--total_num_clients', type=int, default=8,
                        help='total number of clients')  # 保持和 --n_clients 一致

    parser.add_argument('-u', '--num_updates', type=int, default=None, help='number of updates')
    parser.add_argument('-n', '--num_available', type=int, default=None,
                        help='number of available clients at each round')
    parser.add_argument('-d', '--num_candidates', type=int, default=8,
                        help='buffer size; d of power-of-choice')  # 保持和 --n_clients 一致

    parser.add_argument('--loss_div_sqrt', action='store_true', default=False, help='loss_div_sqrt')
    parser.add_argument('--loss_sum', action='store_true', default=False, help='sum of losses')
    parser.add_argument('--num_gn', type=int, default=0, help='number of group normalization')

    parser.add_argument('--distance_type', type=str, default='L1', help='distance type for clustered sampling 2')
    parser.add_argument('--subset_ratio', type=float, default=None, help='subset size for DivFL')

    parser.add_argument('--dirichlet_alpha', type=float, default=0.1,
                        help='ratio of data partition from dirichlet distribution')

    parser.add_argument('--min_num_samples', type=int, default=None, help='mininum number of samples')
    parser.add_argument('--schedule', type=int, nargs='+', default=[0, 5, 10, 15, 20, 30, 40, 60, 90, 140, 210, 300],
                        help='splitting points (epoch number) for multiple episodes of training')
    parser.add_argument('--maxlen', type=int, default=400, help='maxlen for NLP dataset')

    # experiment setting
    parser.add_argument('--fix_seed', action='store_true', default=False, help='fix random seed')
    parser.add_argument('--seed', type=int, default=0, help='seed')
    parser.add_argument('--parallel', action='store_true', default=False, help='use multi GPU')
    parser.add_argument('--use_mp', action='store_true', default=False, help='use multiprocessing')
    parser.add_argument('--nCPU', type=int, default=None, help='number of CPU cores for multiprocessing')
    parser.add_argument('--save_probs', action='store_true', default=False, help='save probs')
    parser.add_argument('--no_save_results', action='store_true', default=False, help='save results')
    parser.add_argument('--test_freq', type=int, default=1, help='test all frequency')

    parser.add_argument('--comment', type=str, default='', help='comment')
    args = parser.parse_args()
    return args

def model_init(dataset):
    """
    Model initialization.
    Args:
        dataset (`str`):
            Name of dataset.
    Returns:
        model (`OrderDict`):
            Model for dataset.
    """
    if dataset == 'MNIST':
        model = LeNet_mnist().to(args.device)
    elif dataset == 'FashionMNIST':
        model = CNN_fmnist().to(args.device)
        #model = resnet20(in_channels=1,num_classes=10).to(device)
    elif dataset == 'CIFAR10':
        model = resnet20(in_channels=3,num_classes=10).to(args.device)
    elif dataset == 'CIFAR100':
        model = resnet50(num_classes=100).to(args.device)
    else:
        raise ValueError("Datset name is invalid, please input MNIST, FashionMNIST, CIFAR10 or CIFAR100")
    return model


def federated_algorithm(dataset, model, args):
    train_sizes = dataset['train']['data_sizes']
    if args.fed_algo == 'FedAdam':
        return FedAdam(train_sizes, model, args=args)
    else:
        return FedAvg(train_sizes, model)

def federated_algorithm_add(dataset, model, args):
    train_sizes = [len(dataset[i]) for i in range(args.total_num_clients)]
    if args.fed_algo == 'FedAdam':
        return FedAdam(train_sizes, model, args=args)
    else:
        return FedAvg(train_sizes, model)


def client_selection_method(args):
    #total = args.total_num_client if args.num_available is None else args.num_available
    kwargs = {'total': args.n_clients, 'device': args.device}
    if args.method == 'Random':
        return RandomSelection(**kwargs)
    elif args.method == 'AFL':
        return ActiveFederatedLearning(**kwargs, args=args)
    elif args.method == 'Cluster1':
        return ClusteredSampling1(**kwargs, n_cluster=args.num_clients_per_round)
    elif args.method == 'Cluster2':
        return ClusteredSampling2(**kwargs, dist=args.distance_type)
    elif args.method == 'Pow-d':
        assert args.num_candidates is not None
        return PowerOfChoice(**kwargs, d=args.num_candidates)
    elif args.method == 'DivFL':
        assert args.subset_ratio is not None
        return DivFL(**kwargs, subset_ratio=args.subset_ratio)
    elif args.method == 'GradNorm':
        return GradNorm(**kwargs)
    else:
        raise('CHECK THE NAME OF YOUR SELECTION METHOD')

if __name__ == '__main__':
    # set up
    args = get_args()
    if args.comment != '': args.comment = '-'+args.comment
    #if args.labeled_ratio < 1: args.comment = f'-L{args.labeled_ratio}{args.comment}'
    if args.fed_algo != 'FedAvg': args.comment = f'-{args.fed_algo}{args.comment}'

    # save to wandb
    args.wandb = AVAILABLE_WANDB
    if args.wandb:
        wandb.init(
            project=f'AFL-{args.dataset}-{args.num_clients_per_round}-{args.num_available}-{args.total_num_clients}',
            name=f"{args.method}{args.comment}",
            config=args,
            dir='.',
            save_code=True
        )
        wandb.run.log_code(".", include_fn=lambda x: 'src/' in x or 'main.py' in x)

    # fix seed
    if args.fix_seed:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # device setting
    if args.gpu_id == 'cpu' or not torch.cuda.is_available():
        args.device = 'cpu'
    else:
        if ',' in args.gpu_id:
            os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_id
        args.device = torch.device(f"cuda:{args.gpu_id[0]}")
        torch.cuda.set_device(args.device)
        print('Current cuda device ', torch.cuda.current_device())

    # set data
    train_file = os.path.join(args.data_dir_path, args.dataset + '_train')
    if not os.path.exists(train_file):
        client_train_datasets, client_test_datasets, data_info, server_test_sets = load_dataset(args)
        print("Generate new files!")
    else:
        client_train_datasets, client_test_datasets, data_info, server_test_sets = load_exist(args)
        print("Load last files!")
    train_weights, test_weights = init_prop(client_train_datasets, client_test_datasets, args.n_clients)

    model_train = model_init(args.dataset) # 需要训练的模型
    dataset_backup = args.dataset

    # 计算模型的参数总数
    params_list, params_num, layer_shape = params_tolist(model_train)
    total_sum = sum(params_num.values())
    batch_num = int(np.ceil(total_sum / args.enc_batch_size))
    print(f"Total number of model parameters: {total_sum}, batch_num:{batch_num}")

    random_r = lsh.gen_random_R(input_len=total_sum, sim_len=args.sim_len)
    # print(f"main process>> random_r: {random_r}")
    # set model
    client_selection = client_selection_method(args)
    # fed_algo = federated_algorithm(dataset, model, args) # 聚合算法

    args.dataset = dataset_backup  # 恢复datasets指定
    fed_algo = federated_algorithm_add(client_train_datasets, model_train, args) # 聚合算法

    # 文件保存句柄，以便后续文件的保存不用再新建保存的文件
    files = utils.save_files(args)

    ## train
    # set federated optim algorithm
    ServerExecute = Server(args, client_selection, fed_algo, files,
                           model_train, client_train_datasets, client_test_datasets, train_weights, test_weights, total_sum, batch_num, random_r)
    ServerExecute.train()
