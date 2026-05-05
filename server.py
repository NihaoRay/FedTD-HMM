import torch
import wandb
from copy import deepcopy
from tqdm import tqdm
import numpy as np
import sys
import multiprocessing as mp
import random
import torch.optim as optim
import os
from utils.model_compress import model_compress, model_dec_compress
from utils.util import params_tomodel, params_tolist
from utils.hmm_online import GaussianHMM_Scratch, OnlineHMM_Monitor
import time

from .client import Client
from .client_selection.config import *

class Server(object):
    def __init__(self, args, selection, fed_algo, files,
                 model_train, client_train_datasets, client_test_datasets,
                 train_weights, test_weights, total_sum, batch_num, random_r):

        # 额外添加
        self.model_train = model_train

        self.client_train_datasets = client_train_datasets
        self.client_test_datasets = client_test_datasets
        self.train_weights = train_weights
        self.test_weights = test_weights
        self.total_sum = total_sum
        self.batch_num = batch_num
        self.record = {}
        self.random_r = random_r

        self.device = args.device
        self.args = args
        self.selection_method = selection
        self.federated_method = fed_algo
        self.files = files

        self.nCPU = mp.cpu_count() // 2 if args.nCPU is None else args.nCPU

        self.total_num_client = args.n_clients
        self.num_clients_per_round = args.num_clients_per_round
        self.num_available = args.num_available
        if self.num_available is not None:
            random.seed(args.seed)

        self.total_round = args.num_round
        self.save_results = not args.no_save_results
        self.save_probs = args.save_probs

        self.test_on_training_data = False

        ## INITIALIZE
        # initialize the training status of each client
        # self._init_clients(init_model)
        self._init_local_clients(model_train)

        if self.args.method in LOSS_THRESHOLD:
            self.ltr = 0.0

    # 额外添加
    def _init_local_clients(self, model_train):
        """
        initialize clients' model
        ---
        Args
            init_model: initial given global model
        """
        self.client_list = []
        for client_idx in range(self.total_num_client):
            # local_train_data = self.train_data[client_idx]
            # local_test_data = self.test_data[client_idx] if client_idx in self.test_clients else np.array([])
            local_train_data = self.client_train_datasets[client_idx]
            local_test_data = self.client_test_datasets[client_idx]
            c = Client(client_idx, local_train_data, local_test_data, deepcopy(model_train), self.random_r, self.args)
            self.client_list.append(c)


    def train(self):
        """
        FL training
        """
        ## ITER COMMUNICATION ROUND
        print("开始联邦学习HMM在线监测...")
        monitor = OnlineHMM_Monitor()

        for round_idx in range(self.total_round):
            print(f'\n>>server process:  ROUND {round_idx} / {self.total_round}')

            ## GET GLOBAL MODEL
            #self.global_model = self.trainer.get_model()
            self.global_model = self.model_train.to(self.device)

            # set clients
            client_indices = [*range(self.total_num_client)]

            ## CLIENT UPDATE (TRAINING)
            # local_losses, accuracy, local_metrics = self.train_clients(client_indices)
            local_losses, accuracy, local_metrics, local_models,  local_model_hash = self.train_clients_add(client_indices)

            rev_list = [0 for i in range(len(client_indices))]  # 此参数用于参数模型选择的，暂且用0选中的符号来代替

            # 压缩模型进行聚合
            aggr_begin = time.time()
            sum_weights, global_param = aggregatie_weights(local_models, rev_list, self.train_weights, self.batch_num,self.args)
            # 模型解压缩，其实这里不是真正的解压缩，只是权重恢复
            globel_list = model_dec_compress(global_param, sum_weights, self.args.enc_batch_size).tolist()
            model_params_list = globel_list[:self.total_sum] #模型的参数集合
            params_list, params_num, layer_shape = params_tolist(self.model_train) # 原始的模型序列化
            #模型真正的解压缩在这里，将聚合后的更新模型权重，覆盖 self.model_train 对应位置的权重，其余保持不变
            self.model_train = params_tomodel(deepcopy(self.model_train), model_params_list, params_num, layer_shape, self.args, params_list)
            aggr_end = time.time()
            # print(f">>server process: 模型完成聚合: epoch {round_idx}, 聚合的权重sum_weights: {sum_weights}, "
            #       f"aggr time: {aggr_end - aggr_begin}")
            print(f">>server process: has aggregated: epoch {round_idx}, aggregated sum_weights: {sum_weights}, "
                  f"aggr time: {aggr_end - aggr_begin}")

            # 模型测试
            metrics = self.test(self.model_train, self.total_num_client, phase='test')

            #### 根据本地模型的loss、accuracy
            hidden_state = monitor.update(np.column_stack((metrics['loss'], metrics['acc'])))
            print(f"hidden_state: {hidden_state}")
            if hidden_state == 0:
                self.args.topk = 0.4
            if hidden_state > 0:
                self.args.topk = 0.1 * 4/(hidden_state+0.2)

            print(f"server train ===> topk: {self.args.topk}")

            ## CLIENT SELECTION 其实这里就是客户端选择的经过这个部分
            # print(f">>server process: 开始执行客户端选择")
            print(f">>server process: client selection has done")


            ## CHECK and SAVE current updates
            # self.weight_variance(local_models) # check variance of client weights
            self.save_current_updates(local_losses, accuracy, len(client_indices), phase='Train', round=round_idx)
            self.save_selected_clients(round_idx, client_indices)

            # del local_models, local_losses, accuracy

        for k in self.files:
            if self.files[k] is not None:
                self.files[k].close()


    ## train_epoch 为额外添加
    def train_clients_add(self, client_indices):
        """
        test one client
        ---
        Args
            client_idx: 选择到的客户端
        return
            results: loss, acc, auc 组合
        """
        local_losses, accuracy, local_metrics = [], [], []
        local_models = []
        local_model_hash = []

        ll, lh = np.inf, 0.
        # local training
        for client_idx in client_indices:  # 这里就是客户端训练的地方，循环去除客户端然后进行训练
            # result = self.local_training(client_idx)
            client = self.client_list[client_idx]
            result = client.train(deepcopy(self.model_train))

            local_losses.append(result['loss'])
            accuracy.append(result['acc'])
            local_metrics.append(result['metric'])
            local_models.append(result['paramlist'])
            local_model_hash.append((client_idx, result['client_hash']))

            if self.args.method in LOSS_THRESHOLD:
                if result['llow'] < ll: ll = result['llow'].item()
                lh += result['lhigh']

            progressBar(len(local_losses), len(client_indices), result) #客户端模型训练进度条

        print()
        return local_losses, accuracy, local_metrics, local_models, local_model_hash

    def test(self, aggregate_model, num_clients_for_test, phase='Test'):
        metrics = {'loss': [], 'acc': []}
        for client_idx in range(num_clients_for_test):
            client = self.client_list[client_idx]
            result = client.test(aggregate_model, False)
            # print(f">>>> test result:{result}")
            metrics['loss'].append(result['loss'])
            metrics['acc'].append(result['acc'])
            progressBar(len(metrics['acc']), num_clients_for_test, result, phase='Test')
        print()
        self.save_current_updates(metrics['loss'], metrics['acc'], num_clients_for_test, phase=phase)
        return metrics

    def save_current_updates(self, losses, accs, num_clients, phase='Train', round=None):
        """
        update current updated results for recording
        ---
        Args
            losses: losses
            accs: accuracies
            num_clients: number of clients
            phase: current phase (Train or TrainALL or Test)
            round: current round
        Return
            record "Round,TrainLoss,TrainAcc,TestLoss,TestAcc"
        """
        loss, acc = sum(losses) / num_clients, sum(accs) / num_clients
        if phase == 'Train':
            self.record = {}
            self.round = round
        self.record[f'{phase}/Loss'] = loss
        self.record[f'{phase}/Acc'] = acc
        status = num_clients if phase == 'Train' else 'ALL'
        print('>>server process: {} Clients {}ing: Loss {:.6f} Acc {:.4f}'.format(status, phase, loss, acc))
        if phase == 'Test':
            # wandb.log(self.record)
            if self.save_results:
                if self.test_on_training_data:
                    tmp = '{:.8f},{:.4f},'.format(self.record['TrainALL/Loss'], self.record['TrainALL/Acc'])
                else:
                    tmp = ''
                rec = '{},{:.8f},{:.4f},{}{:.8f},{:.4f}\n'.format(self.round,
                                                                  self.record['Train/Loss'], self.record['Train/Acc'], tmp,
                                                                  self.record['Test/Loss'], self.record['Test/Acc'])
                self.files['result'].write(rec)

    def save_selected_clients(self, round_idx, client_indices):
        """
        save selected clients' indices
        ---
        Args
            round_idx: current round
            client_indices: clients' indices to save
        """
        self.files['client'].write(f'{round_idx+1},')
        np.array(client_indices).astype(int).tofile(self.files['client'], sep=',')
        self.files['client'].write('\n')

    def weight_variance(self, local_models):
        """
        calculate the variances of model weights
        ---
        Args
            local_models: local clients' models
        """
        variance = 0
        for k in tqdm(local_models[0].state_dict().keys(), desc='>> compute weight variance'):
            tmp = []
            for local_model_param in local_models:
                tmp.extend(torch.flatten(local_model_param.cpu().state_dict()[k]).tolist())
            variance += torch.var(torch.tensor(tmp), dim=0)
        variance /= len(local_models)
        print('variance of model weights {:.8f}'.format(variance))

def aggregatie_weights(local_models, recv_list, weights_client, batch_num, args):
    global_param = [0] *  batch_num
    sum_weights = [0] * batch_num
    for idx,value in enumerate(local_models):#8个客户端的id，mask以及权重参数
        client_id = value[0]
        if recv_list[client_id] != 0:#如果不等于0，就说明这个没有参数训练，recv_list是记录客户端是否有被选中训练的
            continue
        # 1. 加载CKKS上下文和准备客户端权重，将客户端的聚合权重（一个浮点数，如 0.1）也编码成一个CKKS向量，以便进行同态乘法
        frac = weights_client[client_id]
        if  args.isSpars == 'topk':
            # 2. 解析客户端数据，对于topk模式，客户端发送了 [client_id, mask, param_list]
            mask = value[1]
            param = value[2]
            #print("id:",c_id,"mask:",mask)
            # 3. 遍历所有此id客户端下的批次，进行聚合
            for batch in range(batch_num):
                res = 0
                # 4. 检查当前客户端是否上传了此批次的数据
                if mask[batch]:
                    # 5. 计算在紧凑密文列表中的索引，这是非常关键的一步！客户端发来的 `param` 列表是压缩过的。
                    # 例如，如果 mask = [0, 1, 0, 1]，`param` 列表长度为2，包含 batch 1 和 batch 3 的密文。
                    # 当我们处理 batch=3 时，我们需要访问 `param` 列表的第 2 个元素（索引为1）。这个 `cnt` 就是在计算这个索引。
                    cnt = 0
                    for i in range(batch):
                        if mask[i]:
                            cnt += 1
                    # 6. 执行同态加权：Enc(model_i_batch) * weight_i a. 加载客户端上传的该批次的密文 b. 与该客户端的权重 `frac` 进行同态乘法
                    # 结果 `res` 是一个新的密文：Enc(model_i_batch * weight_i)
                    res = np.array(param[cnt]) * frac
                    # 7. 更新 sum_mask，`sum_mask` 记录了每个批次 所有贡献者的权重之和。 客户端解密后需要用这个值来求平均。
                    sum_weights[batch] += weights_client[client_id]
                    # 8. 执行同态累加 检查 `global_param` 中是否已经有这个批次的聚合结果（来自之前的客户端）
                    if isinstance(global_param[batch], np.ndarray) and global_param[batch].any():
                        # 如果有，加载它，并与当前客户端的加权结果 `res` 进行同态加法
                        # res = Enc(model_i_batch * w_i) + Enc(Σ_{j<i} model_j_batch * w_j)
                        # print(f"client{idx}, global_param 是否进入")
                        res = res + global_param[batch]
                    # 9. 存储更新后的聚合密文
                    # 将最终的聚合结果 `res` 序列化并存回 `global_param` 列表
                    global_param[batch] = res

    print(f'>>server process: client models aggregated, sum_mask: {sum_weights}')
    return sum_weights, global_param


def progressBar(idx, total, result, phase='Train', bar_length=20):
    """
    progress bar
    ---
    Args
        idx: current client index or number of trained clients till now
        total: total number of clients
        phase: Train or Test
        bar_length: length of progress bar
    """
    percent = float(idx) / total
    arrow = '=' * int(round(percent * bar_length) - 1) + '>'
    spaces = ' ' * (bar_length - len(arrow))

    sys.stdout.write("\r> Client {}ing: [{}] {}% ({}/{}) Loss {:.6f} Acc {:.4f}".format(
        phase, arrow + spaces, int(round(percent * 100)), idx, total, result['loss'], result['acc'])
    )
    sys.stdout.flush()