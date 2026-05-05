from copy import deepcopy

from Crypto.SelfTest.Hash.test_MD5 import test_data

from .trainer import Trainer
import numpy as np
import torch.nn.functional as F
import torch
from utils.model_compress import model_compress, model_dec_compress
from utils.util import params_tolist, params_tomodel
from utils.min_hash import minHash

class Client(object):
    def __init__(self, client_idx, local_train_data, local_test_data, model, random_r, args):
        """
        A client
        ---
        Args
            client_idx: index of the client
            nTrain: number of train dataset of the client
            local_train_data: train dataset of the client
            local_test_data: test dataset of the client
            model: given model for the client
            args: arguments for overall FL training
        """
        self.client_idx = client_idx
        self.test_data = local_test_data
        self.device = args.device
        self.trainer = Trainer(model, args)
        self.num_epoch = args.num_epoch  # E: number of local epoch
        self.loss_div_sqrt = args.loss_div_sqrt
        self.loss_sum = args.loss_sum
        self.random_r = random_r

        self.labeled_data = local_train_data  # train_data

        self.args = args

    def train(self, global_model):
        """
        train each client
        ---
        Args
            global_model: given current global model
        Return
            result = model, loss, acc
        """
        # SET MODEL
        self.trainer.set_model(global_model)

        # TRAIN
        if self.num_epoch == 0:  # no SGD updates
            result = self.trainer.train_E0(self.labeled_data)
        else:
            result = self.trainer.train(self.labeled_data)
            # result = self.trainer.train_add(self.labeled_loader)
        #result['model'] = self.trainer.get_model()

        # total loss / sqrt (# of local data)
        if self.loss_div_sqrt:  # total loss / sqrt (# of local data)
            result['metric'] *= np.sqrt(len(self.labeled_data))  # loss * n_k / np.sqrt(n_k)
        elif self.loss_sum:
            result['metric'] *= len(self.labeled_data)  # total loss

        # 模型序列化
        params_list, params_num, layer_shape = params_tolist(self.trainer.get_model())
        # 模型的hash，注意，这里的模型的list是一致的
        result['client_hash'] = minHash(self.random_r, params_list, params_list, self.args)

        total_sum = sum(params_num.values())
        # 模型压缩
        plain_list, res_mask = model_compress(params_list, self.args.enc_batch_size, self.args.topk,
                                              self.num_epoch, self.args.randk_seed, self.args.isSpars)
        result['paramlist'] = (self.client_idx, res_mask, plain_list)
        # print(f"==>client process: total_sum:{total_sum}, 压缩res_mask:{res_mask}, client_hash:{result['client_hash']}")

        return result

    def test(self, model, test_on_training_data=False):
        # TEST
        if test_on_training_data:
            # test on training dataset
            result = self.trainer.test(model, self.labeled_data)
        else:
            # test on test dataset
            result = self.trainer.test(model, self.test_data)
        return result

    def get_client_idx(self):
        return self.client_idx
