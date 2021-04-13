import abc
from collections import OrderedDict
from typing import Iterable

from rlkit.core.batch_rl_algorithm import BatchRLAlgorithm
from rlkit.core.online_rl_algorithm import OnlineRLAlgorithm
from rlkit.core.trainer import Trainer
from rlkit.torch.core import np_to_pytorch_batch
from torch import nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP


class TorchOnlineRLAlgorithm(OnlineRLAlgorithm):
    def to(self, device, distributed=False):
        for i, net in enumerate(self.trainer.networks):
            net.to(device)
            if distributed:
                self.trainer.networks[i] = DDP(
                    net, device_ids=[device], find_unused_parameters=True
                )

    def training_mode(self, mode):
        for net in self.trainer.networks:
            net.train(mode)


class TorchBatchRLAlgorithm(BatchRLAlgorithm):
    def to(self, device, distributed=False):
        networks = self.trainer.networks
        for i, net in enumerate(networks):
            net.to(device.index)
            if distributed:
                networks[i] = DDP(
                    net, device_ids=[device.index], find_unused_parameters=True
                )
        self.trainer.networks = networks

    def training_mode(self, mode):
        for net in self.trainer.networks:
            net.train(mode)


class TorchTrainer(Trainer, metaclass=abc.ABCMeta):
    def __init__(self):
        self._num_train_steps = 0

    def train(self, np_batch):
        self._num_train_steps += 1
        torch_batch = np_to_pytorch_batch(np_batch)
        self.train_from_torch(torch_batch)

    def eval(self, np_batch):
        self._num_train_steps += 1
        torch_batch = np_to_pytorch_batch(np_batch)
        self.eval_from_torch(torch_batch)

    def get_diagnostics(self):
        return OrderedDict([("num train calls", self._num_train_steps),])

    @abc.abstractmethod
    def train_from_torch(self, batch):
        pass

    @property
    @abc.abstractmethod
    def networks(self) -> Iterable[nn.Module]:
        pass
