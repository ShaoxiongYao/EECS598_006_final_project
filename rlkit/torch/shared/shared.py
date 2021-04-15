import torch

from rlkit.torch.torch_rl_algorithm import TorchTrainer


class SharedBaseTrainer(TorchTrainer):
    def __init__(self, base_trainer: TorchTrainer, base_model):
        super().__init__()
        self._base_trainer = base_trainer
        self._base_model = base_model

    def train_from_torch(self, data):
        obs = data["observations"]
        next_obs = data["next_observations"]
        data["observations"] = self._base_model(obs)
        data["next_observations"] = self._base_model(next_obs)
        self._base_trainer.train_from_torch(data)

    def get_diagnostics(self):
        return self._base_trainer.get_diagnostics()

    def end_epoch(self, epoch):
        self._base_trainer.end_epoch(epoch)

    @property
    def networks(self):
        return self._base_trainer.networks

    def get_snapshot(self):
        return self._base_trainer.get_snapshot()
