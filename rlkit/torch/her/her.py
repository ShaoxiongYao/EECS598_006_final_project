import torch

from rlkit.torch.torch_rl_algorithm import TorchTrainer


class HERTrainer(TorchTrainer):
    def __init__(self, base_trainer: TorchTrainer):
        super().__init__()
        self._base_trainer = base_trainer

    def train_from_torch(self, data):
        obs = data["observations"]
        next_obs = data["next_observations"]
        rep_goals = data["representation_obs_goals"]
        rep_next_goals = data["representation_next_obs_goals"]
        data["observations"] = torch.cat((obs, rep_goals), dim=1)
        data["next_observations"] = torch.cat((next_obs, rep_next_goals), dim=1)
        self._base_trainer.train_from_torch(data)

    def get_diagnostics(self):
        return self._base_trainer.get_diagnostics()

    def end_epoch(self, epoch):
        self._base_trainer.end_epoch(epoch)

    @property
    def networks(self):
        return self._base_trainer.networks

    @networks.setter
    def networks(self, nets):
        self._base_trainer.networks = nets

    def get_snapshot(self):
        return self._base_trainer.get_snapshot()
