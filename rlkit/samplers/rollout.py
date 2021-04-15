import numpy as np


class Rollout:
    def __init__(self):
        self.dict_obs = []
        self.dict_next_obs = []
        self.actions = []
        self.rewards = []
        self.terminals = []
        self.agent_infos = []
        self.env_infos = {}
        self.path_length = 0

    def __len__(self):
        return self.path_length

    def add_transition(self, obs, action, next_obs, reward, done, env_info, agent_info):
        self.dict_obs.append(obs)
        self.dict_next_obs.append(next_obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.terminals.append(done)
        if not self.env_infos:
            for k, v in env_info.items():
                self.env_infos[k] = [v]
        else:
            for k, v in env_info.items():
                self.env_infos[k].append(v)
        self.path_length += 1

    def to_dict(self):
        self.actions = np.array(self.actions)
        if len(self.actions.shape) == 1:
            self.actions = np.expand_dims(self.actions, 1)
        for k, v in self.env_infos.items():
            self.env_infos[k] = np.array(v)
        self.rewards = np.array(self.rewards)
        self.terminals = np.array(self.terminals).reshape(-1, 1)
        return dict(
            observations=self.dict_obs,
            actions=self.actions,
            rewards=self.rewards,
            next_observations=self.dict_next_obs,
            terminals=self.terminals,
            agent_infos=self.agent_infos,
            env_infos=self.env_infos,
        )
