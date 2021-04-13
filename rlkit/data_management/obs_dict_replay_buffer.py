from collections import OrderedDict

import numpy as np

from gym.spaces import Dict, Discrete
from rlkit.data_management.replay_buffer import ReplayBuffer


class ObsDictRelabelingBuffer(ReplayBuffer):
    """
    Replay buffer for environments whose observations are dictionaries, such as
        - OpenAI Gym GoalEnv environments. https://blog.openai.com/ingredients-for-robotics-research/
        - multiworld MultitaskEnv. https://github.com/vitchyr/multiworld/

    Implementation details:
     - Only add_path is implemented.
     - Image observations are presumed to start with the 'image_' prefix
     - Every sample from [0, self._size] will be valid.
     - Observation and next observation are saved separately. It's a memory
       inefficient to save the observations twice, but it makes the code
       *much* easier since you no longer have to worry about termination
       conditions.
    """

    def __init__(
        self,
        max_replay_buffer_size,
        env,
        fraction_goals_rollout_goals=1.0,
        fraction_goals_env_goals=0.0,
        internal_keys=None,
        goal_keys=None,
        observation_key="observation",
        desired_goal_key="desired_goal",
        achieved_goal_key="achieved_goal",
        representation_goal_key="representation_goal",
        env_infos_sizes=None,
    ):
        if internal_keys is None:
            internal_keys = []
        self.internal_keys = internal_keys
        if goal_keys is None:
            goal_keys = []
        if desired_goal_key not in goal_keys:
            goal_keys.append(desired_goal_key)
        self.goal_keys = goal_keys
        assert isinstance(env.observation_space, Dict)
        assert 0 <= fraction_goals_rollout_goals
        assert 0 <= fraction_goals_env_goals
        assert 0 <= fraction_goals_rollout_goals + fraction_goals_env_goals
        assert fraction_goals_rollout_goals + fraction_goals_env_goals <= 1
        self.max_replay_buffer_size = max_replay_buffer_size
        self.env = env
        self.fraction_goals_rollout_goals = fraction_goals_rollout_goals
        self.fraction_goals_env_goals = fraction_goals_env_goals
        self.ob_keys_to_save = [
            observation_key,
            desired_goal_key,
            achieved_goal_key,
            representation_goal_key,
        ]
        self.observation_key = observation_key
        self.desired_goal_key = desired_goal_key
        self.achieved_goal_key = achieved_goal_key
        self.representation_goal_key = representation_goal_key
        if isinstance(self.env.action_space, Discrete):
            self._action_dim = env.action_space.n
        else:
            self._action_dim = env.action_space.low.size

        self._actions = np.zeros(
            (max_replay_buffer_size, self._action_dim), dtype=np.float32
        )
        self._rewards = np.zeros((max_replay_buffer_size, 1), dtype=np.float32)
        # self._terminals[i] = a terminal was received at time i
        self._terminals = np.zeros((max_replay_buffer_size, 1), dtype=np.uint8)
        # self._obs[key][i] is the value of observation[key] at time i
        self._obs = {}
        self._next_obs = {}
        self.ob_spaces = self.env.observation_space.spaces
        for key in self.ob_keys_to_save + internal_keys:
            assert (
                key in self.ob_spaces
            ), "Key not found in the observation space: {}".format(key)
            type = np.float32
            if key.startswith("image"):
                type = np.uint8
            self._obs[key] = np.zeros(
                (max_replay_buffer_size, self.ob_spaces[key].low.size), dtype=type
            )
            self._next_obs[key] = np.zeros(
                (max_replay_buffer_size, self.ob_spaces[key].low.size), dtype=type
            )
        self.env_infos_sizes = env_infos_sizes
        if env_infos_sizes is None:
            if hasattr(env, "info_sizes"):
                env_infos_sizes = env.info_sizes
            else:
                env_infos_sizes = dict()
        self._env_infos = {}
        for key, size in env_infos_sizes.items():
            self._env_infos[key] = np.zeros((max_replay_buffer_size, size))
        self._env_infos_keys = env_infos_sizes.keys()

        self._top = 0
        self._size = 0

        # Let j be any index in self._idx_to_future_obs_idx[i]
        # Then self._next_obs[j] is a valid next observation for observation i
        self._idx_to_future_obs_idx = [None] * max_replay_buffer_size

    def add_sample(
        self, observation, action, reward, terminal, next_observation, **kwargs
    ):
        raise NotImplementedError("Only use add_path")

    def terminate_episode(self):
        pass

    def num_steps_can_sample(self):
        return self._size

    def add_path(self, path):
        obs = path["observations"]
        actions = path["actions"]
        rewards = path["rewards"]
        next_obs = path["next_observations"]
        terminals = path["terminals"]
        env_infos = path["env_infos"]
        path_len = len(rewards)

        actions = flatten_n(actions)
        if isinstance(self.env.action_space, Discrete):
            actions = np.eye(self._action_dim)[actions].reshape((-1, self._action_dim))
        obs = flatten_dict(obs, self.ob_keys_to_save + self.internal_keys)
        next_obs = flatten_dict(next_obs, self.ob_keys_to_save + self.internal_keys)
        obs = preprocess_obs_dict(obs)
        next_obs = preprocess_obs_dict(next_obs)

        if self._top + path_len >= self.max_replay_buffer_size:
            """
            All of this logic is to handle wrapping the pointer when the
            replay buffer gets full.
            """
            num_pre_wrap_steps = self.max_replay_buffer_size - self._top
            # numpy slice
            pre_wrap_buffer_slice = np.s_[self._top : self._top + num_pre_wrap_steps, :]
            pre_wrap_path_slice = np.s_[0:num_pre_wrap_steps, :]

            num_post_wrap_steps = path_len - num_pre_wrap_steps
            post_wrap_buffer_slice = slice(0, num_post_wrap_steps)
            post_wrap_path_slice = slice(num_pre_wrap_steps, path_len)
            for buffer_slice, path_slice in [
                (pre_wrap_buffer_slice, pre_wrap_path_slice),
                (post_wrap_buffer_slice, post_wrap_path_slice),
            ]:
                self._actions[buffer_slice] = actions[path_slice]
                self._rewards[buffer_slice] = rewards[path_slice]
                self._terminals[buffer_slice] = terminals[path_slice]
                for key in self.ob_keys_to_save + self.internal_keys:
                    self._obs[key][buffer_slice] = obs[key][path_slice]
                    self._next_obs[key][buffer_slice] = next_obs[key][path_slice]
                for key in self._env_infos_keys:
                    self._env_infos[key][buffer_slice] = env_infos[key][path_slice]
            # Pointers from before the wrap
            for i in range(self._top, self.max_replay_buffer_size):
                self._idx_to_future_obs_idx[i] = np.hstack(
                    (
                        # Pre-wrap indices
                        np.arange(i, self.max_replay_buffer_size),
                        # Post-wrap indices
                        np.arange(0, num_post_wrap_steps),
                    )
                )
            # Pointers after the wrap
            for i in range(0, num_post_wrap_steps):
                self._idx_to_future_obs_idx[i] = np.arange(i, num_post_wrap_steps,)
        else:
            slc = np.s_[self._top : self._top + path_len, :]
            self._actions[slc] = actions
            self._rewards[slc] = rewards
            self._terminals[slc] = terminals
            for key in self.ob_keys_to_save + self.internal_keys:
                if not np.allclose(self._obs[key].shape[1:], obs[key].shape[1:]):
                    raise ValueError(
                        f"Observation ({key}) shape problem: "
                        f"buffer {self._obs[key].shape[1:]}, path {obs[key].shape[1:]}"
                    )
                self._obs[key][slc] = obs[key]
                self._next_obs[key][slc] = next_obs[key]
            for key in self._env_infos_keys:
                self._env_infos[key][slc] = env_infos[key]
            for i in range(self._top, self._top + path_len):
                self._idx_to_future_obs_idx[i] = np.arange(i, self._top + path_len)
        self._top = (self._top + path_len) % self.max_replay_buffer_size
        self._size = min(self._size + path_len, self.max_replay_buffer_size)

    def _sample_indices(self, batch_size):
        return np.random.randint(0, self._size, batch_size)

    def random_batch(self, batch_size):
        indices = self._sample_indices(batch_size)
        resampled_goals = self._next_obs[self.desired_goal_key][indices]

        num_env_goals = int(batch_size * self.fraction_goals_env_goals)
        num_rollout_goals = int(batch_size * self.fraction_goals_rollout_goals)
        num_future_goals = batch_size - (num_env_goals + num_rollout_goals)
        new_obs_dict = self._batch_obs_dict(indices)
        new_actions = self._actions[indices]
        new_next_obs_dict = self._batch_next_obs_dict(indices)
        old_rewards = self._rewards[indices]
        env_infos = {}
        for k, v in self._env_infos.items():
            env_infos[k] = v[indices]

        if num_env_goals > 0:
            env_goals = self.env.sample_goals(num_env_goals)
            env_goals = preprocess_obs_dict(env_goals)
            last_env_goal_idx = num_rollout_goals + num_env_goals
            resampled_goals[num_rollout_goals:last_env_goal_idx] = env_goals[
                self.desired_goal_key
            ]
            for goal_key in self.goal_keys:
                new_obs_dict[goal_key][num_rollout_goals:last_env_goal_idx] = env_goals[
                    goal_key
                ]
                new_next_obs_dict[goal_key][
                    num_rollout_goals:last_env_goal_idx
                ] = env_goals[goal_key]
        # sampling future goals somehow takes time
        if num_future_goals > 0:
            future_obs_idxs = []
            for i in indices[-num_future_goals:]:
                possible_future_obs_idxs = self._idx_to_future_obs_idx[i]
                # This is generally faster than random.choice. Makes you wonder what
                # random.choice is doing
                num_options = len(possible_future_obs_idxs)
                next_obs_i = int(np.random.randint(0, num_options))
                future_obs_idxs.append(possible_future_obs_idxs[next_obs_i])
            future_obs_idxs = np.array(future_obs_idxs)
            # rg[-nfg:] = e_{t+1}[future_indices]
            resampled_goals[-num_future_goals:] = self._next_obs[
                self.achieved_goal_key
            ][future_obs_idxs]
            for goal_key in self.goal_keys:
                new_obs_dict[goal_key][-num_future_goals:] = self._next_obs[goal_key][
                    future_obs_idxs
                ]
                new_next_obs_dict[goal_key][-num_future_goals:] = self._next_obs[
                    goal_key
                ][future_obs_idxs]

        new_obs_dict[self.desired_goal_key] = resampled_goals
        new_next_obs_dict[self.desired_goal_key] = resampled_goals
        new_obs_dict = postprocess_obs_dict(new_obs_dict)
        new_next_obs_dict = postprocess_obs_dict(new_next_obs_dict)
        # resampled_goals must be postprocessed as well
        resampled_goals = new_next_obs_dict[self.desired_goal_key]

        """
        For example, the environments in this repo have batch-wise
        implementations of computing rewards:

        https://github.com/vitchyr/multiworld
        """

        new_rewards = old_rewards.copy()
        new_terminals = self._terminals[indices].copy()
        # if hasattr(self.env, "compute_rewards"):
        #     new_rewards = self.env.compute_rewards(new_actions, new_next_obs_dict,)
        if num_future_goals > 0:  # Assuming it's a (possibly wrapped) gym GoalEnv
            relabel_indices = slice(batch_size - num_future_goals, batch_size)
            for k, v in env_infos.items():
                env_infos[k] = v[relabel_indices]
            (
                new_rewards[relabel_indices],
                new_terminals[relabel_indices],
                _,
            ) = self.env.batch_compute_rewards(
                batch_next_achieved_goal=new_next_obs_dict[self.achieved_goal_key][
                    relabel_indices
                ],
                batch_goal=new_next_obs_dict[self.desired_goal_key][relabel_indices],
                batch_action=new_actions[relabel_indices],
                her_previous_reward=old_rewards[relabel_indices],
                **env_infos,
            )

        rep_obs_goals = self.env.represent_goal(
            new_obs_dict[self.achieved_goal_key], resampled_goals
        )
        rep_next_obs_goals = self.env.represent_goal(
            new_next_obs_dict[self.achieved_goal_key], resampled_goals
        )
        new_rewards = new_rewards.reshape(-1, 1)
        new_obs = new_obs_dict[self.observation_key]
        new_next_obs = new_next_obs_dict[self.observation_key]
        batch = {
            "observations": new_obs,
            "actions": new_actions,
            "rewards": new_rewards,
            "terminals": new_terminals,
            "next_observations": new_next_obs,
            "representation_obs_goals": rep_obs_goals,
            "representation_next_obs_goals": rep_next_obs_goals,
            "indices": np.array(indices).reshape(-1, 1),
        }
        return batch

    def _batch_obs_dict(self, indices):
        return {key: self._obs[key][indices] for key in self.ob_keys_to_save}

    def _batch_next_obs_dict(self, indices):
        return {key: self._next_obs[key][indices] for key in self.ob_keys_to_save}

    def get_diagnostics(self):
        buffer_infos = OrderedDict()
        buffer_infos["Size"] = self._size
        obs = self._obs["observation"][: self._size]
        obs_rep_goal = self._obs["representation_goal"][: self._size]
        buffer_infos["MeanObs"] = obs.mean()
        buffer_infos["StdObs"] = obs.std()
        buffer_infos["MeanGoal"] = obs_rep_goal.mean()
        buffer_infos["StdGoal"] = obs_rep_goal.std()
        # for k, v in buffer_infos.items():
        #     print("{}: {}".format(k, v))
        return buffer_infos


def flatten_n(xs):
    xs = np.asarray(xs)
    return xs.reshape((xs.shape[0], -1))


def flatten_dict(dicts, keys):
    """
    Turns list of dicts into dict of np arrays
    """
    # return {key: flatten_torch([d[key] for d in dicts]) for key in keys}
    return {key: flatten_n([d[key] for d in dicts]) for key in keys}


# def dict_to_device(d):
#     for k, v in d.items():
#         d[k] = v.to(ptu.device)
#     return d


# def flatten_torch(xs):
#     return torch.from_numpy(flatten_n(xs))


# def from_numpy_dict(x):
#     for k, v in x.items():
#         x[k] = ptu.from_numpy(v)
#     return x


def preprocess_obs_dict(obs_dict):
    """
    Apply internal replay buffer representation changes: save images as bytes
    """
    for obs_key, obs in obs_dict.items():
        if "image" in obs_key and obs is not None:
            obs_dict[obs_key] = unnormalize_image(obs)
    return obs_dict


def postprocess_obs_dict(obs_dict):
    """
    Undo internal replay buffer representation changes: save images as bytes
    """
    for obs_key, obs in obs_dict.items():
        if "image" in obs_key and obs is not None:
            obs_dict[obs_key] = normalize_image(obs)
    return obs_dict


def normalize_image(image):
    assert image.dtype == np.uint8
    return np.float64(image) / 255.0


def unnormalize_image(image):
    assert image.dtype != np.uint8
    return np.uint8(image * 255.0)
