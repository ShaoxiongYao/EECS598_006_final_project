from collections import OrderedDict

from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.samplers.data_collector.path_collector import MdpPathCollector
from rlkit.samplers.rollout_functions import rollout


class HybridPathCollector(MdpPathCollector):
    """
    Collector which query an expert to solve the task each time the policy fails
    """
    def __init__(
            self,
            env,
            policy,
            expert_policy,
            max_num_epoch_paths_saved=None,
            render=False,
            render_kwargs=None,
    ):
        super().__init__(env, policy, max_num_epoch_paths_saved, render,
                         render_kwargs)
        self._expert_policy = expert_policy

    def collect_new_paths(self, max_path_length, num_steps,
                          discard_incomplete_paths):
        paths = []
        num_steps_collected = 0
        while num_steps_collected < num_steps:
            max_path_length_this_loop = min(  # Do not go over num_steps
                max_path_length,
                num_steps - num_steps_collected,
            )
            path = rollout(
                self._env,
                self._policy,
                max_path_length=max_path_length_this_loop,
            )
            path_len = len(path['actions'])
            if (path_len != max_path_length and not path['terminals'][-1]
                    and discard_incomplete_paths):
                break
            path_expert_len = 0
            # if the path did not reach the goal, add expert demonstration
            if not path['rewards'][-1] > 0:
                path_expert = rollout(self._env, self._expert_policy,
                                      max_path_length=max_path_length_this_loop)
                # if expert demonstration successfully reached goal, add it to buffer
                if path_expert['rewards'][-1] > 0:
                    paths.append(path_expert)
                    path_expert_len = len(path_expert['actions'])
                else:
                    print('No expert solution found.')
                    # import pickle as pkl
                    # import os
                    # filename_fails = '/sequoia/data1/rstrudel/code/nmp/fails.pkl'
                    # if os.path.exists(filename_fails):
                    #     with open(filename_fails, 'rb') as fpkl:
                    #         file_fails = pkl.load(fpkl)
                    # else:
                    #     file_fails = []
                    # file_fails.append((self._env.idx_env, self._env.start, self._env.goal))
                    # with open(filename_fails, 'wb') as fpkl:
                    #     pkl.dump(file_fails, fpkl)
            num_steps_collected += path_len + path_expert_len
            paths.append(path)
        self._num_paths_total += len(paths)
        self._num_steps_total += num_steps_collected
        self._epoch_paths.extend(paths)
        return paths

    def get_diagnostics(self):
        path_lens = [len(path['actions']) for path in self._epoch_paths]
        stats = OrderedDict([
            ('num steps total', self._num_steps_total),
            ('num paths total', self._num_paths_total),
        ])
        stats.update(create_stats_ordered_dict(
            "path length",
            path_lens,
            always_show_all_stats=True,
        ))
        paths_policy = [path for path in self._epoch_paths if 'expert' not in path['agent_infos'][0]]
        success = [path['rewards'][-1][0] > 0 for path in paths_policy]
        stats['SuccessRate'] = sum(success) / len(success)
        stats['Expert_Supervision'] = 1 - len(paths_policy) / len(self._epoch_paths)
        return stats
