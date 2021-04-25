import click
import numpy as np
import gym
import time
import pickle

import mpenv.envs
from mpenv.core.model import ConfigurationWrapper

from nmp.policy import utils
from nmp.policy import RandomPolicy, StraightLinePolicy
from nmp import settings

from rlkit.torch.pytorch_util import set_gpu_mode
from rlkit.launchers.launcher_util import set_seed
from rlkit.samplers.rollout_functions import (
    multitask_rollout,
    rollout,
)

def path_len(path):
    # print("path type: ", type(path))
    # print("path element type: ", path.keys())
    # print("path element q type: ", type(path['points'][0]))
    n = len(path)
    length = 0
    # calculate path length
    for i in range(n):
        q0 = path['points'][i].q
        if i < n - 1:
            q1 = path['points'][i+1].q
            length += np.linalg.norm(q1 - q0)
    return length

@click.command()
@click.argument("env_name", type=str)
@click.option("-exp", "--exp-name", default="", type=str)
@click.option("-s", "--seed", default=None, type=int)
@click.option("-h", "--horizon", default=50, type=int, help="max steps allowed")
@click.option(
    "-e", "--episodes", default=0, type=int, help="number of episodes to evaluate"
)
@click.option("-cpu", "--cpu/--no-cpu", default=False, is_flag=True, help="use cpu")
@click.option(
    "-stoch",
    "--stochastic/--no-stochastic",
    default=False,
    is_flag=True,
    help="stochastic mode",
)
@click.option("-solver_type", "--solver_type", default=None, type=str, help="type of solver")
def main(env_name, exp_name, seed, horizon, episodes, cpu, stochastic, solver_type):
    print("-------- start running --------")
    begin_t = time.time()
    if not cpu:
        set_gpu_mode(True)
    set_seed(seed)
    env = gym.make(env_name)
    env.seed(seed)
    env.set_eval()
    log_dir = settings.log_dir()

    print("seed:", seed)
    print("horizon: ", horizon)
    print("cpu: ", cpu)
    if exp_name:
        policy = utils.load(log_dir, exp_name, cpu, stochastic)
        if stochastic:
            num_params = policy.num_params()
        else:
            num_params = policy.stochastic_policy.num_params()
        # print(f"num params: {num_params}")
    else:
        policy = RandomPolicy(env)

    render = episodes == 0

    reset_kwargs = {}

    o = env.reset(start=None, goal=None)
    if solver_type == "Normal_RRT":
        success, path, trees, iterations = env.env.env.solve_rrt(True)
        print("SOLVER: Normal RRT")
        print("success: ", success)
        print("length: ", path_len(path))
        # print("iterations:", iterations)

    elif solver_type == "RL_RRT":
        success, path, trees, iterations = env.env.env.solve_rrt(True, nmp_input=[env, policy, horizon, render], max_iterations=int(2000/horizon))
        print("SOLVER: RL_RRT")
        print("success: ", success)
        print("length: ", path_len(path))
        # print("iterations:", iterations)

    elif solver_type == "RL":

        def rollout_fn():
            return multitask_rollout(
                env,
                policy,
                horizon,
                # render,
                observation_key="observation",
                desired_goal_key="desired_goal",
                representation_goal_key="representation_goal",
                **reset_kwargs,
                is_reset=False
            )

        returns = []
        rewards = []
        n_steps = []
        lengths = []
        successes = []
        paths_states = []

        def process_path(path):
            obs = path["observations"]
            n = obs.shape[0]
            length = 0
            path_states = []
            # calculate path length
            for i in range(n):
                q0 = obs[i]["achieved_q"]
                if i < n - 1:
                    q1 = obs[i + 1]["achieved_q"]
                    length += np.linalg.norm(q1 - q0)
                path_states.append(q0[:2])
            # append a list of states
            paths_states.append(path_states)
            lengths.append(length)
            successes.append(path["env_infos"]["success"][-1])
            rewards.append(path["rewards"])
            returns.append(np.sum(path["rewards"]))
            n_steps.append(len(path["rewards"]))

        path = rollout_fn()
        process_path(path)
        print("SOLVER: RL")
        print("successes", successes)
        print("length: ", length)

    stop_t = time.time()
    print("time spent: ", stop_t - begin_t)
    print("-------- stop running --------")


if __name__ == "__main__":
    main()
