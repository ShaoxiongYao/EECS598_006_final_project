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
@click.option("-from_file", "--from_file", default=None, type=str, help="use cpu")
def main(env_name, exp_name, seed, horizon, episodes, cpu, stochastic, from_file):
    if not cpu:
        set_gpu_mode(True)
    set_seed(seed)
    env = gym.make(env_name)
    env.seed(seed)
    env.set_eval()
    log_dir = settings.log_dir()

    print("seed:", seed)
    if exp_name:
        policy = utils.load(log_dir, exp_name, cpu, stochastic)
        if stochastic:
            num_params = policy.num_params()
        else:
            num_params = policy.stochastic_policy.num_params()
        print(f"num params: {num_params}")
    else:
        policy = RandomPolicy(env)

    render = episodes == 0

    reset_kwargs = {}

    # start = np.array([ 0.3631,  -0.10471,  0.40929,  0.22057,  0.89626, -0.23803,  0.30234])
    # goal = np.array([-0.25956, -0.04894,  0.48368,  0.35592,  0.62283 , 0.45564 , 0.52707])
    num_trails = 0
    # iterations_list = []
    # o = env.reset(start=None, goal=None)
    for _ in range(num_trails):
        o = env.reset(start=None, goal=None)

        # Try to use solve RRT
        success, path, trees, iterations = env.env.env.solve_rrt(True, nmp_input=[env, policy, horizon, render], max_iterations=250)
        # success, path, trees, iterations = env.env.env.solve_rrt(True)

        print("success:", success)
        # print("path:", path)
        # print("path keys:", path.keys())
        print("iterations:", iterations)

        # iterations_list.append(iterations)
    
    # print("average iterations:", np.mean(iterations_list))


    def rollout_fn():
        return multitask_rollout(
            env,
            policy,
            horizon,
            render,
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
    
    # env.env.env.init_viz()
    # env.env.env.viz.display(ConfigurationWrapper(env.env.env.model_wrapper,goal))
    o = env.reset(start=None, goal=None)
    path = rollout_fn()
    process_path(path)
    print("successes", successes)
    '''
    print("type of path:", type(path))
    print("keys of path:", path.keys())

    print("returns:", returns)
    print("rewards:", rewards)
    print("n_steps:", n_steps)
    print("lengths:", lengths)
    print("successes:", successes)
    print("observation exmaple:", path["observations"][0])
    print("terminal:", path["terminals"])
    print("path_states:", paths_states)
    '''

if __name__ == "__main__":
    main()
