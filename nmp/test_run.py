import click
import numpy as np
import gym
import time

import mpenv.envs

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
def main(env_name, exp_name, seed, horizon, episodes, cpu, stochastic):
    if not cpu:
        set_gpu_mode(True)
    set_seed(seed)
    env = gym.make(env_name)
    env.seed(seed)
    env.set_eval()
    log_dir = settings.log_dir()

    print("env RayTracingObserver type:", type(env.env.env))

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

    # start = np.array([-0.3957,   0.21246, -0.39556,  0.55368, -0.40724,  0.52797,  0.49884])
    # goal = np.array([ 0.48026,  0.22924,  0.11136, -0.51902,  0.81974,  0.09123,  0.22434])
    o = env.reset(start=None, goal=None)

    # Try to use solve RRT
    success, path, trees, iterations = env.env.env.solve_rrt(True, nmp_input=[env, policy, horizon, render])
    # success, path, trees, iterations = env.env.env.solve_rrt(True)

    print("success:", success)
    print("path:", path)
    print("path keys:", path.keys())
    print("iterations:", iterations)

    # return

    # TODO: 
    # - define start and goal in env - pass in to rlkit

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
    
    
    # path = rollout_fn()
    '''
    print("type of path:", type(path))
    print("keys of path:", path.keys())
    process_path(path)

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
