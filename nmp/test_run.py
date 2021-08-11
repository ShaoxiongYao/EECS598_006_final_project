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
    our_multiagent_rollout_final
)

def path_len(path):
    # print("path type: ", type(path))
    # print("path element type: ", path.keys())
    # print("path element q type: ", type(path['points'][0]))
    n = len(path['points'])
    length = 0
    # calculate path length
    for i in range(n):
        q0 = path['points'][i].q
        # print(q0)
        if i < n - 1:
            q1 = path['points'][i+1].q
            length += np.linalg.norm(q1 - q0)
    return length

@click.command()
@click.argument("env_name", type=str)
@click.option("-exp", "--exp-names", default=[], multiple=True, type=str)
@click.option("-s", "--seed", default=None, type=int)
@click.option("-h", "--horizon", default=50, type=int, help="max steps allowed")
@click.option("-cpu", "--cpu/--no-cpu", default=False, is_flag=True, help="use cpu")
@click.option("-r", "--render", default=False, is_flag=True, help="render using gepetto-gui")
@click.option("-v", "--verbose", default=False, is_flag=True, help="verbose output")
@click.option(
    "-stoch",
    "--stochastic/--no-stochastic",
    default=False,
    is_flag=True,
    help="stochastic mode",
)
@click.option("-solver_type", "--solver_type", default=None, type=str, help="type of solver")
@click.option(
    "-obstacles_type", "--obstacles_type", default="boxes", type=str, 
    help="type of obstacles, boxes, ycb, handcraft:set_3, or handcraft:hardest")
@click.option("-max_iters", "--max_iterations", default=50000, type=int, 
    help="maximum number of iterations in Normal_RRT or RL_RRT")
def main(env_name, exp_names, seed, horizon, cpu, 
         render, verbose, stochastic, solver_type, obstacles_type, max_iterations):
    print("-------- start running --------")
    if not cpu:
        set_gpu_mode(True)
    set_seed(seed)
    env = gym.make(env_name)
    env.seed(seed)
    env.set_eval()
    log_dir = settings.log_dir()

    print("seed:", seed)
    print("horizon:", horizon)
    print("cpu:", cpu)
    print("render:", render)
    policies = []
    if exp_names:
        for exp_name in exp_names:
            policies.append(utils.load(log_dir, exp_name, cpu, stochastic))
        # if stochastic:
        #     num_params = policy.num_params()
        # else:
        #     num_params = policy.stochastic_policy.num_params()
        # print(f"num params: {num_params}")
    else:
        policies.append(RandomPolicy(env))
    
    begin_t = time.time()
    # sampling obstacles parameters
    geoms_args = {
        "num_obstacles_range": [3, 10],
        "obstacles_type": obstacles_type 
    }

    # for fairness, max_iterations for RL_RRT is divided by horizon
    if solver_type == "RL_RRT":
        max_iterations //= horizon

    # RRT configurations
    solver_config = {
        "simplify": True,
        "max_iterations": max_iterations, 
        "render": render, 
        "verbose": verbose, 
        "sampler": "Full", 
        "expand_mode": "all"
    }

    # setup environment, do not use the first observation
    env.reset(start=None, goal=None, geoms_args=geoms_args)
    if solver_type == "Normal_RRT":
        success, path, trees, iterations, inference_time = env.env.env.solve_rrt(solver_config=solver_config)
        print("SOLVER: Normal RRT")
        print("success: ", success)
        if 'points' in path.keys():
            print("length: ", path_len(path))
        print("iterations:", iterations)

    elif solver_type == "RL_RRT":
        success, path, trees, iterations, inference_time = env.env.env.solve_rrt(nmp_input=[env, policies, horizon], 
                                                                 solver_config=solver_config)
        print("SOLVER: RL_RRT")
        print("success: ", success)
        if 'points' in path.keys():
            print("length: ", path_len(path))
        print("iterations:", iterations)

    elif solver_type == "RL":
        
        reset_kwargs = {}

        def rollout_fn():
            return our_multiagent_rollout_final(
                env,
                policies,
                horizon,  # max length in one step
                render,
                observation_key="observation",
                desired_goal_key="desired_goal",
                representation_goal_key="representation_goal",
                is_reset=False,
                **reset_kwargs,
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
        inference_time = 0
        print("SOLVER: RL")
        print("successes", successes)
        print("length: ", lengths[0])

    stop_t = time.time()
    print("time spent: ", stop_t - begin_t)
    print("inference time:", inference_time)
    print("parallel time:", stop_t - begin_t - inference_time*0.8)
    print("-------- stop running --------")


if __name__ == "__main__":
    main()
