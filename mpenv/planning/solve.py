import numpy as np
from mpenv.planning import rrt_bidir
from mpenv.planning import utils
from mpenv.core.model import ConfigurationWrapper
from rlkit.samplers.rollout_functions import multitask_rollout

EPSILON = 1e-7


def solve(env, delta_growth, iterations, simplify, nmp_input=None):
    """
    env: mpenv.envs.boxes.Boxes
    collision_fn : maps x to True (free) / False (collision)
    sample_fn : return a configuration
    """
    algo = rrt_bidir.rrt_bidir
    model_wrapper = env.model_wrapper
    delta_collision_check = env.delta_collision_check
    action_range = env.robot_props["action_range"]

    def collision_fn(q):
        return not model_wrapper.collision(q)

    def sample_fn():
        return model_wrapper.random_configuration()

    def distance_fn(q0, q1):
        return model_wrapper.distance(q0, q1)

    def interpolate_fn(q0, q1, t):
        return model_wrapper.interpolate(q0, q1, t)

    def arange_fn(q0, q1, resolution):
        print("q0:", q0)
        print("q1:", q1)
        print("resolution:", resolution)
        return model_wrapper.arange(q0, q1, resolution)

    def expand_fn(q0, q1, limit_growth=True, nmp_input=nmp_input):
        """
        policy_env: mpenv.observers.robot_links.RobotLinksObserver
        """
        policy_env, policy, horizon, render = None, None, None, None
        if not nmp_input == None:
            policy_env, policy, horizon, render = nmp_input
            print("Use RL policy to extend")

        if limit_growth:
            dist = distance_fn(q0, q1)
            t1 = min(dist, delta_growth) / (dist + EPSILON)
            q1 = interpolate_fn(q0, q1, t1)

        reset_kwargs = {}
        def rollout_fn():
            return multitask_rollout(
                policy_env,
                policy,
                horizon, # max length in one step
                render,
                observation_key="observation",
                desired_goal_key="desired_goal",
                representation_goal_key="representation_goal",
                is_reset=False,
                **reset_kwargs,
            )

        if policy == None:
            print("Normal extension function")
            path = arange_fn(q0, q1, delta_collision_check)
            q_stop, collide = env.stopping_configuration(path)
            q_stop_list = []
            q_stop_list.append(q_stop)
            # q_stop: ConfigurationWrapper

            return q_stop_list, not collide.any()
        else:
            path = arange_fn(q0, q1, delta_collision_check)
            q_stop, collide = env.stopping_configuration(path)
            q_stop_list = []
            if collide.any():

                start, goal = q0, q1
                if not isinstance(start, ConfigurationWrapper):
                    start = ConfigurationWrapper(policy_env.env.env.model_wrapper, start)
                policy_env.env.env.state = start
                if not isinstance(goal, ConfigurationWrapper):
                    goal = ConfigurationWrapper(policy_env.env.env.model_wrapper, goal)
                policy_env.env.env.goal_state = goal


                policy_path = rollout_fn()
                end = policy_path["terminals"][-1][0]
                obs = policy_path["observations"]
                n = obs.shape[0]
                for i in range(n):
                    q = obs[i]["achieved_q"]
                    config = ConfigurationWrapper(model_wrapper, q)
                    q_stop_list.append(config)

                return q_stop_list, end
            else:
                q_stop_list.append(q_stop)
                return q_stop_list, not collide.any()

    '''
    def expand_fn(q0, q1, limit_growth=True):
        if limit_growth:
            dist = distance_fn(q0, q1)
            t1 = min(dist, delta_growth) / (dist + EPSILON)
            q1 = interpolate_fn(q0, q1, t1)
        path = arange_fn(q0, q1, delta_collision_check)
        q_stop, collide = env.stopping_configuration(path)
        return q_stop, not collide.any()
    '''

    def close_fn(qw0, qw1):
        return np.allclose(qw0.q, qw1.q)

    start = env.state
    goal = env.goal_state

    success, path, trees, iterations = algo(
        start, goal, sample_fn, expand_fn, distance_fn, close_fn, iterations=iterations
    )
    iterations_simplify = 0
    if success:
        if simplify:
            path["points"], iterations_simplify = utils.shorten(
                path["points"], expand_fn, interpolate_fn, distance_fn
            )
            path["points"] = utils.limit_step_size(
                # path["points"], arange_fn, action_range
                path["points"], arange_fn, delta_growth
            )
        else:
            path["points"] = np.array(path["points"])
        path["collisions"] = np.array(path["collisions"])
        path["start"] = path["points"][0]
        path["goal"] = path["points"][-1]
    return success, path, trees, iterations + iterations_simplify
