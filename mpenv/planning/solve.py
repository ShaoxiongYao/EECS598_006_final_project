import numpy as np
import time
from mpenv.planning import rrt_bidir
from mpenv.planning import utils
from mpenv.core.model import ConfigurationWrapper
from rlkit.samplers.rollout_functions import multitask_rollout

EPSILON = 1e-7


def solve(env, delta_growth, iterations, simplify, render=False, verbose=False, nmp_input=None, sampler="Full"):
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

    def sample_full_fn():
        x_rand = model_wrapper.random_configuration()
        
        if render:
            env.viz.display(x_rand)

        return x_rand

    def sample_free_fn():
        x_rand = model_wrapper.random_free_configuration()

        if render:
            env.viz.display(x_rand)

        return x_rand

    def sample_bridge_fn():
        return model_wrapper.random_bridge_configuration()

    def sample_near_surface_fn():
        return model_wrapper.random_near_surface()

    def distance_fn(q0, q1):
        return model_wrapper.distance(q0, q1)

    def interpolate_fn(q0, q1, t):
        return model_wrapper.interpolate(q0, q1, t)

    def arange_fn(q0, q1, resolution):
        # print("q0:", q0)
        # print("q1:", q1)
        # print("resolution:", resolution)
        return model_wrapper.arange(q0, q1, resolution)

    def expand_fn(q0, q1, limit_growth=False, render=render, nmp_input=nmp_input, 
                  rl_start = 'q_stop'):
        """
        policy_env: mpenv.observers.robot_links.RobotLinksObserver
        """
        start_t = time.time()
        policy_env, policy, horizon = None, None, None
        if not nmp_input == None:
            policy_env, policy, horizon = nmp_input
            # print("Use RL policy to extend")

        if limit_growth:
            dist = distance_fn(q0, q1)
            t1 = min(dist, delta_growth) / (dist + EPSILON)
            q1 = interpolate_fn(q0, q1, t1)

        reset_kwargs = {}

        def rollout_fn():
            return multitask_rollout(
                policy_env,
                policy,
                horizon,  # max length in one step
                render,
                observation_key="observation",
                desired_goal_key="desired_goal",
                representation_goal_key="representation_goal",
                is_reset=False,
                **reset_kwargs,
            )

        if policy == None:
            # print("Normal extension function")
            path = arange_fn(q0, q1, delta_collision_check)
            # q_stop: ConfigurationWrapper
            q_stop, collide, stop_idx = env.stopping_configuration(path)
            q_stop_list = []
            q_stop_list.append(q_stop)

            if render:
                previous_oMg = q0.q_oM[2]
                current_oMg = q_stop.q_oM[2]
                previous_ee = env.robot.get_ee(previous_oMg).translation
                current_ee = env.robot.get_ee(current_oMg).translation
                # path is the node name, which can be modified
                env.viz.add_edge_to_roadmap("path", previous_ee, current_ee)

            end_t = time.time()
            if verbose:
                print("expanding needs time:", end_t-start_t)
            return q_stop_list, not collide.any(), "straight_line"
        else:
            path = arange_fn(q0, q1, delta_collision_check)
            assert(not policy_env.env.env.model_wrapper.collision(path[0]))
            # collide means whether individual mesh on robot collides
            q_stop, collide, stop_idx = env.stopping_configuration(path)
            
            # debug straight line
            for idx in range(stop_idx+1):
                assert(not policy_env.env.env.model_wrapper.collision(path[idx]))
                # env.viz.display(path[idx])
                # time.sleep(0.1)
            assert(not policy_env.env.env.model_wrapper.collision(q_stop))
            
            q_stop_list = []
            if collide.any():

                # q0 = np.array([-0.3957, 0.21246, -0.39556, 0.55368, -0.40724, 0.52797, 0.49884])
                # q1 = np.array([-0.24471, 0.10095, -0.29217, -0.50942, -0.26528, -0.81633, 0.06103])

                if rl_start == 'q_stop':
                    q_stop_list.append(q_stop)

                    # visualization
                    if render:
                        env.render()
                        previous_oMg = q0.q_oM[2]
                        current_oMg = q_stop.q_oM[2]
                        previous_ee = env.robot.get_ee(previous_oMg).translation
                        current_ee = env.robot.get_ee(current_oMg).translation
                        # path is the node name, which can be modified
                        # normal RRT extend rendered in yellow

                        color_RRT = (0, 1, 1, 1)
                        env.viz.create_roadmap("path_RRT", color=color_RRT)
                        env.viz.add_edge_to_roadmap("path_RRT", previous_ee, current_ee)

                    start, goal = q_stop, q1
                elif rl_start == 'q0':
                    start, goal = q0, q1

                # set start and goal in the environment
                if not isinstance(start, ConfigurationWrapper):
                    start = ConfigurationWrapper(policy_env.env.env.model_wrapper, start)
                policy_env.env.env.state = start
                if not isinstance(goal, ConfigurationWrapper):
                    goal = ConfigurationWrapper(policy_env.env.env.model_wrapper, goal)
                policy_env.env.env.goal_state = goal

                policy_path = rollout_fn()
                end = policy_path["terminals"][-1][0]
                # print("end or nor: ", end)

                obs = policy_path["observations"]
                n = obs.shape[0]
                for i in range(n):
                    q = obs[i]["achieved_q"]
                    config = ConfigurationWrapper(model_wrapper, q)
                    q_stop_list.append(config)
                end_t = time.time()
                if verbose:
                    print("expanding needs time:", end_t-start_t)
                return q_stop_list, end, "rl_controller"
            else:
                q_stop_list.append(q_stop)

                # visualization
                if render:
                    env.render()
                    previous_oMg = q0.q_oM[2]
                    current_oMg = q_stop.q_oM[2]
                    previous_ee = env.robot.get_ee(previous_oMg).translation
                    current_ee = env.robot.get_ee(current_oMg).translation
                    # path is the node name, which can be modified
                    # normal RRT extend rendered in yellow

                    color_RRT = (0, 1, 1, 1)
                    env.viz.create_roadmap("path_RRT", color=color_RRT)
                    env.viz.add_edge_to_roadmap("path_RRT", previous_ee, current_ee)

                end_t = time.time()
                if verbose:
                    print("expanding needs:", end_t-start_t)
                return q_stop_list, not collide.any(), "straight_line"

    def expand_fn_short(q0, q1, limit_growth=False):
        if limit_growth:
            dist = distance_fn(q0, q1)
            t1 = min(dist, delta_growth) / (dist + EPSILON)
            q1 = interpolate_fn(q0, q1, t1)
        path = arange_fn(q0, q1, delta_collision_check)
        q_stop, collide, stop_idx = env.stopping_configuration(path)
        return q_stop, not collide.any()

    def close_fn(qw0, qw1):
        return np.allclose(qw0.q, qw1.q)

    start = env.state
    goal = env.goal_state

    if sampler == "Full":
        sample_fn = sample_full_fn
    elif sampler == "Free":
        sample_fn = sample_free_fn
    elif sampler == "Bridge":
        sample_fn = sample_bridge_fn
    elif sampler == "NearSurface":
        sample_fn = sample_near_surface_fn

    if render:
        env.render()

    if nmp_input is None:
        switch_tree_policy = 'compare_size'
    else:
        switch_tree_policy = 'per_iteration'

    success, path, trees, iterations = algo(
        start, goal, sample_fn, expand_fn, distance_fn, close_fn, iterations=iterations, 
        verbose=verbose, switch_tree_policy=switch_tree_policy # argument to decide when to switch tree
    )
    iterations_simplify = 0

    if success:
        if simplify:
            path["points"], iterations_simplify = utils.shorten(
                path["points"], expand_fn_short, interpolate_fn, distance_fn
            )
            path["points"] = utils.limit_step_size(
                # path["points"], arange_fn, action_range
                path["points"], arange_fn, delta_growth
            )
            if render:
                print("About to visualize the smoothed path\n")
                for idx, x in enumerate(path["points"]):
                    env.viz.display(x)
                    time.sleep(0.1)
        else:
            path["points"] = np.array(path["points"])
        path["collisions"] = np.array(path["collisions"])
        path["start"] = path["points"][0]
        path["goal"] = path["points"][-1]
    return success, path, trees, iterations
