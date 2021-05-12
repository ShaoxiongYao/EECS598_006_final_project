import numpy as np
from rlkit.samplers.rollout import Rollout


def flatten_n(xs):
    xs = np.asarray(xs)
    return xs.reshape((xs.shape[0], -1))


def flatten_dict(dicts, keys):
    """
    Turns list of dicts into dict of np arrays
    """
    return {key: flatten_n([d[key] for d in dicts]) for key in keys}


def vec_multitask_rollout(
    env,
    agent,
    envs_rollout,
    obs_reset,
    max_path_length=np.inf,
    render=False,
    render_kwargs=None,
    observation_key=None,
    desired_goal_key=None,
    representation_goal_key=None,
    get_action_kwargs=None,
    return_dict_obs=False,
    reset_kwargs=None,
):
    if render_kwargs is None:
        render_kwargs = {}
    if get_action_kwargs is None:
        get_action_kwargs = {}
    n_envs = env.n_envs
    if envs_rollout is None:
        envs_rollout = [Rollout() for _ in range(n_envs)]
    o = obs_reset
    if o is None:
        o = env.reset()
    if render:
        env.render(**render_kwargs)
    d = np.zeros(n_envs, dtype=bool)
    rollouts_length = np.array([len(rollout) for rollout in envs_rollout])
    # print(f"rollouts length:{rollouts_length}")
    # print(f"max length:{max_path_length}")
    while rollouts_length.max() < max_path_length:
        np_o = flatten_dict(o, o[0].keys())
        np_s = np_o[observation_key]
        np_g = np_o[representation_goal_key]
        new_obs = np.hstack((np_s, np_g))
        np_a = agent.get_actions(new_obs, **get_action_kwargs)
        agent_info = [{} for _ in range(n_envs)]
        next_o, r, d, env_info = env.step(np_a)
        if render:
            env.render(**render_kwargs)
        for i in range(n_envs):
            envs_rollout[i].add_transition(
                o[i], np_a[i], next_o[i], r[i], d[i], env_info[i], agent_info[i]
            )
        if sum(d) > 0:
            # print(f"done: {d}")
            break
        o = next_o
        rollouts_length += 1
    paths = []
    for i, rollout in enumerate(envs_rollout):
        if d[i] or rollouts_length[i] >= max_path_length:
            o[i] = env.reset(i)
            paths.append(rollout.to_dict())
            envs_rollout[i] = Rollout()
    return paths, envs_rollout, o


def multitask_rollout(
    env,
    agent,
    max_path_length=np.inf,
    render=False,
    render_kwargs=None,
    observation_key=None,
    desired_goal_key=None,
    representation_goal_key=None,
    get_action_kwargs=None,
    return_dict_obs=False,
    reset_kwargs=None,
    is_reset=True
):
    if render_kwargs is None:
        render_kwargs = {}
    if get_action_kwargs is None:
        get_action_kwargs = {}
    dict_obs = []
    dict_next_obs = []
    observations = []
    actions = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = {}
    next_observations = []
    path_length = 0
    if is_reset:
        if reset_kwargs:
            o = env.reset(**reset_kwargs)
        else:
            o = env.reset()
    else:
        # check no reset
        # print("No reset")
        # input()
        o = env.env.env.observation()
        o = env.env.observation(o)
        o = env.observation(o)

    agent.reset()
    if render:
        env.render(**render_kwargs)
    desired_goal = o[desired_goal_key]
    step_time, policy_time = 0, 0
    while path_length < max_path_length:
        # print("before step")
        # input()
        dict_obs.append(o)
        if observation_key:
            s = o[observation_key]
        g = o[representation_goal_key]
        new_obs = np.hstack((s, g))

        # policy time
        start_time = time.time()
        a, agent_info = agent.get_action(new_obs, **get_action_kwargs)
        policy_time += time.time()-start_time

        # step time
        start_time = time.time()
        next_o, r, d, env_info = env.step(a)
        step_time += time.time()-start_time

        # print("environment step")
        # input()
        if render:
            env.render(**render_kwargs)
        # print("after render")
        # input()

        observations.append(o)
        rewards.append(r)
        terminals.append(d)
        actions.append(a)
        next_observations.append(next_o)
        dict_next_obs.append(next_o)
        agent_infos.append(agent_info)

        if not env_infos:
            for k, v in env_info.items():
                env_infos[k] = [v]
        else:
            for k, v in env_info.items():
                env_infos[k].append(v)
        path_length += 1
        if d:
            break
        o = next_o
    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    observations = np.array(observations)
    next_observations = np.array(next_observations)
    if return_dict_obs:
        observations = dict_obs
        next_observations = dict_next_obs
    for k, v in env_infos.items():
        env_infos[k] = np.array(v)
    # print("stop policy run")
    return dict(
        observations=observations,
        actions=actions,
        # rewards=np.array(rewards).reshape(-1, 1),
        rewards=np.array(rewards),
        next_observations=next_observations,
        terminals=np.array(terminals).reshape(-1, 1),
        agent_infos=agent_infos,
        env_infos=env_infos,
        desired_goals=np.repeat(desired_goal[None], path_length, 0),
        full_observations=dict_obs,
    )


def multiagent_multitask_rollout(
    env,
    agent,
    max_path_length=np.inf,
    render=False,
    render_kwargs=None,
    observation_key=None,
    achieved_q_key=None,
    desired_q_key=None,
    representation_goal_key=None,
    get_action_kwargs=None,
    reset_kwargs=None,
):
    if render_kwargs is None:
        render_kwargs = {}
    if get_action_kwargs is None:
        get_action_kwargs = {}
    observations = [[], []]
    actions = [[], []]
    rewards = [[], []]
    terminals = [[], []]
    agent_infos = [[], []]
    env_infos = [{}, {}]
    next_observations = [[], []]
    paths_length = 0
    if reset_kwargs:
        o = env.reset(**reset_kwargs)
    else:
        o = env.reset()
    agent.reset()
    if render:
        env.render(**render_kwargs)

    def step_agent(env, agent, o):
        if observation_key:
            s = o[observation_key]
        g = o[representation_goal_key]
        new_o = np.hstack((s, g))
        a, agent_info = agent.get_action(new_o, **get_action_kwargs)
        next_o, r, d, env_info = env.step(a)
        return a, r, d, next_o, agent_info, env_info

    def append_to_buffer(idx, o, a, r, d, agent_info, env_info):
        observations[idx].append(o)
        rewards[idx].append(r)
        terminals[idx].append(d)
        actions[idx].append(a)
        agent_infos[idx].append(agent_info)
        # observations[running_agent].append(next_o)
        if not env_infos[idx]:
            for k, v in env_info.items():
                env_infos[idx][k] = [v]
        else:
            for k, v in env_info.items():
                env_infos[idx][k].append(v)

    agents_q = [o[achieved_q_key], o[desired_q_key]]
    while paths_length < max_path_length:
        # agent0 turn
        env.set_state_goal(agents_q[0], agents_q[1])
        o = env.observe()
        if len(observations[0]) > 0:
            next_observations[0].append(o)
        a, r, d, next_o, agent_info, env_info = step_agent(env, agent, o)
        append_to_buffer(0, o, a, r, d, agent_info, env_info)
        agents_q[0] = next_o[achieved_q_key]
        if render:
            env.render(**render_kwargs)
        paths_length += 1
        if d or paths_length == max_path_length:
            # if the task is done then there is no agent1 action
            # before getting agent0 observation
            next_observations[0].append(next_o)
            # update agent 1 next_obs after agent 0 move
            env.set_state_goal(agents_q[1], agents_q[0])
            o = env.observe()
            if len(observations[1]) > 0:
                next_observations[1].append(o)
                rewards[1][-1] = r
                terminals[1][-1] = d
                for k, v in env_info.items():
                    env_infos[1][k][-1] = v
            break
        # agent1 turn
        # switch position and goal of the environment from agent1 perspective
        # and recompute observation
        env.set_state_goal(agents_q[1], agents_q[0])
        o = env.observe()
        if len(observations[1]) > 0:
            next_observations[1].append(o)
        a, r, d, next_o, agent_info, env_info = step_agent(env, agent, o)
        append_to_buffer(1, o, a, r, d, agent_info, env_info)
        agents_q[1] = next_o[achieved_q_key]
        if render:
            env.render(**render_kwargs)
        paths_length += 1
        if d or paths_length == max_path_length:
            # update agent 0 next_obs after agent 1 move
            next_observations[1].append(next_o)
            env.set_state_goal(agents_q[0], agents_q[1])
            o = env.observe()
            next_observations[0].append(o)
            rewards[0][-1] = r
            terminals[0][-1] = d
            for k, v in env_info.items():
                env_infos[0][k][-1] = v
            break
    paths = []
    for i in range(2):
        actions[i] = np.array(actions[i])
        if len(actions[i].shape) == 1:
            actions[i] = np.expand_dims(actions[i], 1)
        observations[i] = np.array(observations[i])
        next_observations[i] = np.array(next_observations[i])
        for k, v in env_infos[i].items():
            env_infos[i][k] = np.array(v)
        paths.append(
            dict(
                observations=observations[i],
                actions=actions[i],
                rewards=np.array(rewards[i]).reshape(-1, 1),
                next_observations=next_observations[i],
                terminals=np.array(terminals[i]).reshape(-1, 1),
                agent_infos=agent_infos[i],
                env_infos=env_infos[i],
            )
        )
    # plot_paths(paths)
    return paths


def plot_paths(paths):
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111, aspect="equal")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    for path, color in zip(paths, ["green", "red"]):
        print(path["terminals"][-1], path["rewards"][-1])
        path = [path["observations"][0]] + list(path["next_observations"])
        for i in range(len(path) - 1):
            p = path[i]["achieved_goal"]
            next_p = path[i + 1]["achieved_goal"]
            ax.scatter(p[0], p[1], c=color)
            ax.scatter(next_p[0], next_p[1], c=color)
            ax.plot([p[0], next_p[0]], [p[1], next_p[1]], c=color)
    plt.show()


def rollout(
    env,
    agent,
    max_path_length=np.inf,
    render=False,
    render_kwargs=None,
    reset_kwargs=None,
):
    """
    The following value for the following keys will be a 2D array, with the
    first dimension corresponding to the time dimension.
     - observations
     - actions
     - rewards
     - next_observations
     - terminals

    The next two elements will be lists of dictionaries, with the index into
    the list being the index into the time
     - agent_infos
     - env_infos
    """
    if render_kwargs is None:
        render_kwargs = {}
    observations = []
    actions = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = {}
    o = env.reset()
    agent.reset()
    next_o = None
    path_length = 0
    if reset_kwargs:
        o = env.reset(**reset_kwargs)
    else:
        o = env.reset()
    if render:
        env.render(**render_kwargs)
    while path_length < max_path_length:
        a, agent_info = agent.get_action(o)
        next_o, r, d, env_info = env.step(a)
        observations.append(o)
        rewards.append(r)
        terminals.append(d)
        actions.append(a)
        agent_infos.append(agent_info)
        if not env_infos:
            for k, v in env_info.items():
                env_infos[k] = [v]
        else:
            for k, v in env_info.items():
                env_infos[k].append(v)
        path_length += 1
        if d:
            break
        o = next_o
        if render:
            env.render(**render_kwargs)

    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    observations = np.array(observations)
    if len(observations.shape) == 1:
        observations = np.expand_dims(observations, 1)
        next_o = np.array([next_o])
    next_observations = np.vstack((observations[1:, :], np.expand_dims(next_o, 0)))
    for k, v in env_infos.items():
        env_infos[k] = np.array(v)
    return dict(
        observations=observations,
        actions=actions,
        rewards=np.array(rewards).reshape(-1, 1),
        next_observations=next_observations,
        terminals=np.array(terminals).reshape(-1, 1),
        agent_infos=agent_infos,
        env_infos=env_infos,
    )
