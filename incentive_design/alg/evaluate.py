from __future__ import print_function

import os
import time
import numpy as np
import tensorflow as tf

from incentive_design.alg import ppo_agent

def test_escape_room(n_eval, env, sess, designer, list_agents, alg_name,
                     show_agent_spec=False):
    """Eval episodes.

    Args:
        n_eval: number of episodes to run
        env: env object
        sess: TF session
        designer: incentive designer object
        list_agents: list of agent objects
        alg_name: str
        show_agent_spec: if True, measures incentives in hardcoded cases.
    """
    rewards_total = np.zeros(env.n_agents)
    n_move_lever = np.zeros(env.n_agents)
    n_move_door = np.zeros(env.n_agents)
    incentives = np.zeros(env.n_agents)
    r_lever = np.zeros(env.n_agents)
    r_start = np.zeros(env.n_agents)
    r_door = np.zeros(env.n_agents)
    reward_id = 0
    lever_incentives = np.zeros(env.n_agents)
    optimal_incentives = np.zeros(env.n_agents)
    
    total_steps = 0
    epsilon = 0
    for idx_episode in range(1, n_eval + 1):

        list_obs = env.reset()
        done = False
        while not done:
            if show_agent_spec:
                # Every agent takes the same action
                list_actions = [0 for i in range(len(list_agents))]
                incentives_step = designer.compute_incentive(
                    list_obs[0], list_actions, sess)
                if designer.output_type == 'action':
                    incentives_step = np.take(incentives_step, list_actions)
                lever_incentives += incentives_step
                
                list_actions = [0 for i in range(len(list_agents) - 1)]
                list_actions.append(2)
                incentives_step = designer.compute_incentive(
                    list_obs[0], list_actions, sess)
                if designer.output_type == 'action':
                    incentives_step = np.take(incentives_step, list_actions)
                optimal_incentives += incentives_step

            list_actions = []
            for idx, agent in enumerate(list_agents):
                # +1 because idx=0 is the incentive designer's obs
                action = agent.run_actor(list_obs[idx+1], sess, epsilon)
                list_actions.append(action)
                if action == 0:
                    n_move_lever[idx] += 1
                elif action == 2:
                    n_move_door[idx] += 1
            
            if alg_name in ['m1', 'm2']:
                incentives_step = designer.compute_incentive(
                    list_obs[0], list_actions, sess)
                if designer.output_type == 'action':
                    incentives_step = np.take(incentives_step, list_actions)
            elif alg_name == 'dual_RL':
                if designer.action_space == 'continuous':
                    incentives_step, _ = designer.compute_incentive(
                        list_obs[0], list_actions, sess)
                else:
                    action_designer = designer.compute_incentive(
                        list_obs[0], list_actions, sess)
                    # Map action to vector of incentive values
                    incentives_step = env.map_discrete_action_to_incentives(
                        action_designer, list_actions)

            incentives += incentives_step

            for idx, agent in enumerate(list_agents):
                if list_actions[idx] == 0:
                    r_lever[idx] += incentives_step[idx]
                elif list_actions[idx] == 1:
                    r_start[idx] += incentives_step[idx]
                else:
                    r_door[idx] += incentives_step[idx]

            list_obs_next, env_rewards, done = env.step(list_actions)

            # Accumulate agents' env rewards
            rewards_total += env_rewards[1:]
            # add incentives to total reward
            rewards_total += incentives_step

            # Accumulate ID's reward
            reward_id += env_rewards[0]
            # subtract sum of incentives from ID's reward
            reward_id -= np.sum(incentives_step)

            list_obs = list_obs_next

        total_steps += env.steps

    rewards_total /= n_eval
    n_move_lever /= n_eval
    n_move_door /= n_eval
    incentives /= n_eval
    reward_id /= n_eval
    r_lever /= n_eval
    r_start /= n_eval
    r_door /= n_eval
    steps_per_episode = total_steps / n_eval
    lever_incentives /= n_eval
    optimal_incentives /= n_eval
    return (rewards_total, n_move_lever, n_move_door, incentives,
            reward_id, r_lever, r_start, r_door, steps_per_episode, lever_incentives, optimal_incentives)


# Map from name of map to the largest column position
# where a cleaning beam fired from that position can clear waste
cleanup_map_river_boundary = {'cleanup_small_sym': 2,
                              'cleanup_7x7_n3': 2,
                              'cleanup_10x10_sym': 3,
                              'cleanup_10x10_n5': 3,
                              'cleanup_15x15_sym': 5}

def test_ssd(n_eval, env, sess, designer, agents, alg='m1', log='',
             log_path='', render=False):

    """Runs test episodes for sequential social dilemma.

    Args:
        n_eval: number of eval episodes
        env: ssd env
        sess: TF session
        designer: id_ssd.MetaGrad1Step object
        agents: m1 or dual-RL ActorCritic object
        alg: if 'ac', then agents must be AC baseline agents with continuous reward-giving actions
        log: only used for testing a trained model
        log_path: path to log location
        render: only used for testing a trained model

    Returns:
        np.arrays of received incentives, env rewards,
        total rewards, waste cleared
    """
    reward_designer = np.zeros(n_eval)
    rewards_env = np.zeros((n_eval, env.n_agents))
    incentive_received = np.zeros((n_eval, env.n_agents))
    rewards_total = np.zeros((n_eval, env.n_agents))
    waste_cleared = np.zeros((n_eval, env.n_agents))
    received_riverside = np.zeros((n_eval, env.n_agents))
    received_beam = np.zeros((n_eval, env.n_agents))
    received_cleared = np.zeros((n_eval, env.n_agents))

    if log:
        list_agent_meas = []
        list_suffix = ['received', 'reward_env',
                       'reward_total', 'waste_cleared']
        for agent_id in range(1, env.n_agents + 1):
            for suffix in list_suffix:
                list_agent_meas.append('A%d_%s' % (agent_id, suffix))

        header = 'episode,'
        header += ','.join(list_agent_meas)
        header += ',r_designer\n'
        with open(os.path.join(log_path, 'test.csv'), 'w') as f:
            f.write(header)

    epsilon = 0
    for idx_episode in range(1, n_eval + 1):

        obs = env.reset()
        done = False
        if render:
            env.render()
            input('Episode %d. Press enter to start: ' % idx_episode)

        agent_actor_state = None
        designer_actor_state = None
        while not done:

            list_obs_t = []
            for idx in range(env.n_agents):
                list_obs_t.append(obs[str(idx)])
            if isinstance(agents, ppo_agent.Agent):
                (actions, agent_log_probs,
                 agent_actor_state) = agents.run_actor(
                     list_obs_t, sess, epsilon, agent_actor_state)
            else:
                actions = agents.run_actor(list_obs_t, sess, epsilon)
            list_binary_actions = []
            for action in actions:
                list_binary_actions.append(
                    1 if action == env.cleaning_action_idx else 0)

            # These are the positions seen by the incentive function
            list_agent_positions = env.env.agent_pos

            obs_next, env_rewards, done, info = env.step(np.squeeze(actions))
            if render:
                env.render()
                time.sleep(0.1)

            reward_designer[idx_episode-1] += env_rewards['p']
            env_reward_agents = [env_rewards[str(idx)] for idx in
                                 range(env.n_agents)]

            # Preprocess agents' actions to have values in {0,1,2},
            # 0: clean, 1: collected apples, 2: else
            action_type = []
            for idx in range(env.n_agents):
                if actions[idx] == env.cleaning_action_idx:
                    action_type.append(0)
                elif env_rewards[str(idx)] > 0:
                    action_type.append(1)
                else:
                    action_type.append(2)

            if alg == 'm1':
                incentives = designer.compute_incentive(
                    obs['p'], list_binary_actions, sess)
            elif alg == 'dual_RL':
                incentives, _ = designer.run_actor(
                obs['p'], list_binary_actions, sess)

            incentive_agents = np.take(incentives, action_type)
            incentive_received[idx_episode-1] += incentive_agents

            rewards_env[idx_episode-1] += env_reward_agents
            rewards_total[idx_episode-1] += env_reward_agents
            for idx in range(env.n_agents):
                # add incentives
                rewards_total[idx_episode-1, idx] += incentive_agents[idx]

            waste_cleared[idx_episode-1] += np.array(info['n_cleaned_each_agent'])

            for idx in range(env.n_agents):
                received = incentive_agents[idx]
                if (list_agent_positions[idx][1] <=
                    cleanup_map_river_boundary[env.config.map_name]):
                    received_riverside[idx_episode-1, idx] += received
                if list_binary_actions[idx] == 1:
                    received_beam[idx_episode-1, idx] += received
                if info['n_cleaned_each_agent'][idx] > 0:
                    received_cleared[idx_episode-1, idx] += received

            obs = obs_next

        if log:
            temp = idx_episode - 1
            combined = np.stack([incentive_received[temp],
                                 rewards_env[temp], rewards_total[temp],
                                 waste_cleared[temp]])
            s = '%d' % idx_episode
            for idx in range(env.n_agents):
                s += ','
                s += '{:.3e},{:.3e},{:.3e},{:.2f}'.format(
                    *combined[:, idx])
            s += ',%.3e\n' % reward_designer[temp]
            with open(os.path.join(log_path, 'test.csv'), 'a') as f:
                f.write(s)

    reward_designer = np.average(reward_designer)
    rewards_env = np.average(rewards_env, axis=0)
    incentive_received = np.average(incentive_received, axis=0)
    rewards_total = np.average(rewards_total, axis=0)
    waste_cleared = np.average(waste_cleared, axis=0)
    received_riverside = np.average(received_riverside, axis=0)
    received_beam = np.average(received_beam, axis=0)
    received_cleared = np.average(received_cleared, axis=0)

    if log:
        s = '\nAverage\n'
        combined = np.stack([incentive_received, rewards_env, rewards_total,
                             waste_cleared])
        for idx in range(env.n_agents):
            s += ','
            s += '{:.3e},{:.3e},{:.3e},{:.2f}'.format(
                *combined[:, idx])
        s += ',%.3e' % reward_designer
        with open(os.path.join(log_path, 'test.csv'), 'a') as f:
            f.write(s)

    return (incentive_received, rewards_env, rewards_total,
            waste_cleared, received_riverside, received_beam,
            received_cleared, reward_designer)


def test_foundation(n_eval, env, sess, designer, agents,
                    phase, alg_name, update_alg,
                    extra_meas=False, path='',
                    meas_tax_during_train=False,
                    meas_income_during_train=False,
                    idx_train_eps=0, model_eps=None):
    """
    Args:
        n_eval: number of episodes to run
        env: env object
        sess: TF session
        designer: incentive designer object
        agents: list of agent objects, or an agent object iff share parameter
        phase: 1 means free market, ID is forced to take no-op
        alg_name: string
        update_alg: str
        extra_meas: bool, set True to log measurements for trained policies
        path: str path to log directory, relative to here

    Returns np.array of average episode return for designer and agents
    """
    share_parameter = not isinstance(agents, list)
    epsilon = 0
    total_returns = np.zeros(env.n_agents+1)
    coin = np.zeros(env.n_agents)
    stone = np.zeros(env.n_agents)
    wood = np.zeros(env.n_agents)
    productivity = 0
    equality = 0
    component_tax = env.env._components_dict['PeriodicBracketTax']
    
    for idx_episode in range(1, n_eval+1):
        # print('episode', idx_episode)
        done = False
        tax_period = 0
        returns = np.zeros(env.n_agents+1)
        obs = env.reset()
        swf_0 = env.env.metrics['social_welfare/coin_eq_times_productivity']
        # id_reward = []
        # swf = []
        
        agent_actor_state = [None for i in range(env.n_agents)]
        if share_parameter:
            agent_actor_state = None
        designer_actor_state = None
        
        while not done:

            if share_parameter:
                list_obs_t = []
                list_obs_f = []
                list_action_mask = []
                for idx in range(env.n_agents):
                    o_i = obs[str(idx)]
                    list_obs_t.append(o_i['tensor'])
                    list_obs_f.append(o_i['flat'])
                    list_action_mask.append(o_i['action_mask'])
                if isinstance(agents, ppo_agent.Agent):
                    actions, agent_log_probs, agent_actor_state = agents.run_actor(list_obs_t, list_obs_f,
                                               list_action_mask, sess, epsilon, agent_actor_state)
                else:
                    actions = agents.run_actor(list_obs_t, list_obs_f,
                                               list_action_mask, sess, epsilon)
                action_dict = {str(idx): actions[idx] for idx
                               in range(env.n_agents)}
            else:
                action_dict = {}
                for idx, agent in enumerate(agents):
                    obs_i = obs[str(idx)]
                    if isinstance(agent, ppo_agent.Agent):
                        action, log_probs, actor_state = agent.run_actor(
                            obs_i['tensor'], obs_i['flat'], [obs_i['action_mask']],
                            sess, epsilon, agent_actor_state[idx])
                        agent_log_probs.append(log_probs)
                        agent_actor_state[idx] = actor_state
                    else:
                        action = agent.run_actor(
                            obs_i['tensor'], obs_i['flat'], [obs_i['action_mask']],
                            sess, epsilon)
                    action_dict[str(idx)] = action
            # print(action_dict, agent_log_probs, list_action_mask)
            obs_p = obs['p']
            # list of np.array
            action_masks = np.split(obs_p['action_mask'],
                                    env.p_action_spaces.cumsum()[:-1])
            if alg_name in ['us_federal', 'free_market']:
                action_dict['p'] = [0]
            else:
                if phase == 1:
                    # Tax is disabled in phase 1
                    action_dict['p'] = [0]
                elif phase == 2:
                    if isinstance(designer, ppo_agent.Agent):
                        action_dict['p'], designer_log_probs, designer_actor_state = designer.run_actor(
                            obs_p['tensor'], obs_p['flat'], action_masks,
                            sess, epsilon, designer_actor_state)
                        # print(action_dict['p'])
                    elif alg_name == 'dual_RL' and update_alg == 'ac':
                        action_dict['p'] = designer.run_actor(
                            obs_p['tensor'], obs_p['flat'], action_masks,
                            sess, epsilon)
                    elif alg_name in ['m1', 'm2']:
                        action_dict['p'] = [0]
                        # Apply these tax rates to the env
                        # component = env.env._components_dict['PeriodicBracketTax']
                        if component_tax.tax_cycle_pos == 1:
                            tax_period += 1
                            tax_rates, _, designer_actor_state = designer.compute_tax(
                                obs_p['tensor'], obs_p['flat'], sess, lstm_state=designer_actor_state)
                            # Apply these tax rates to the env
                            component_tax.set_tax_rates(list(tax_rates))
                            if extra_meas:
                                meas_1(env, path, idx_episode, tax_period,
                                       model_eps)
                            if meas_tax_during_train and idx_episode == 1:
                                meas_1(env, path, idx_train_eps, tax_period)
                            # if np.any(np.isnan(tax_rates)):
                            #     print('any nan in obs_p_tensor',
                            #           np.any(np.isnan(obs_p['tensor'])))
                            #     print('any nan in obs_p_flat',
                            #           np.any(np.isnan(obs_p['flat'])))
                            #     for var in tf.trainable_variables('designer/eta'):
                            #         print(var.name, np.any(
                            #             np.isnan(sess.run(var))))
                            #     print('printed eta')

                            

            # swf.append(env.env.metrics['social_welfare/coin_eq_times_productivity'])
            obs_next, rewards, done, info = env.step(action_dict)
            # id_reward.append(rewards['p'])

            if alg_name == 'dual_RL' and component_tax.tax_cycle_pos == 2:
                # dual_RL updates the tax rate within env.step
                # and tax_cycle_pos becomes 2 after the update
                tax_period += 1
                if extra_meas:
                    meas_1(env, path, idx_episode, tax_period)
                elif meas_tax_during_train and idx_episode == 1:
                    meas_1(env, path, idx_train_eps, tax_period)
            # if extra_meas and component.tax_cycle_pos == 1:
            #     # tax_cycle_pos == 1 immediately after env.step, means
            #     # enact_taxes() was called during that step in redistribution.py
            #     meas_2_each_period(env, path, idx_episode, tax_period)

            returns[0] += rewards['p']
            for idx in range(env.n_agents):
                returns[idx+1] += rewards[str(idx)]

            obs = obs_next
            
        # Account for the initial social welfare
        returns[0] += swf_0
        # if returns[0] < 0:
        #     for idx in range(len(id_reward)):
        #         print(id_reward[idx], swf[idx])
        #     input('negative')

        for idx in range(env.n_agents):
            agent = env.env.get_agent(str(idx))
            coin[idx] += agent.total_endowment('Coin')
            stone[idx] += agent.total_endowment('Stone')
            wood[idx] += agent.total_endowment('Wood')
            
        productivity += env.env.metrics['social/productivity']
        equality += env.env.metrics['social/equality']

        total_returns += returns

        if extra_meas:
            meas_2(env, path, idx_episode, model_eps)
            meas_3(env, path, idx_episode, model_eps)
        if meas_income_during_train and idx_episode==1:
            meas_3(env, path, idx_train_eps)

    returns = total_returns / n_eval
    productivity /= n_eval
    equality /= n_eval
    coin /= n_eval
    stone /= n_eval
    wood /= n_eval

    return returns, productivity, equality, coin, stone, wood


def meas_1(env, path, episode, tax_period, model_eps=None):
    """measurement 1: tax rate in each bracket"""
    component = env.env._components_dict['PeriodicBracketTax']
    curr_marginal_rates = component.curr_marginal_rates

    s = '%d,%d,' % (episode, tax_period)
    s += ','.join([str(rate) for rate in curr_marginal_rates])
    s += '\n'

    fname = 'tax_rate_%d.csv' % model_eps if model_eps else 'tax_rate.csv'
    with open(os.path.join(path, fname), 'a') as f:
        f.write(s)


def get_map_agent_idx_to_skill(env):
    """Get a map from agent idx to skill."""
    
    list_skills = []
    for idx in range(0, env.n_agents):
        agent = env.env.get_agent(str(idx))
        list_skills.append((agent.state['build_payment'], idx))
    list_skills.sort()
    # map_skill_idx_to_agent_idx[0] is the idx of the least skilled agent
    map_skill_idx_to_agent_idx = [t[1] for t in list_skills]

    return map_skill_idx_to_agent_idx


def meas_2(env, path, episode, model_eps=None):
    """measurement 2: income and tax paid before and after redistribution.

    Summed over the episode.
    """
    component = env.env._components_dict['PeriodicBracketTax']

    taxes = component.get_dense_log()
    
    map_skill_idx_to_agent_idx = get_map_agent_idx_to_skill(env)

    # list_arrays[0] is the record for the least skilled agent
    list_arrays = [np.zeros(4) for _ in range(env.n_agents)]

    for tax_dict in taxes:
        if tax_dict == []:
            continue
        # for idx in range(0, env.n_agents):
        for idx_skill in range(0, env.n_agents):
            idx_agent = map_skill_idx_to_agent_idx[idx_skill]
            agent_dict = tax_dict[str(idx_agent)]
            income_before = agent_dict['income']
            tax_before = agent_dict['tax_paid']
            income_after = income_before - tax_before + agent_dict['lump_sum']
            tax_after = income_before - income_after
            list_arrays[idx_skill] += np.array([income_before, tax_before,
                                                income_after, tax_after])

    s = '%d' % episode
    for array in list_arrays:
        s += ','
        s += ','.join(['%.2f' % val for val in array])
    s += '\n'

    fname = 'income_and_tax_%d.csv' % model_eps if model_eps else 'income_and_tax.csv'
    with open(os.path.join(path, fname), 'a') as f:
        f.write(s)


def meas_2_each_period(env, path, episode, tax_period):
    """measurement 2: income and tax paid before and after redistribution."""
    component = env.env._components_dict['PeriodicBracketTax']

    tax_dict = component.taxes[-1]

    list_tup = []
    for idx in range(0, env.n_agents):
        agent = env.env.get_agent(str(idx))
        skill = agent.state['build_payment']
        agent_dict = tax_dict[str(idx)]

        income_before = agent_dict['income']
        tax_before = agent_dict['tax_paid']
        income_after = income_before - tax_before + agent_dict['lump_sum']
        tax_after = income_before - income_after

        tup = (skill, [income_before, tax_before, income_after, tax_after])
        list_tup.append(tup)

    # Ordered from lowest to highest skill
    list_tup.sort()

    s = '%d,%d' % (episode, tax_period)
    for tup in list_tup:
        income_and_tax = tup[1]
        s += ','
        s += ','.join(['%.2f' % val for val in income_and_tax])
    s += '\n'

    with open(os.path.join(path, 'income_and_tax.csv'), 'a') as f:
        f.write(s)        


def meas_3(env, path, episode, model_eps=None):
    """measurement 3: resource collected, net income from building,
    net income from trading for each agent

    Summed over the episode.
    """
    # Each array is (resource collected, net income from building,
    # net income from trading)
    # list_arrays[0] is the record for the least skilled agent
    list_arrays = [np.zeros(3) for _ in range(env.n_agents)]

    map_skill_idx_to_agent_idx = get_map_agent_idx_to_skill(env)

    # Record wood and stone gathered at each step
    component_gather = env.env._components_dict['Gather']
    # list over time steps of list over gather events of
    # dict with fields 'agent', 'resource', 'n'
    gathers = component_gather.get_dense_log()
    for gather_step in gathers:
        if gather_step == []:
            continue
        for event in gather_step:
            agent_idx = event['agent']
            resource = event['resource']
            n = event['n']
            idx_skill = map_skill_idx_to_agent_idx.index(agent_idx)
            list_arrays[idx_skill][0] += n
            
    # Record net income from building
    component_build = env.env._components_dict['Build']
    # list over time steps of list over build events of
    # dict with fields 'builder', 'loc', 'income'
    builds = component_build.get_dense_log()
    for build_step in builds:
        if build_step == []:
            continue
        for event in build_step:
            agent_idx = event['builder']
            income = event['income']
            idx_skill = map_skill_idx_to_agent_idx.index(agent_idx)
            list_arrays[idx_skill][1] += income

    # Record net income from trading
    component_trade = env.env._components_dict['ContinuousDoubleAuction']
    metrics =  component_trade.get_metrics()
    for key, income in metrics.items():
        if not 'income' in key:
            continue
        if np.isnan(income):
            income = 0
        agent_idx = int(key.split('/')[0])
        idx_skill = map_skill_idx_to_agent_idx.index(agent_idx)
        if 'Sell' in key:
            list_arrays[idx_skill][2] += income
        elif 'Buy' in key:
            list_arrays[idx_skill][2] -= income

    s = '%d' % episode
    for array in list_arrays:
        s += ','
        s += ','.join(['%.2f' % val for val in array])
    s += '\n'

    fname = 'activity_%d.csv' % model_eps if model_eps else 'activity.csv'
    with open(os.path.join(path, fname), 'a') as f:
        f.write(s)
