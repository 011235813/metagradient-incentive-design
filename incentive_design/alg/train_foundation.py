"""Trains tax planner and agents on Foundation."""

from __future__ import division
from __future__ import print_function

import argparse
import json
import os
import random
import sys
import time
import csv

import numpy as np
import tensorflow as tf

from incentive_design.alg import evaluate
from incentive_design.alg import actor_critic
from incentive_design.alg import actor_critic_ps
from incentive_design.alg import ppo_agent
from incentive_design.env import foundation
from incentive_design.utils import buffers


def train_function(config):
    seed = config.main.seed
    np.random.seed(seed)
    random.seed(seed)
    tf.set_random_seed(seed)

    dir_name = config.main.dir_name
    exp_name = config.main.exp_name
    log_path = os.path.join('..', 'results', exp_name, dir_name)
    model_name = config.main.model_name
    save_period = config.main.save_period

    # Keep a record of parameters used for this run
    if not config.alg.resume_phase2:
        os.makedirs(log_path, exist_ok=True)

        with open(os.path.join(log_path, 'config.json'), 'w') as f:
            json.dump(config, f, indent=4, sort_keys=True)

    phase = config.alg.phase

    if phase == 1:  # force free market for phase 1
        for idx, tup in enumerate(config.env.components):
            if tup[0] == 'PeriodicBracketTax':
                tup[1]['disable_taxes'] = True
    elif phase == 2 and ((config.alg.name in ['us_federal', 'dual_RL'] and config.designer.cap_tax_rate) or
                         (config.alg.name in ['m1', 'm2'] and config.designer_m.cap_tax_rate)):
        for idx, tup in enumerate(config.env.components):
            if tup[0] == 'PeriodicBracketTax':
                slope = (1.0 - config.alg.initial_max_tax_rate)/ (
                    config.alg.n_episodes_reach_max_rate)
                if config.alg.resume_phase2:
                    initial_max_tax_rate = min(
                        1.0, config.alg.initial_max_tax_rate + config.alg.resume_ep*slope)
                else:
                    initial_max_tax_rate = config.alg.initial_max_tax_rate
                # Negative so that tax begins at initial_max_tax_rate,
                # see foundation/components/utils.py: annealed_tax_limit
                warmup_episodes = -initial_max_tax_rate / slope
                tup[1]['tax_annealing_schedule'] = [warmup_episodes, slope]

    n_episodes = int(config.alg.n_episodes)
    n_eval = config.alg.n_eval
    period = config.alg.period
    
    epsilon_step = ((config.agent.epsilon_start - config.agent.epsilon_end) /
                    config.agent.epsilon_div)
    if not config.alg.resume_phase2:
        epsilon = config.agent.epsilon_start
    else:
        if config.alg.name in ['m1', 'm2']:
            # in m1/m2, epsilon decreases once every 2 episodes
            epsilon_decrease = (config.alg.resume_ep//2) * epsilon_step
        else:
            epsilon_decrease = config.alg.resume_ep * epsilon_step
        epsilon = max(config.agent.epsilon_end,
                      (config.agent.epsilon_start - epsilon_decrease))
            
    env = foundation.Env(config.env)
    if phase == 2 and config.alg.name in ['m1', 'm2']:
        from incentive_design.alg import redistribution
        tax = redistribution.PeriodicBracketTax(
            env.n_agents, config.designer_m.cap_tax_rate,
            rate_max=1.0, rate_min=0.0, usd_scaling=1000.0)
        utility = redistribution.Utility()
    
    update_alg = 'ac'
    if "update_alg" in config.alg:
        update_alg = config.alg.update_alg
    
    # Initialize tax designer
    if phase == 1 or config.alg.name in ['us_federal', 'free_market']:
        designer = None
    elif phase == 2 and config.alg.name =='dual_RL':
        # ID is an RL agent in dual_RL
        if update_alg == 'ppo':
            designer = ppo_agent.Agent(
                'designer', config.designer, env.p_action_spaces,
                env.obs_dim_p_flat, env.obs_dim_p_t, 1, config.alg.objective,
                config.nn_designer)
        elif update_alg == 'ac':
            designer = actor_critic.ActorCritic(
                'designer', config.designer, env.p_action_spaces,
                env.obs_dim_p_flat, env.obs_dim_p_t, config.nn_designer)
        else:
            raise ValueError('%s is not an option' % update_alg)
    elif phase == 2 and config.alg.name in ['m1', 'm2']:
        # AC and PPO versions of m1 and m2 both use id_foundation
        if config.designer_m.use_ppo:
            from incentive_design.alg.id_foundation_ppo import MetaGrad1Step
        else:
            from incentive_design.alg.id_foundation import MetaGrad1Step
        designer = MetaGrad1Step(
            0, 'designer', config.designer_m,
            env.periodic_bracket_tax.n_brackets,
            env.obs_dim_p_flat, env.obs_dim_p_t, env.n_agents,
            config.nn_designer)
    else:
        raise ValueError('%s is not an option' % config.alg.name)
    
    # Initialize agents
    if update_alg == 'ppo':
        if config.alg.name in ['m1', 'm2']:
            if config.alg.name == 'm1':
                from incentive_design.alg import ppo_agent_m1 as ppom
            else:
                from incentive_design.alg import ppo_agent_m2 as ppom
            agents = ppom.Agent(
                    1, 'agent', config.agent, env.agent_action_space,
                    env.obs_dim_agent_flat, env.obs_dim_agent_t,
                    env.n_agents, config.alg.objective, config.nn_agent, tax, utility)
        elif config.agent.share_parameter:
            agents = ppo_agent.Agent(
                    'agent', config.agent, env.agent_action_space,
                    env.obs_dim_agent_flat, env.obs_dim_agent_t,
                    env.n_agents, config.alg.objective, config.nn_agent)
        else:
            agents = []
            for agent_id in range(0, env.n_agents):
                agents.append(ppo_agent.Agent(
                    'agent_%d'%agent_id, config.agent, env.agent_action_space,
                    env.obs_dim_agent_flat, env.obs_dim_agent_t,
                    1, config.alg.objective, config.nn_agent))
    else:
        if config.agent.share_parameter:
            if config.alg.name in ['m1', 'm2']:
                if config.alg.name == 'm1':
                    from incentive_design.alg import actor_critic_ps_m1 as acm
                else:
                    from incentive_design.alg import actor_critic_ps_m2 as acm
                agents = acm.ActorCritic(
                    1, 'agent', config.agent, env.agent_action_space[0],
                    env.obs_dim_agent_flat, env.obs_dim_agent_t,
                    env.n_agents, config.nn_agent, tax, utility)
            else:
                agents = actor_critic_ps.ActorCritic(
                    'agent', config.agent, env.agent_action_space[0],
                    env.obs_dim_agent_flat, env.obs_dim_agent_t,
                    env.n_agents, config.nn_agent)
        else:
            agents = []
            for agent_id in range(0, env.n_agents):
                agents.append(
                    actor_critic.ActorCritic(
                        'agent_%d'%agent_id, config.agent,
                        env.agent_action_space, env.obs_dim_agent_flat,
                        env.obs_dim_agent_t, config.nn_agent))

    # Initialization for m1 or m2
    if phase == 2 and config.alg.name in ['m1', 'm2']:
        designer.receive_agents(agents)
        agents.receive_designer(designer)
        agents.create_policy_gradient_op()
        agents.create_update_op()
        if not config.alg.restore_do_not_train_designer:
            if config.designer_m.use_critic:
                designer.create_critic_train_op()
            designer.create_tax_train_op()

    config_proto = tf.ConfigProto()
    if config.main.use_gpu:
        config_proto.device_count['GPU'] = 1
        config_proto.gpu_options.allow_growth = True
    else:
        config_proto.device_count['GPU'] = 0
    sess = tf.Session(config=config_proto)
    sess.run(tf.global_variables_initializer())
    if phase == 1 or config.alg.train_phase2_without_restoring or config.alg.resume_phase2:
        saver = tf.train.Saver(max_to_keep=config.main.max_to_keep)
    elif config.alg.restore_do_not_train_designer:
        # Only restore designer, not agents.
        to_restore = []
        for v in tf.trainable_variables():
            if 'agent' not in v.name.split('/'):
                to_restore.append(v)
        saver = tf.train.Saver(var_list=to_restore,
                               max_to_keep=config.main.max_to_keep)
    else:
        # Only restore agents, since the designer did not exist in phase 1
        to_restore = []
        for v in tf.trainable_variables():
            list_split = v.name.split('/')
            if ('designer' not in list_split) and (
                    'policy_prime' not in list_split):
                to_restore.append(v)
        saver = tf.train.Saver(var_list=to_restore,
                               max_to_keep=config.main.max_to_keep)

    if (phase == 1 or config.alg.train_phase2_without_restoring or
        config.alg.restore_do_not_train_designer):
        if config.agent.share_parameter:
            sess.run(agents.list_initialize_v_ops)
        else:
            for a in agents:
                sess.run(a.list_initialize_v_ops)

    if phase == 2:
        if not config.alg.train_phase2_without_restoring:
            print('Restoring variables from %s' % config.main.dir_restore)
            p = os.path.join('..', 'results', exp_name,
                             config.main.dir_restore,
                             config.main.model_restore)
            if config.alg.resume_phase2:
                p_prefix = os.path.join('..', 'results', exp_name,
                                        config.main.dir_restore)
                p = os.path.join(p_prefix, config.main.model_restore + '.'
                                 + str(config.alg.resume_ep))
                if not os.path.isfile(p + '.index'):
                    raise FileNotFoundError('Could not find %s' % p)
                    # did not find a model with episode suffix, then can assume
                    # that a final model.ckpt was stored
                    # p = os.path.join(p_prefix, config.main.model_restore)
            saver.restore(sess, p)
            # After restoring only the agents, save everything (now including
            # the ID if it exists)
            saver = tf.train.Saver(max_to_keep=config.main.max_to_keep)

        if config.alg.name in ['m1', 'm2'] and config.designer_m.use_critic and not config.alg.resume_phase2:
            sess.run(designer.list_initialize_v_ops)

    if phase == 2 and config.alg.name == 'm1':
        sess.run(agents.list_copy_main_to_prime_ops)

    if not config.alg.resume_phase2:
        header = 'episode,step_train,step,time,'
        header += ','.join(['return_%d'%idx for idx in range(0, env.n_agents)]+
                        ['return_p'])
        for s in ['coin', 'stone', 'wood']:
            header += ','
            header += ','.join(['%s_%d'%(s,idx) for idx in range(0, env.n_agents)])
        header += ',productivity,equality'
        header += '\n'
        with open(os.path.join(log_path, 'log.csv'), 'w') as f:
            f.write(header)
    else:
        log_path = os.path.join('..', 'results', exp_name,
                             config.main.dir_restore)
        cleaned_log = []
        with open(os.path.join(log_path, 'log.csv'), 'r') as f:
            reader = csv.reader(f, delimiter=',')
            cleaned_log.append(next(reader))
            for r in reader:
                if int(r[0]) <= config.alg.resume_ep:
                    cleaned_log.append(r)
                    t_start = time.time() - float(r[3])
        with open(os.path.join(log_path, 'log.csv'), 'w') as f:
            writer = csv.writer(f)
            writer.writerows(cleaned_log)   
    
    if config.main.meas_tax_during_train:
        n_brackets = env.env._components_dict['PeriodicBracketTax'].n_brackets
        if n_brackets != 7:
            raise ValueError('Expected n_brackets = 7 but found %d' %
                             n_brackets)
        header = 'episode,tax_period,'
        header += ','.join(['b%d' % idx for idx in range(1, n_brackets+1)])
        header += '\n'
        with open(os.path.join(log_path, 'tax_rate.csv'), 'w') as f:
            f.write(header)
    if config.main.meas_income_during_train:
        measurements = ['resource', 'income_build', 'income_trade']
        header = 'episode'
        for idx in range(1, env.n_agents+1):
            header += ','
            header += ','.join([s + '_%d'%idx for s in measurements])
        header += '\n'
        with open(os.path.join(log_path, 'activity.csv'), 'w') as f:
            f.write(header)

    if not config.alg.resume_phase2:
        idx_episode = 0
        step = 0
        step_train = 0
        t_start = time.time()
    else:
        idx_episode = config.alg.resume_ep
        step = idx_episode * config.env.episode_length
        step_train = idx_episode
    list_buffers = None

    while idx_episode < n_episodes:

        # print('Episode', idx_episode)
        if (config.alg.name != 'm1' or config.alg.restore_do_not_train_designer
            or (list_buffers is None or not config.designer_m.pipeline)):
            # For m1 pipeline=T, this is run only for the first episode to start pipeline.
            # Afterwards, the \hat{traj} will exist
            # list_buffers[0] is the designer's trajectory,
            # list_buffers[1:] are the agents' trajectories
            list_buffers = run_episode(sess, env, designer, agents,
                                       epsilon, phase, config.alg.name,
                                       update_alg, idx_episode, False,
                                       prime=False)
            step += len(list_buffers[1].reward)
            idx_episode += 1
        # Update the prime parameters
        if config.alg.name == 'm1':
            agents.update(sess, list_buffers[1], epsilon)

        if not config.alg.restore_do_not_train_designer:
            # Collect validation trajectory for method m1 or m2
            if phase == 2 and config.alg.name in ['m1', 'm2']:
                val_traj = (config.alg.name == 'm2') or (
                    config.alg.name == 'm1' and
                    not config.designer_m.pipeline)
                list_buffers_new = run_episode(
                    sess, env, designer, agents, epsilon, phase,
                    config.alg.name, update_alg, idx_episode, val_traj,
                    prime=config.alg.name=='m1')
                step += len(list_buffers_new[1].reward)
                idx_episode += 1
            # Train designer
            if phase == 2 and config.alg.name in ['dual_RL']:
                designer.train(sess, list_buffers[0], epsilon)
            elif phase == 2 and config.alg.name in ['m1', 'm2']:
                if config.designer_m.use_critic:
                    designer.train_critic(sess, list_buffers[0])
                designer.train(sess, list_buffers, list_buffers_new,
                               epsilon)
        if config.alg.name == 'm1':
            # Copy prime parameters, which already had policy update, to main
            agents.update_main(sess)
            if config.designer_m.pipeline and (
                    not config.alg.restore_do_not_train_designer):
                list_buffers = list_buffers_new
        else:
            # Train agents
            if config.agent.share_parameter:
                agents.train(sess, list_buffers[1], epsilon)
            else:
                for idx, agent in enumerate(agents):
                    agent.train(sess, list_buffers[idx+1], epsilon)
        step_train += 1
        if idx_episode % period == 0:
            print('Evaluating\n')
            (returns, productivity, equality,
             coin, stone, wood) = evaluate.test_foundation(
                 n_eval, env, sess, designer, agents, phase,
                 config.alg.name, update_alg,
                 path=log_path, idx_train_eps=idx_episode,
                 meas_tax_during_train=config.main.meas_tax_during_train,
                 meas_income_during_train=config.main.meas_income_during_train)

            s = '%d,%d,%d,%d' % (idx_episode, step_train, step,
                                 time.time()-t_start)
            for idx in range(env.n_agents):
                s += ',%.2f' % returns[idx+1]
            s += ',%.2f,' % returns[0]
            s += ','.join(['%.2f' % val for val in np.concatenate([coin, stone, wood])])
            s += ',%.2f,%.2f\n' % (productivity, equality)
            with open(os.path.join(log_path, 'log.csv'), 'a') as f:
                f.write(s)

        if idx_episode % save_period == 0:
            saver.save(sess, os.path.join(log_path, '%s.%d'%(
                model_name, idx_episode)))

        if epsilon > config.agent.epsilon_end:
            epsilon -= epsilon_step
        
    saver.save(sess, os.path.join(log_path, model_name))

def run_episode(sess, env, designer, agents, epsilon, phase,
                alg_name, update_alg, idx_episode, val_traj=False,
                prime=False):
    """Runs one episode.

    Args:
        sess: TF session
        env: an environment object with Gym interface
        designer: the incentive designer object
        agents: either a list of agent objects, or an agent object
        epsilon: float exploration rate
        phase: 1 means free market, ID is forced to take no-op
        alg_name: string
        update_alg: str, 'ppo' or 'ac'
        idx_episode: int
        val_traj: bool

    Returns: list of trajectories of ID and agents
    """
    share_parameter = not isinstance(agents, list)
    if share_parameter:
        # One for planner, one for the agents
        if phase == 2 and alg_name in ['m1', 'm2']:
            list_buffers = [buffers.BufferFoundationM2(),
                            buffers.Buffer()]
        else:
            list_buffers = [buffers.Buffer() for _ in range(2)]
    else:
        list_buffers = [buffers.Buffer() for _ in range(env.n_agents+1)]
    obs = env.reset()
    done = False

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
            if update_alg == "ppo":
                if alg_name == 'm1':
                    (actions, agent_log_probs,
                     agent_actor_state) = agents.run_actor(
                         list_obs_t, list_obs_f, list_action_mask,
                         sess, epsilon, agent_actor_state, prime)
                else:
                    (actions, agent_log_probs,
                     agent_actor_state) = agents.run_actor(
                         list_obs_t, list_obs_f, list_action_mask,
                         sess, epsilon, agent_actor_state)
            else:
                if alg_name == 'm1':
                    actions = agents.run_actor(list_obs_t, list_obs_f,
                                               list_action_mask, sess,
                                               epsilon, prime)
                else:
                    actions = agents.run_actor(list_obs_t, list_obs_f,
                                               list_action_mask, sess,
                                               epsilon)
            action_dict = {str(idx): actions[idx] for idx
                           in range(env.n_agents)}
        else:
            action_dict = {}
            agent_log_probs = []
            for idx, agent in enumerate(agents):
                obs_i = obs[str(idx)]
                action = agent.run_actor(
                    obs_i['tensor'], obs_i['flat'], [obs_i['action_mask']],
                    sess, epsilon)
                action_dict[str(idx)] = action
        
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
                if update_alg == "ppo" and alg_name not in ['m1', 'm2']:
                    (action_dict['p'], designer_log_probs,
                     designer_actor_state) = designer.run_actor(
                        obs_p['tensor'], obs_p['flat'], action_masks,
                        sess, epsilon, designer_actor_state)
                elif alg_name == 'dual_RL' and update_alg == 'ac':
                    action_dict['p'] = designer.run_actor(
                        obs_p['tensor'], obs_p['flat'], action_masks,
                        sess, epsilon)
                elif alg_name in ['m1', 'm2']:
                    action_dict['p'] = [0]
                    component = env.env._components_dict['PeriodicBracketTax']
                    if component.tax_cycle_pos == 1:
                        tax_rates, noise, designer_actor_state = designer.compute_tax(
                            obs_p['tensor'], obs_p['flat'], sess, True, designer_actor_state)
                        # Apply these tax rates to the env
                        component.set_tax_rates(list(tax_rates))
                        # These obs are updated at tax_cycle_pos==1 and
                        # held constant for the rest of the tax cycle,
                        # so that recomputation of tax in metagrad step
                        # uses them as inputs at the time step when
                        # enact_tax=True
                        obs_flat_constant = obs_p['flat']
                        obs_tensor_constant = obs_p['tensor']

        if phase == 2 and alg_name in ['m1', 'm2'] and not val_traj:
            # Only needed for the first trajectory
            # Need to record before env.step changes things
            component = env.env._components_dict['PeriodicBracketTax']
            total_endowment_coin = [None] * env.n_agents
            last_coin = component.last_coin
            escrow_coin = [None] * env.n_agents
            util_prev = [env.env.curr_optimization_metric[idx]
                         for idx in range(env.n_agents)]
            inventory_coin = [None] * env.n_agents
            total_labor = [None] * env.n_agents
            curr_rate_max = component.curr_rate_max
            # Store indicator of whether tax will be applied at the
            # env.step immediately below this line.
            enact_tax = (1 if (component.tax_cycle_pos >= component.period)
                         else 0)
            for idx in range(env.n_agents):
                agent = env.env.get_agent(str(idx))
                total_endowment_coin[idx] = agent.total_endowment('Coin')
                escrow_coin[idx] = agent.escrow['Coin']
                inventory_coin[idx] = agent.inventory['Coin']
                total_labor[idx] = agent.state['endogenous']['Labor']

        # Note: for m2, these rewards already have tax applied.
        obs_next, rewards, done, info = env.step(action_dict)
        
        # Store ID experience
        obs_p_next = obs_next['p']
        if phase == 2 and alg_name in ['m1', 'm2', 'dual_RL']:
            if alg_name in ['m1', 'm2']:
                # Record everything needed to duplicate calculation of
                # taxation and rewards within the TF graph
                reward_p = (env.env.metrics['social/productivity'] *
                            env.env.metrics['social/equality'] *
                            designer.r_scalar)
                list_to_add = [obs_p['flat'], obs_p['tensor'],
                               action_dict['p'], action_masks,
                               reward_p,
                               # rewards['p'],
                               obs_p_next['flat'], obs_p_next['tensor'],
                               done, noise]
                list_buffers[0].add(list_to_add)
                list_buffers[0].add_constant(obs_flat_constant,
                                             obs_tensor_constant)
                if not val_traj:  # only needed for the first trajectory
                    list_tax_info= [
                        total_endowment_coin, last_coin, escrow_coin,
                        util_prev, inventory_coin, total_labor,
                        curr_rate_max, enact_tax, idx_episode]
                    list_buffers[0].add_tax_info(list_tax_info)
                if designer.nn.use_lstm_actor:
                    list_buffers[0].add_lstm(designer_actor_state)
            else:
                list_buffers[0].add([
                        obs_p['flat'], obs_p['tensor'],
                        action_dict['p'], action_masks,
                        rewards['p'],
                        obs_p_next['flat'], obs_p_next['tensor'],
                        done])
                if update_alg == 'ppo':
                    list_buffers[0].add_ppo(designer_log_probs, designer_actor_state)

        # Agents' experiences
        if share_parameter:
            list_obs_next_t = []
            list_obs_next_f = []
            list_rewards = []
            for idx in range(env.n_agents):
                o_i = obs_next[str(idx)]
                list_obs_next_t.append(o_i['tensor'])
                list_obs_next_f.append(o_i['flat'])
                list_rewards.append(rewards[str(idx)])
            
            list_buffers[1].add([list_obs_f, list_obs_t,
                                     actions, list_action_mask,
                                     np.array(list_rewards),
                                     list_obs_next_f, list_obs_next_t,
                                     done])
            if update_alg == 'ppo':
                list_buffers[1].add_ppo(agent_log_probs, agent_actor_state)

        else:
            for idx, buf in enumerate(list_buffers[1:]):
                obs_i = obs[str(idx)]
                obs_i_next = obs_next[str(idx)]
                buf.add([obs_i['flat'], obs_i['tensor'],
                             [action_dict[str(idx)]],
                             [obs_i['action_mask']],
                             rewards[str(idx)], obs_i_next['flat'],
                             obs_i_next['tensor'], done])

        obs = obs_next

    return list_buffers


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('ID', type=str,
                        choices=['us_federal', 'free_market', 'ppo', 'curr'])
    args = parser.parse_args()

    if args.ID == 'us_federal':
        from incentive_design.configs import config_foundation_us_federal
        config = config_foundation_us_federal.get_config()
    elif args.ID == 'free_market':
        from incentive_design.configs import config_foundation_free_market
        config = config_foundation_free_market.get_config()
    elif args.ID == 'ppo':
        from incentive_design.configs import config_foundation_ppo
        config = config_foundation_ppo.get_config()
    elif args.ID == 'curr':
        from incentive_design.configs import config_foundation_ppo_curriculum
        config = config_foundation_ppo_curriculum.get_config()

    train_function(config)
