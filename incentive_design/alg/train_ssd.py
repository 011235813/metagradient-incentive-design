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
from incentive_design.env import ssd
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
    if not config.alg.resume:
        os.makedirs(log_path, exist_ok=True)

        with open(os.path.join(log_path, 'config.json'), 'w') as f:
            json.dump(config, f, indent=4, sort_keys=True)

    n_episodes = int(config.alg.n_episodes)
    n_eval = config.alg.n_eval
    period = config.alg.period
    
    epsilon_step = ((config.agent.epsilon_start - config.agent.epsilon_end) /
                    config.agent.epsilon_div)
    if not config.alg.resume:
        epsilon = config.agent.epsilon_start
    else:
        epsilon_decrease = config.alg.resume_ep * epsilon_step
        epsilon = max(config.agent.epsilon_end,
                      (config.agent.epsilon_start - epsilon_decrease))
            
    env = ssd.Env(config.env)
    
    update_alg = 'ac'
    if "update_alg" in config.alg:
        update_alg = config.alg.update_alg

    # Initialize incentive designer
    if config.alg.name == 'm1':
        from incentive_design.alg.id_ssd import MetaGrad1Step
        designer = MetaGrad1Step(0, 'designer', config.designer_m,
                                 config.env.n_action_types,
                                 env.dim_obs, env.n_agents, config.nn,
                                 env.dim_action_for_incentive,
                                 config.env.r_multiplier)
    elif config.alg.name == 'dual_RL':
        from incentive_design.alg.id_ssd_dualrl import Agent
        designer = Agent(0, 'designer', config.designer,
                         config.env.n_action_types,
                         env.dim_obs, env.n_agents, config.nn,
                         env.dim_action_for_incentive,
                         config.env.r_multiplier)

    # Initialize agents
    if update_alg == 'ac':
        if config.alg.name == 'm1':
            from incentive_design.alg import actor_critic_ssd_m1 as ac
            agents = ac.ActorCritic(
                1, 'agent', config.agent, env.l_action,
                env.dim_obs, env.n_agents, config.nn,
                config.env.n_action_types)
        elif config.alg.name == 'dual_RL':
            from incentive_design.alg import actor_critic_ssd_ps as ac
            agents = ac.ActorCritic(
                1, 'agent', config.agent, env.l_action,
                env.dim_obs, env.n_agents, config.nn)

    # Initialization for m1
    if config.alg.name == 'm1':
        designer.receive_agents(agents)
        agents.receive_designer(designer)
        agents.create_policy_gradient_op()
        agents.create_critic_train_op()
        agents.create_update_op()
        if not config.alg.restore_do_not_train_designer:
            if config.designer_m.use_critic:
                designer.create_critic_train_op()
            designer.create_incentive_train_op()

    config_proto = tf.ConfigProto()
    if config.main.use_gpu:
        config_proto.device_count['GPU'] = 1
        config_proto.gpu_options.allow_growth = True
    else:
        config_proto.device_count['GPU'] = 0
    sess = tf.Session(config=config_proto)
    sess.run(tf.global_variables_initializer())

    if config.alg.restore_do_not_train_designer:
        # Only restore designer, not agents.
        to_restore = []
        for v in tf.trainable_variables():
            if 'agent' not in v.name.split('/'):
                to_restore.append(v)
        saver = tf.train.Saver(var_list=to_restore,
                               max_to_keep=config.main.max_to_keep)
    else:
        saver = tf.train.Saver(max_to_keep=config.main.max_to_keep)

    if (config.agent.share_parameter and
        hasattr(agents, 'list_initialize_v_ops')):
        sess.run(agents.list_initialize_v_ops)

    if config.alg.resume:
        p_prefix = os.path.join('..', 'results', exp_name,
                                config.main.dir_restore)
        p = os.path.join(p_prefix, config.main.model_restore + '.'
                         + str(config.alg.resume_ep))
        if not os.path.isfile(p + '.index'):
            raise FileNotFoundError('Could not find %s' % p)
        saver.restore(sess, p)


    if config.alg.name == 'm1' and not config.alg.resume:
        sess.run(designer.list_initialize_v_ops)

    if config.alg.name == 'm1':
        sess.run(agents.list_copy_main_to_prime_ops)

    if not config.alg.resume:
        list_agent_meas = []
        list_suffix = ['received', 'reward_env',
                       'reward_total', 'waste_cleared',
                       'r_riverside', 'r_beam', 'r_cleared']
        for agent_id in range(1, env.n_agents + 1):
            for suffix in list_suffix:
                list_agent_meas.append('A%d_%s' % (agent_id, suffix))
        header = 'episode,step_train,step,time,'
        header += ','.join(list_agent_meas)
        header += ',reward_env_total\n'
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

    if not config.alg.resume:
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

        if (config.alg.name != 'm1' or config.alg.restore_do_not_train_designer
            or (list_buffers is None or not config.designer_m.pipeline)):
            # For m1 pipeline=T, this is run only for the first episode
            # to start pipeline.
            # Afterwards, the \hat{traj} will exist
            # list_buffers[0] is the designer's trajectory,
            # list_buffers[1:] are the agents' trajectories
            list_buffers = run_episode(sess, env, designer, agents,
                                       epsilon, config.alg.name,
                                       update_alg, idx_episode, False,
                                       prime=False)
            step += len(list_buffers[1].reward)
            idx_episode += 1
        # Update the prime parameters
        if config.alg.name == 'm1':
            agents.update(sess, list_buffers[1], epsilon)

        if not config.alg.restore_do_not_train_designer:
            # Collect validation trajectory for method m1 or m2
            if config.alg.name in ['m1', 'm2']:
                val_traj = (config.alg.name == 'm2') or (
                    config.alg.name == 'm1' and
                    not config.designer_m.pipeline)
                list_buffers_new = run_episode(
                    sess, env, designer, agents, epsilon,
                    config.alg.name, update_alg, idx_episode, val_traj,
                    prime=config.alg.name=='m1')
                step += len(list_buffers_new[1].reward)
                idx_episode += 1
            # Train designer
            if config.alg.name in ['dual_RL']:
                designer.train(sess, list_buffers[0], epsilon)
            elif config.alg.name in ['m1', 'm2']:
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
            # print('Evaluating\n')
            (received, reward_env, reward_total,
             waste_cleared, r_riverside, r_beam,
             r_cleared, r_designer) = evaluate.test_ssd(
                 n_eval, env, sess, designer, agents, config.alg.name)

            s = '%d,%d,%d,%d' % (idx_episode, step_train, step,
                                 time.time()-t_start)
            combined = np.stack([received, reward_env,
                                 reward_total, waste_cleared,
                                 r_riverside, r_beam, r_cleared])
            for idx in range(env.n_agents):
                s += ','
                s += '{:.2e},{:.2f},{:.2e},{:.2f},{:.2e},{:.2e},{:.2e}'.format(
                    *combined[:, idx])
            s += ',%.2e\n' % r_designer
            with open(os.path.join(log_path, 'log.csv'), 'a') as f:
                f.write(s)

        if idx_episode % save_period == 0:
            saver.save(sess, os.path.join(log_path, '%s.%d'%(
                model_name, idx_episode)))

        if epsilon > config.agent.epsilon_end:
            epsilon -= epsilon_step
        
    saver.save(sess, os.path.join(log_path, model_name))

def run_episode(sess, env, designer, agents, epsilon,
                alg_name, update_alg, idx_episode, val_traj=False,
                prime=False):
    """Runs one episode.

    Args:
        sess: TF session
        env: an environment object with Gym interface
        designer: the incentive designer object
        agents: either a list of agent objects, or an agent object
        epsilon: float exploration rate
        alg_name: string
        update_alg: str, 'ppo' or 'ac'
        idx_episode: int
        val_traj: bool

    Returns: list of trajectories of ID and agents
    """
    share_parameter = not isinstance(agents, list)
    if share_parameter:
        # One for planner, one for the agents
        list_buffers = [buffers.Designer_SSD(),
                        buffers.Agents_SSD()]
    else:
        raise NotImplementedError
    obs = env.reset()
    done = False

    agent_actor_state = None
    designer_actor_state = None
    while not done:

        if share_parameter:
            list_obs_t = []
            for idx in range(env.n_agents):
                list_obs_t.append(obs[str(idx)])
            if update_alg == "ppo":
                if alg_name == 'm1':
                    actions = agents.run_actor(
                        list_obs_t, sess, epsilon, prime)
                else:
                    (actions, agent_log_probs,
                     agent_actor_state) = agents.run_actor(
                         list_obs_t, sess, epsilon, agent_actor_state)
            else:
                if alg_name == 'm1':
                    actions = agents.run_actor(
                        list_obs_t, sess, epsilon, prime)
                else:
                    actions = agents.run_actor(list_obs_t, sess, epsilon)
            list_binary_actions = []
            for action in np.squeeze(actions):
                list_binary_actions.append(
                    1 if action == env.cleaning_action_idx else 0)
        else:
            raise NotImplementedError
        
        # Note: these rewards do not contain incentives
        obs_next, rewards, done, info = env.step(np.squeeze(actions))

        # Preprocess agents' actions to have values in {0,1,2},
        # 0: clean, 1: collected apples, 2: else
        action_type = []
        for idx in range(env.n_agents):
            if actions[idx] == env.cleaning_action_idx:
                action_type.append(0)
            elif rewards[str(idx)] > 0:
                action_type.append(1)
            else:
                action_type.append(2)

        if alg_name == 'dual_RL':
            incentives, r_sample = designer.run_actor(
                obs['p'], list_binary_actions, sess)
        elif alg_name in ['m1', 'm2']:
            # incentives is a length 3 array, one for each action type
            incentives = designer.compute_incentive(
                obs['p'], list_binary_actions, sess)
        
        # Apply incentives to agents based on their (processed) actions
        # incentives: [value for clean, value for collect, else]
        # incentive_agents has length n_agents
        incentive_agents = np.take(incentives, action_type)

        # Store ID experience
        list_to_add = [obs['p'], list_binary_actions,
                       rewards['p'], obs_next['p'], done]
        list_buffers[0].add(list_to_add)
        if alg_name == 'dual_RL':
            list_buffers[0].add_r_sample(r_sample)

        # Agents' experiences
        if share_parameter:
            list_obs_next_t = []
            list_rewards = []
            list_r_env = []
            for idx in range(env.n_agents):
                list_obs_next_t.append(obs_next[str(idx)])
                total_reward = rewards[str(idx)] + incentive_agents[idx]
                list_rewards.append(total_reward)
                list_r_env.append(rewards[str(idx)])
            list_buffers[1].add([
                list_obs_t, actions, np.array(action_type),
                np.array(list_rewards), np.array(list_r_env),
                list_obs_next_t, done])
            if alg_name == 'dual_RL' and update_alg == 'ppo':
                list_buffers[1].add_ppo(agent_log_probs, agent_actor_state)
        else:
            raise NotImplementedError

        obs = obs_next

    return list_buffers


if __name__ == '__main__':

    from incentive_design.configs import config_ssd
    config = config_ssd.get_config()

    train_function(config)
