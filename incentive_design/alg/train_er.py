"""Trains PG and oracle ID agents on Escape Room game."""

from __future__ import division
from __future__ import print_function

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import tensorflow as tf

from incentive_design.alg import evaluate
from incentive_design.env import escape_room


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

    os.makedirs(log_path, exist_ok=True)

    # Keep a record of parameters used for this run
    with open(os.path.join(log_path, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4, sort_keys=True)

    n_episodes = int(config.alg.n_episodes)
    n_eval = config.alg.n_eval
    period = config.alg.period

    epsilon = config.agent.epsilon_start
    epsilon_step = (
        epsilon - config.agent.epsilon_end) / config.agent.epsilon_div

    env = escape_room.Env(config.env)

    if config.alg.name in ['m1', 'm2']:
        from incentive_design.alg import incentive_designer
        designer = incentive_designer.MetaGrad1Step(
            0, config.designer_m, env.l_action, env.l_obs, env.n_agents,
            config.nn_m, config.env.r_multiplier)
        if config.alg.name == 'm1':
            if config.alg.update_alg == 'pg':
                from incentive_design.alg import pg_agent_m1 as agent
            elif config.alg.update_alg == 'ac':
                from incentive_design.alg import ac_agent_m1 as agent
        else:
            from incentive_design.alg import pg_agent_m2 as agent
    else:
        from incentive_design.alg import id_er_pg
        if config.designer.action_space == 'continuous':
            l_action_ID = 0
        else:
            l_action_ID = env.l_action_discrete_dualRL
        designer = id_er_pg.DualRLPG(
            0, config.designer, env.l_action, env.l_obs, env.n_agents,
            config.nn_dual, config.env.r_multiplier, l_action_ID)
        from incentive_design.alg import pg_agent as agent

    list_agents = []
    for agent_id in range(1, env.n_agents+1):
        list_agents.append(
            agent.Agent(agent_id, config.agent, env.l_action,
                        env.l_obs, env.n_agents,
                        config.nn_agent, r_multiplier=2))

    # Setup for meta-gradient
    if config.alg.name in ['m1', 'm2']:
        # ID receives list of all agents (starts at index 1)
        designer.receive_list_of_agents(list_agents)
        # Agents hold ID model in oracle version
        for idx in range(env.n_agents):
            list_agents[idx].receive_designer(designer)
            list_agents[idx].create_policy_gradient_op()
            list_agents[idx].create_update_op()
            if config.alg.update_alg == 'ac':
                list_agents[idx].create_critic_train_op()

        designer.create_incentive_train_op()

    config_proto = tf.ConfigProto()
    if config.main.use_gpu:
        config_proto.device_count['GPU'] = 1
        config_proto.gpu_options.allow_growth = True
    else:
        config_proto.device_count['GPU'] = 0
    sess = tf.Session(config=config_proto)
    sess.run(tf.global_variables_initializer())

    list_agent_meas = []
    list_suffix = ['reward_total', 'n_lever', 'n_door',
                   'incentives', 'r-lever', 'r-start', 'r-door']
    for agent_id in range(1, env.n_agents + 1):
        for suffix in list_suffix:
            list_agent_meas.append('A%d_%s' % (agent_id, suffix))

    saver = tf.train.Saver(max_to_keep=config.main.max_to_keep)

    header = 'episode,step_train,step,time,'
    header += ','.join(list_agent_meas)
    header += ',steps_per_eps,r_id'

    if config.env.show_agent_spec:
        for agent_id in range(1, env.n_agents + 1):
            header += ',A%d_%s' % (agent_id, "lever_incentive")
            header += ',A%d_%s' % (agent_id, "optimal_incentive")
    header += '\n'
    with open(os.path.join(log_path, 'log.csv'), 'w') as f:
        f.write(header)    

    idx_episode = 0
    step = 0
    step_train = 0
    t_start = time.time()
    list_buffers = None
    while idx_episode < n_episodes:

        # list_buffers[0] is the designer's trajectory,
        # list_buffers[1:] are the agents' trajectories
        if (not config.alg.name in ['m1', 'm2']) or (
                list_buffers is None or not config.designer_m.pipeline):
            # Only needed for the first episode to start the pipeline.
            # Afterwards, the \hat{traj} will exist
            list_buffers = run_episode(sess, env, designer, list_agents,
                                       epsilon, config.alg.name, prime=False)
            step += len(list_buffers[0].obs)
            idx_episode += 1

        # update the prime policy, which is used below to collect the validation trajectory
        if config.alg.name == 'm1':
            for idx, agent in enumerate(list_agents):
                agent.update(sess, list_buffers[idx+1], epsilon)

        # Collect a validation trajectory. If prime==T, then
        # trajectory is generated by the updated agent policies.
        # Otherwise, it is generated by the original policies.
        if config.alg.name in ['m1', 'm2']:
            list_buffers_new = run_episode(sess, env, designer, list_agents,
                                           epsilon, config.alg.name,
                                           prime=config.alg.name=='m1')
            step += len(list_buffers_new[0].obs)
            idx_episode += 1

        # Update ID.
        if config.alg.name in ['m1', 'm2']:
            designer.train_incentive(sess, list_buffers,
                                     list_buffers_new, epsilon)
        else:
            designer.train(sess, list_buffers[0])

        if config.alg.name in ['m1', 'm2']:
            if config.alg.name == 'm1':
                # Copy prime parameters, which already had a gradient step, to main
                for idx, agent in enumerate(list_agents):
                    agent.update_main(sess)
                if config.designer_m.pipeline:
                    list_buffers = list_buffers_new
            else:
                # Take a gradient step for main parameters
                for idx, agent in enumerate(list_agents):
                    agent.update(sess, list_buffers[idx+1], epsilon)
                if config.designer_m.pipeline:
                    list_buffers = list_buffers_new
        else:
            for idx, agent in enumerate(list_agents):
                agent.train(sess, list_buffers[idx+1], epsilon)

        step_train += 1

        if idx_episode % period == 0:
            # print('Evaluating')
            if config.env.name == 'er':
                (rewards_total, n_move_lever, n_move_door, incentives,
                 r_id, r_lever, r_start, r_door, steps_per_episode,
                 lever_incentives,
                 optimal_incentives) = evaluate.test_escape_room(
                     n_eval, env, sess, designer, list_agents,
                     config.alg.name, config.env.show_agent_spec)

                matrix_combined = np.stack(
                    [rewards_total, n_move_lever, n_move_door,
                     incentives, r_lever, r_start, r_door])

            s = '%d,%d,%d,%d' % (idx_episode, step_train, step,
                                 time.time()-t_start)
            for idx in range(env.n_agents):
                s += ','
                s += ('{:.3e},{:.3e},{:.3e},{:.3e},'
                      '{:.3e},{:.3e},{:.3e}').format(
                          *matrix_combined[:, idx])
            s += ',%.2f,%.3f' % (steps_per_episode, r_id)
            if config.env.show_agent_spec:
                for idx in range(env.n_agents): 
                    s += ',%.3e' % (lever_incentives[idx])
                    s += ',%.3e' % (optimal_incentives[idx])
            s += '\n'
            with open(os.path.join(log_path, 'log.csv'), 'a') as f:
                f.write(s)

        if idx_episode % save_period == 0:
            saver.save(sess, os.path.join(log_path, '%s.%d'%(
                model_name, idx_episode)))

        if epsilon > config.agent.epsilon_end:
            epsilon -= epsilon_step

    saver.save(sess, os.path.join(log_path, model_name))
    

def run_episode(sess, env, designer, list_agents, epsilon, alg_name,
                prime=False):
    """Runs one episode.

    Args:
        sess: TF session
        env: an environment object with Gym interface
        designer: the incentive designer object
        list_agents: list of agent objects
        epsilon: float exploration rate
        alg_name: str
        prime: if True, runs the agent's prime parameters

    Returns: list of trajectories of ID and agents
    """
    list_buffers = [Buffer_ID()]
    list_buffers += [Buffer() for _ in range(env.n_agents)]
    list_obs = env.reset()
    done = False

    while not done:
        list_actions = []
        for agent in list_agents:
            if alg_name in ['m1', 'm2']:
                action = agent.run_actor(list_obs[agent.agent_id], sess,
                                         epsilon, prime)
            else:
                action = agent.run_actor(list_obs[agent.agent_id], sess,
                                         epsilon)
            list_actions.append(action)

        # ID gives incentives based on obs and agents' actions
        if alg_name in ['m1', 'm2']:
            incentives = designer.compute_incentive(list_obs[0],
                                                    list_actions, sess)
            if designer.output_type == 'action':
                # In this case, incentives is a vector of values for
                # each possible action. Need to create a vector of
                # actual agent incentives based on the chosen actions
                incentives = np.take(incentives, list_actions)
        elif alg_name == 'dual_RL':
            if designer.action_space == 'continuous':
                incentives, r_sample = designer.compute_incentive(
                    list_obs[0], list_actions, sess)
            else:
                action_designer = designer.compute_incentive(
                    list_obs[0], list_actions, sess)
                # Map action to vector of incentive values
                incentives = env.map_discrete_action_to_incentives(
                    action_designer, list_actions)

        list_obs_next, env_rewards, done = env.step(list_actions)

        if alg_name == 'dual_RL':
            # Cost for giving rewards is accounted here for dual_RL
            # Cost is accounted inside loss function for m1/m2
            env_rewards[0] -= np.sum(incentives)

        # ID's experiences
        list_buffers[0].add([list_obs[0], list_actions, env_rewards[0],
                             list_obs_next[0], done])
        if alg_name == 'dual_RL':
            if designer.action_space == 'continuous':
                list_buffers[0].add_r_sample(r_sample)
            else:
                list_buffers[0].add_action(action_designer)
            
        # Agents' experiences
        for idx, buf in enumerate(list_buffers[1:]):
            agent_id = idx+1
            buf.add([list_obs[agent_id], list_actions[idx],
                     env_rewards[agent_id], incentives[idx],
                     list_obs_next[agent_id], done])

        list_obs = list_obs_next

    return list_buffers


class Buffer(object):
    """An Agent's trajectory buffer."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.obs = []
        self.action = []
        self.r_env = []
        self.incentive_received = []
        self.obs_next = []
        self.done = []

    def add(self, transition):
        self.obs.append(transition[0])
        self.action.append(transition[1])
        self.r_env.append(transition[2])
        self.incentive_received.append(transition[3])
        self.obs_next.append(transition[4])
        self.done.append(transition[5])


class Buffer_ID(object):
    """Incentive designer's trajectory buffer."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.obs = []
        self.action_all = []
        self.reward = []
        self.obs_next = []
        self.done = []
        self.action = []
        self.r_sample = []

    def add(self, transition):
        self.obs.append(transition[0])
        self.action_all.append(transition[1])
        self.reward.append(transition[2])
        self.obs_next.append(transition[3])
        self.done.append(transition[4])

    def add_action(self, action):
        """Store action int, used by discrete action dual-RL."""
        self.action.append(action)

    def add_r_sample(self, r_sample):
        """Store sample from Gaussian, used by continuous action dual-RL."""
        self.r_sample.append(r_sample)


if __name__ == '__main__':

    from incentive_design.configs import config_er_pg
    config = config_er_pg.get_config()

    train_function(config)
