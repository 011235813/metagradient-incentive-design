import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import json
import math
import os
import random
import time

from incentive_design.alg.baumann_rl import config_er_baumann
from incentive_design.env import escape_room
from incentive_design.alg.baumann_rl.Agents import Actor_Critic_Agent, PG, Critic_Variant
from incentive_design.alg.baumann_rl.Planning_Agent import Planning_Agent


def train(config):

    seed = config.main.seed
    np.random.seed(seed)
    random.seed(seed)
    tf.set_random_seed(seed)    

    dir_name = config.main.dir_name
    exp_name = config.main.exp_name
    cwd_suffix = os.getcwd().split('/')[-1]
    if cwd_suffix == 'baumann_rl':
        log_path = os.path.join('..', '..', 'results', exp_name, dir_name)
    elif cwd_suffix == 'alg':
        log_path = os.path.join('..', 'results', exp_name, dir_name)
    else:
        raise ValueError('Did not expect code to be run from %s' % os.getcwd())
    model_name = config.main.model_name
    save_period = config.main.save_period
    os.makedirs(log_path, exist_ok=True)

    with open(os.path.join(log_path, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4, sort_keys=True)

    n_episodes = config.alg.n_episodes
    n_eval = config.alg.n_eval
    period = config.alg.period

    action_flip_prob = config.designer.action_flip_prob
    with_redistribution = config.designer.with_redistribution
    n_planning_eps = config.designer.n_planning_eps

    epsilon = config.agent.epsilon_start
    epsilon_step = (
        epsilon - config.agent.epsilon_end) / config.agent.epsilon_div

    config_proto = tf.ConfigProto()
    if config.main.use_gpu:
        config_proto.device_count['GPU'] = 1
        config_proto.gpu_options.allow_growth = True
    else:
        config_proto.device_count['GPU'] = 0
    sess = tf.Session(config=config_proto)

    env = escape_room.Env(config.env)
    agents = create_population(sess, env, config.env.n_agents, config.agent)
    planning_agent = Planning_Agent(sess, env, agents, config.designer)

    sess.run(tf.global_variables_initializer())

    list_agent_meas = []
    list_suffix = ['reward_total', 'incentives']
    for agent_id in range(1, env.n_agents + 1):
        for suffix in list_suffix:
            list_agent_meas.append('A%d_%s' % (agent_id, suffix))

    saver = tf.train.Saver(max_to_keep=config.main.max_to_keep)

    header = 'episode,step_train,step,time,'
    header += ','.join(list_agent_meas)
    header += ',steps_per_eps,r_id'
    header += '\n'
    with open(os.path.join(log_path, 'log.csv'), 'w') as f:
        f.write(header)    

    if config.agent.alg_name == 'pg':
        list_buffers = [Buffer() for _ in range(env.n_agents)]
    step = 0
    step_train = 0
    t_start = time.time()
    for idx_episode in range(1, n_episodes+1):
        # print('Episode', idx_episode)
        # s[0] is planner's obs, remaining s[1:] are agents' obs
        s = env.reset()
        if config.agent.alg_name == 'pg':
            for buf in list_buffers:
                buf.reset()

        while True:

            actions = [agent.choose_action(s[idx+1], epsilon)
                       for idx, agent in enumerate(agents)]

            s_, rewards, done = env.step(actions)

            if planning_agent is not None and idx_episode < n_planning_eps:
                planning_rs = planning_agent.choose_action(s[0], actions)

                if with_redistribution:
                    sum_planning_r = sum(planning_rs)
                    mean_planning_r = sum_planning_r / env.n_agents
                    planning_rs = [r-mean_planning_r for r in planning_rs]
                # total rewards given to agents
                reward_agents = [ sum(r) for r in zip(rewards[1:], planning_rs)]

                # Training planning agent
                planning_agent.learn(s[0], actions, rewards[0], done, epsilon)
                step_train += 1

            if config.agent.alg_name == 'ac':
                # update at every env step
                for idx, agent in enumerate(agents):
                    agent.learn(s[idx+1], actions[idx], reward_agents[idx],
                                s_[idx+1], done, epsilon, s[1:], s_[1:])
            elif config.agent.alg_name == 'pg':
                for idx, buf in enumerate(list_buffers):
                    buf.add([s[idx+1], actions[idx], reward_agents[idx]])

            s = s_

            if done:
                step += env.steps
                break

        if config.agent.alg_name == 'pg':
            # batch update
            for idx, agent in enumerate(agents):
                agent.learn(sess, list_buffers[idx], epsilon)

        if epsilon > config.agent.epsilon_end:
            epsilon -= epsilon_step

        # status updates
        if idx_episode % period == 0:
            # print('Episode {} finished.'.format(idx_episode))
            (rewards_total, incentives, r_id,
             steps_per_episode) = test_amd(n_eval, env, planning_agent, agents)

            matrix_combined = np.stack([rewards_total, incentives])
            s = '%d,%d,%d,%d' % (idx_episode, step_train, step,
                                 time.time()-t_start)
            for idx in range(env.n_agents):
                s += ','
                s += ('{:.3e},{:.3e}').format(*matrix_combined[:, idx])
            s += ',%.2f,%.3f' % (steps_per_episode, r_id)
            s += '\n'

            with open(os.path.join(log_path, 'log.csv'), 'a') as f:
                f.write(s)

        if idx_episode % save_period == 0:
            saver.save(sess, os.path.join(log_path, '%s.%d'%(
                model_name, idx_episode)))

    saver.save(sess, os.path.join(log_path, model_name))


def test_amd(n_eval, env, designer, agents):

    rewards_total = np.zeros(env.n_agents)
    incentives = np.zeros(env.n_agents)
    reward_id = 0
    total_steps = 0
    epsilon = 0
    for idx_episode in range(1, n_eval+1):

        s = env.reset()
        done = False
        while not done:

            actions = [agent.choose_action(s[idx+1], epsilon)
                       for idx, agent in enumerate(agents)]
            s_, rewards, done = env.step(actions)

            planning_rs = designer.choose_action(s[0], actions)

            incentives += planning_rs
            
            rewards_total += rewards[1:]
            rewards_total += planning_rs
            
            reward_id += rewards[0]
            reward_id -= np.sum(planning_rs)

            s = s_

        total_steps += env.steps

    rewards_total /= n_eval
    incentives /= n_eval
    reward_id /= n_eval
    steps_per_episode = total_steps / n_eval

    return rewards_total, incentives, reward_id, steps_per_episode
    

def create_population(sess, env, n_agents, config):

    if config.alg_name == 'pg':
        l = [PG(sess, env.l_obs, env.l_action, config, agent_idx=i)
             for i in range(n_agents)]
    elif config.alg_name == 'ac':
        critic_variant = Critic_Variant.INDEPENDENT
        l = [Actor_Critic_Agent(sess, env,
                            learning_rate=config.lr,
                            gamma=config.gamma,
                            n_units_actor=config.nn,
                            n_units_critic=config.nn,
                            agent_idx=i,
                            critic_variant=critic_variant)
         for i in range(n_agents)]

        # Pass list of agents for centralized critic
        if critic_variant is Critic_Variant.CENTRALIZED:
            for agent in l:
                agent.pass_agent_list(l)

    return l


def run_episode(sess, env, planning_agent, agents, epsilon, *args):

    list_buffers = [Buffer() for _ in range(env.n_agents)]
    s = env.reset()
    done = False
    while not done:
        try:
            actions = [agent.choose_action(s[idx+1], epsilon)
                       for idx, agent in enumerate(agents)]
        except:  # amd has NaN issues
            break

        s_, rewards, done = env.step(actions)

        planning_rs = planning_agent.choose_action(s[0], actions)

        reward_agents = [ sum(r) for r in zip(rewards[1:], planning_rs)]

        planning_agent.learn(s[0], actions, rewards[0], done, epsilon)

        for idx, buf in enumerate(list_buffers):
            buf.add([s[idx+1], actions[idx], reward_agents[idx]])

        s = s_

    return list_buffers


class Buffer(object):
    """An Agent's trajectory buffer."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.obs = []
        self.action = []
        self.r_env = []

    def add(self, transition):
        self.obs.append(transition[0])
        self.action.append(transition[1])
        self.r_env.append(transition[2])


if __name__ == "__main__":

    config = config_er_baumann.get_config()
    train(config)
