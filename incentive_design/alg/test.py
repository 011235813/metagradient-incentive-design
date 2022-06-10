from ai_economist import foundation

import numpy as np
import os
import random
import tensorflow as tf

from incentive_design.alg import constructor
from incentive_design.alg import evaluate


from incentive_design.env.foundation import Env
from incentive_design.test.test_foundation_basic import sample_random_actions


def test_foundation_random(config):

    seed = config.main.seed
    np.random.seed(seed)
    random.seed(seed)

    n_episodes = int(config.alg.n_test)

    env = foundation.make_env_instance(**config.env)
    # env = Env(config.env)

    idx_episode = 0
    total_returns = np.zeros(config.env.n_agents+1)


    while idx_episode < n_episodes:    

        print('\nEpisode\n')
        idx_episode += 1
        obs = env.reset()
        done = False

        swf_0 = env.metrics['social_welfare/coin_eq_times_productivity']
        print('swf 0', swf_0)

        returns = np.zeros(config.env.n_agents+1)

        id_reward = []
        swf = []
        step = 0
        while not done:

            actions = sample_random_actions(env, obs)

            swf.append(env.metrics['social_welfare/coin_eq_times_productivity'])
            obs, rewards, done, info = env.step(actions)
            step += 1
            done = done['__all__']

            id_reward.append(rewards['p'])

            returns[0] += rewards['p']
            for idx in range(config.env.n_agents):
                returns[idx+1] += rewards[str(idx)]

        print('Return', returns)
        id_return_plus_swf0 = returns[0] + swf_0
        print('ID plus swf 0', id_return_plus_swf0)
        if id_return_plus_swf0 < 0:
            for idx in range(len(id_reward)):
                print(id_reward[idx], swf[idx])
            input('Negative!!!!!')

        for idx in range(env.n_agents):
            print(env.get_agent(str(idx)).inventory)
        input('')

        total_returns += returns

    total_returns = total_returns / n_episodes

    print(total_returns)


def test_trained_model(config):
    """Restores a trained policy and logs metrics over test episodes."""
    seed = config.main.seed
    np.random.seed(seed)
    random.seed(seed)
    tf.set_random_seed(seed)    

    dir_name = config.main.dir_name
    # dir_restore = config.main.dir_restore
    exp_name = config.main.exp_name
    path = os.path.join('..', 'results', exp_name, dir_name)
    model_restore = config.main.model_restore
    if model_restore != 'model.ckpt':
        # extract suffix, which should be an int
        model_eps = int(model_restore.split('.')[-1])
    else:
        model_eps = None

    env = Env(config.env)

    designer = constructor.get_designer(config, env, phase=2)
    
    agents = constructor.get_agents(config, env)

    config_proto = tf.ConfigProto()
    if config.main.use_gpu:
        config_proto.device_count['GPU'] = 1
        config_proto.gpu_options.allow_growth = True
    else:
        config_proto.device_count['GPU'] = 0
    sess = tf.Session(config=config_proto)

    saver = tf.train.Saver()
    print('Restoring variables from %s' % dir_name)
    saver.restore(sess, os.path.join(path, model_restore))

    # measurement 1: tax rate in each bracket
    n_brackets = env.env._components_dict['PeriodicBracketTax'].n_brackets
    if n_brackets != 7:
        raise ValueError('Expected n_brackets = 7 but found %d' %
                         n_brackets)
    header = 'episode,tax_period,'
    header += ','.join(['b%d' % idx for idx in range(1, n_brackets+1)])
    header += '\n'
    fname = 'tax_rate_%d.csv' % model_eps if model_eps else 'tax_rate.csv'
    with open(os.path.join(path, fname), 'w') as f:
        f.write(header)
        
    # measurement 2: income and tax before and after redistribution
    # by each agent. Ordering will be lowest skill to highest skill
    # from left to right
    measurements = ['income_before', 'tax_paid_before', 'income_after',
                    'tax_paid_after']
    header = 'episode'
    for idx in range(1, env.n_agents+1):
        header += ','
        header += ','.join([s + '_%d'%idx for s in measurements])
    header += '\n'
    fname = 'income_and_tax_%d.csv' % model_eps if model_eps else 'income_and_tax.csv'
    with open(os.path.join(path, fname), 'w') as f:
        f.write(header)

    # measurement 3: resource collected, net income from building,
    # net income from trading for each agent
    measurements = ['resource', 'income_build', 'income_trade']
    header = 'episode'
    for idx in range(1, env.n_agents+1):
        header += ','
        header += ','.join([s + '_%d'%idx for s in measurements])
    header += '\n'
    fname = 'activity_%d.csv' % model_eps if model_eps else 'activity.csv'
    with open(os.path.join(path, fname), 'w') as f:
        f.write(header)

    _ = evaluate.test_foundation(
        config.alg.n_test, env, sess, designer, agents, 2,
        config.alg.name, config.alg.update_alg,
        extra_meas=True, path=path, model_eps=model_eps)


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--random', action='store_true',
                        help='runs random policy')
    parser.add_argument('--multiprocess', action='store_true',
                        help='tests models trained over multiple seeds')
    parser.add_argument('--n_seeds', type=int, default=4)
    parser.add_argument('--seed_min', type=int, default=12340)
    args = parser.parse_args()
    
    if args.random:
        from incentive_design.configs import config_foundation_free_market
        config = config_foundation_free_market.get_config()
        test_foundation_random(config)
    else:
        from incentive_design.configs import config_foundation_ppo
        config = config_foundation_ppo.get_config()
        if args.multiprocess:
            from multiprocessing import Process
            from copy import deepcopy
            processes = []
            dir_name_base = config.main.dir_name
            seed_min = args.seed_min
            values = range(args.n_seeds)
            for idx_run in range(len(values)):
                config_copy = deepcopy(config)
                config_copy['main']['seed'] = seed_min + idx_run
                config_copy.main.dir_name = (
                    dir_name_base + '_{:1d}'.format(idx_run))
                p = Process(target=test_trained_model, args=(config_copy,))
                p.start()
                processes.append(p)
                
            for p in processes:
                p.join()
        else:
            test_trained_model(config)
