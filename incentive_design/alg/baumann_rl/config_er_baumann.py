from incentive_design.utils import configdict
import numpy as np


def get_config():

    config = configdict.ConfigDict()

    config.agent = configdict.ConfigDict()
    config.agent.alg_name = 'pg' # 'pg' or 'ac'
    config.agent.entropy_coeff = 0.01
    config.agent.epsilon_div = 1000
    config.agent.epsilon_end = 0.1
    config.agent.epsilon_start = 0.5
    config.agent.gamma = 0.99
    config.agent.lr = 0.0001
    config.agent.nn = 64

    config.alg = configdict.ConfigDict()
    config.alg.n_episodes = 50000
    config.alg.n_eval = 10
    config.alg.n_test = 100
    config.alg.name = 'amd'
    config.alg.period = 100

    config.designer = configdict.ConfigDict()
    config.designer.action_flip_prob = 0
    config.designer.cost_param = 0.000424
    config.designer.gamma = 0.95
    config.designer.grad_clip = 10.0
    config.designer.lr = 2.74e-5
    config.designer.n_planning_eps = np.inf
    config.designer.nn_h1 = 64
    config.designer.nn_h2 = 16
    config.designer.r_multiplier = 2.0
    config.designer.value_fn_variant = 'estimated'
    config.designer.with_redistribution = False

    config.env = configdict.ConfigDict()
    config.env.list_discrete_incentives = []
    config.env.max_steps = 5
    config.env.min_at_lever = 1
    config.env.n_agents = 2
    config.env.name = 'er'
    config.env.r_multiplier = 2.0
    config.env.randomize = False
    config.env.show_agent_spec = False

    config.main = configdict.ConfigDict()
    config.main.dir_name = 'n2m1_amd_pg'
    config.main.exp_name = 'er'
    config.main.max_to_keep = 100
    config.main.model_name = 'model.ckpt'
    config.main.save_period = 100000
    config.main.seed = 12340
    config.main.summarize = False
    config.main.use_gpu = False

    config.nn = configdict.ConfigDict()
    config.nn.n_h1 = 64
    config.nn.n_h2 = 32
    config.nn.n_hr1 = 64
    config.nn.n_hr2 = 16

    return config
