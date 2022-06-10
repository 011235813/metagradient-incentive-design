from incentive_design.utils import configdict


def get_config():

    config = configdict.ConfigDict()

    config.agent = configdict.ConfigDict()
    config.agent.entropy_coeff = 0.0166
    config.agent.epsilon_div = 1000
    config.agent.epsilon_end = 0.0
    config.agent.epsilon_start = 0.0
    config.agent.gamma = 0.99
    config.agent.lr_actor = 9.56e-5

    config.alg = configdict.ConfigDict()
    config.alg.n_episodes = 100000
    config.alg.n_eval = 10
    config.alg.n_test = 100
    # Options: 'dual_RL', 'm1', 'm2'
    config.alg.name = 'm1'
    config.alg.period = 100
    config.alg.tune_log_uniform = True
    config.alg.update_alg = 'pg'

    config.designer = configdict.ConfigDict()
    # config.designer.action_space = 'continuous'
    config.designer.action_space = 'discrete' # 'continuous' or 'discrete'
    config.designer.entropy_coeff = 0.0289
    config.designer.epsilon_div = 5000
    config.designer.epsilon_end = 0
    config.designer.epsilon_start = 0
    config.designer.gamma = 0.99
    config.designer.grad_clip = None
    config.designer.lr_actor = 8.0e-3

    config.designer_m = configdict.ConfigDict()
    config.designer_m.agent_spec = False
    config.designer_m.gamma = 0.99
    config.designer_m.lr_cost = 6.03e-5
    config.designer_m.lr_incentive = 0.000793
    config.designer_m.lr_spec = 1e-4
    config.designer_m.optimizer = 'adam'
    config.designer_m.output_type = 'action'  # 'action' or 'agent'
    config.designer_m.pipeline = True
    config.designer_m.reg_coeff = 1.0
    config.designer_m.separate_cost_optimizer = True
    config.designer_m.spec_coeff = 0.4

    config.env = configdict.ConfigDict()
    config.env.list_discrete_incentives = [0, 1.1]  # small
    # config.env.list_discrete_incentives = [0, 1.1, 2.0] # med
    # config.env.list_discrete_incentives = [0, 0.5, 1.1, 1.5, 2.0] # large
    config.env.max_steps = 5
    config.env.min_at_lever = 2
    config.env.n_agents = 5
    config.env.name = 'er'
    config.env.r_multiplier = 2.0
    config.env.randomize = False
    config.env.show_agent_spec = False

    config.main = configdict.ConfigDict()
    # Naming convention: n<n_agents>m<min_at_lever>_<method>
    config.main.dir_name = 'n5m2_m1'
    config.main.exp_name = 'er'
    config.main.max_to_keep = 100
    config.main.model_name = 'model.ckpt'
    config.main.save_period = 100000
    config.main.seed = 12340
    config.main.summarize = False
    config.main.use_gpu = False

    config.nn_agent = configdict.ConfigDict()
    config.nn_agent.n_h1 = 64
    config.nn_agent.n_h2 = 32
    # config.nn_agent.n_h1 = 128
    # config.nn_agent.n_h2 = 64
    # Set to True for comparison with AMD
    config.nn_agent.use_single_layer = False

    config.nn_m = configdict.ConfigDict()
    config.nn_m.n_h1 = 64
    config.nn_m.n_h2 = 16
    # config.nn_m.n_h1 = 256
    # config.nn_m.n_h2 = 128

    config.nn_dual = configdict.ConfigDict()
    config.nn_dual.n_h1 = 64
    config.nn_dual.n_h2 = 32
    

    return config
