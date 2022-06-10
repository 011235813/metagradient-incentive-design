from incentive_design.utils import configdict


def get_config():

    config = configdict.ConfigDict()

    # 15x15
    config.agent = configdict.ConfigDict()
    config.agent.entropy_coeff = 0.0316
    config.agent.epsilon_div = 5000
    config.agent.epsilon_end = 0.05
    config.agent.epsilon_start = 0.5
    config.agent.gae_lambda = .98
    config.agent.gamma = 0.99
    config.agent.grad_clip = None
    config.agent.lr_actor = 1.09e-5
    config.agent.lr_v = 1.14e-5
    config.agent.minibatch_size = 20
    config.agent.ppo_epsilon = 0.0308
    config.agent.share_parameter = True
    config.agent.tau = 0.021
    
    config.alg = configdict.ConfigDict()
    # Parameters for annealing in phase 2
    config.alg.initial_max_tax_rate = 0.1
    config.alg.n_episodes = 200000
    config.alg.n_episodes_reach_max_rate = 8000
    config.alg.n_eval = 5
    config.alg.n_test = 100
    config.alg.name = 'm1'
    config.alg.objective = 'clipped_surrogate'
    config.alg.period = 100
    config.alg.phase = 2
    config.alg.restore_do_not_train_designer = False
    config.alg.resume_ep = 100
    config.alg.resume_phase2 = False
    config.alg.train_phase2_without_restoring = False  # must be False if resume_phase2=True
    config.alg.tune_agents = True
    config.alg.tune_log_uniform = True
    config.alg.update_alg = 'ppo'

    config.designer = configdict.ConfigDict()
    config.designer.cap_tax_rate = True
    config.designer.entropy_coeff = 0.4509968096552509
    config.designer.epsilon_div = 5000
    config.designer.epsilon_end = 0.05
    config.designer.epsilon_start = 0.5
    config.designer.gae_lambda = .98
    config.designer.gamma = 0.99
    config.designer.grad_clip = 10.0
    config.designer.lr_actor = 0.0005767557066758135
    config.designer.lr_v = 0.00020762245791923093
    config.designer.minibatch_size = 20
    config.designer.ppo_epsilon = 0.12582701476948835
    config.designer.tau = 0.4806449619035664

    config.designer_m = configdict.ConfigDict()
    config.designer_m.cap_tax_rate = False
    config.designer_m.gae_lambda = 0.98
    config.designer_m.gamma = 0.99
    config.designer_m.grad_clip = None
    config.designer_m.lr_incentive = 0.000203
    config.designer_m.lr_v = 1e-5
    config.designer_m.pipeline = True
    config.designer_m.ppo_epsilon = 0.0394
    config.designer_m.r_scalar = 1e-2
    config.designer_m.tau = 0.0573
    config.designer_m.use_critic = True
    config.designer_m.use_noise = False
    config.designer_m.use_ppo = True

    config.env = configdict.ConfigDict()
    config.env.components = [
        # (1) Building houses
        ('Build', {'skill_dist': "pareto", 'payment_max_skill_multiplier': 3}),
        # (2) Trading collectible resources
        ('ContinuousDoubleAuction', {'max_num_orders': 5}),
        # (3) Movement and resource collection
        ('Gather', {}),
        # Taxes set externally
        ('PeriodicBracketTax', dict(bracket_spacing="us-federal", period=10,
                                    tax_model='external', usd_scaling=1e3))
        # Dual RL
        # ('PeriodicBracketTax', dict(bracket_spacing="us-federal", period=10, usd_scaling=1e3))
        # US federal
        # ('PeriodicBracketTax', dict(bracket_spacing="us-federal", period=10,
        #                             tax_model='us-federal-single-filer-2018-scaled',
        #                             usd_scaling=1e3))
        # Free market
        # ('PeriodicBracketTax', dict(bracket_spacing="us-federal", period=10, disable_taxes=True)
    ]
    config.env.env_layout_file = 'env-pure_and_mixed-15x15.txt'
    config.env.episode_length = 100 # Number of timesteps per episode
    config.env.fixed_four_skill_and_loc = True
    config.env.flatten_masks =  True
    config.env.flatten_observations = True
    # In multi-action-mode, the policy selects an action for each action subspace (defined in component code).
    # Otherwise, the policy selects only 1 action.
    config.env.multi_action_mode_agents = False
    config.env.multi_action_mode_planner = True
    config.env.n_agents = 4 # Number of non-planner agents (must be > 1)
    config.env.scenario_name = 'layout_from_file/simple_wood_and_stone'
    config.env.starting_agent_coin = 10
    # [Height, Width] of the env world    
    config.env.world_size = [15, 15]

    config.main = configdict.ConfigDict()
    config.main.dir_name = '15x15_phase2_curr_m1'
    config.main.dir_restore = '15x15_phase1_free_market'
    config.main.exp_name = 'foundation'
    config.main.max_to_keep = 10
    config.main.meas_income_during_train = False
    config.main.meas_tax_during_train = False
    config.main.model_name = 'model.ckpt'
    config.main.model_restore = 'model.ckpt'
    config.main.save_period = 50000
    config.main.seed = 12340
    config.main.summarize = False
    config.main.use_gpu = True

    config.nn_agent = configdict.ConfigDict()
    config.nn_agent.kernel = [[5, 5], [5,5]]
    config.nn_agent.n_filters = [6, 6]
    config.nn_agent.n_fc = [128, 128]
    config.nn_agent.n_lstm = 128
    config.nn_agent.stride = [[1, 1], [1,1]]
    # For static_rnn
    config.nn_agent.max_timesteps = 100
    config.nn_agent.use_lstm_actor = False
    config.nn_agent.use_lstm_critic = True

    config.nn_designer = configdict.ConfigDict()
    config.nn_designer.kernel = [[5, 5], [5,5]]
    config.nn_designer.n_filters = [6, 6]
    config.nn_designer.n_fc = [256, 256]
    config.nn_designer.n_lstm = 128
    config.nn_designer.stride = [[1, 1], [1,1]]
    # For static_rnn
    config.nn_designer.max_timesteps = 100
    config.nn_designer.use_lstm_actor = True
    config.nn_designer.use_lstm_critic = True

    return config
