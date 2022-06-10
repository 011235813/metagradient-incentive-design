from incentive_design.utils import configdict


def get_config():

    config = configdict.ConfigDict()

    config.agent = configdict.ConfigDict()
    config.agent.entropy_coeff = 0.129
    config.agent.epsilon_div = 100
    config.agent.epsilon_end = 0.05
    config.agent.epsilon_start = 1.0
    config.agent.gamma = 0.99
    config.agent.grad_clip = None
    config.agent.lr_actor = 6.73e-4
    config.agent.lr_v = 5.54e-4
    config.agent.share_parameter = True
    config.agent.tau = 7.93e-3

    config.alg = configdict.ConfigDict()
    config.alg.n_episodes = 100000
    config.alg.n_eval = 5
    config.alg.n_test = 100
    # Options: 'dual_RL', 'm1'
    config.alg.name = 'm1'
    config.alg.period = 100
    config.alg.resume = False
    config.alg.resume_ep = 80000
    config.alg.restore_do_not_train_designer = False
    config.alg.tune_agents = True
    config.alg.tune_log_uniform = True
    config.alg.update_alg = 'ac'

    # dual RL designer
    config.designer = configdict.ConfigDict()
    config.designer.entropy_coeff = 0.230
    config.designer.epsilon_div = 5000
    config.designer.epsilon_end = 0
    config.designer.epsilon_start = 0
    config.designer.gae_lambda = 0.99
    config.designer.gamma = 0.99
    config.designer.grad_clip = None
    config.designer.lr_actor = 1.15e-5
    config.designer.lr_v = 1.72e-5
    config.designer.ppo_epsilon = 0.0164
    config.designer.tau = 0.846

    config.designer_m = configdict.ConfigDict()
    config.designer_m.gae_lambda = 0.99
    config.designer_m.gamma = 0.99
    config.designer_m.grad_clip = None
    config.designer_m.lr_incentive = 1.24e-5
    config.designer_m.lr_v = 2.4e-5
    config.designer_m.pipeline = True
    config.designer_m.ppo_epsilon = 0.0172
    config.designer_m.tau = 0.114
    config.designer_m.use_critic = True
    config.designer_m.use_ppo = True

    config.env = configdict.ConfigDict()
    config.env.beam_width = 3  # default 3
    config.env.cleaning_penalty = 0.0
    config.env.disable_left_right_action = False
    config.env.disable_rotation_action = True
    # if not None, a fixed global reference frame is used for all agents

    config.env.global_ref_point = None  # agents do not use global ref
    config.env.map_name = 'cleanup_small_sym'
    config.env.max_steps = 50 # small: 50 | 10x10: 50 | 15x15: 50
    config.env.n_agents = 2 # max small 2 | 10x10_sym 3 | 10x10_n5 5 | 15x15: 5
    config.env.n_action_types = 3 # clean, get apple, else
    # If T, reward function takes in 1-hot representation of
    # whether the other agent used the cleaning beam
    # Else, observe the full 1-hot action of other agent
    config.env.obs_cleaned_1hot = True
    # ---------- for 7x7 map cleanup_small_sym ------------
    config.env.obs_height = 9
    config.env.obs_width = 9
    config.env.planner_ref_point = [3, 3]  # for cleanup_small
    config.env.r_multiplier = 2.0  # scale up sigmoid output
    config.env.random_orientation = False
    config.env.shuffle_spawn = False
    # 0.5(height - 1)
    config.env.view_size = 4  # cleanup_small
    config.env.cleanup_params = configdict.ConfigDict()
    config.env.cleanup_params.appleRespawnProbability = 0.5
    config.env.cleanup_params.thresholdDepletion = 0.6
    config.env.cleanup_params.thresholdRestoration = 0.0
    config.env.cleanup_params.wasteSpawnProbability = 0.5

    config.main = configdict.ConfigDict()
    config.main.dir_name = '7x7_m1'
    config.main.dir_restore = ''
    config.main.exp_name = 'ssd'
    config.main.max_to_keep = 10
    config.main.model_name = 'model.ckpt'
    config.main.model_restore = 'model.ckpt'
    config.main.save_period = 10000
    config.main.seed = 12340
    config.main.summarize = False
    config.main.use_gpu = True

    config.nn = configdict.ConfigDict()
    config.nn.kernel = [3, 3]
    config.nn.n_filters = 6
    # config.nn.n_h = 128
    config.nn.n_h1 = 64
    config.nn.n_h2 = 64
    config.nn.stride = [1, 1]

    return config
