import numpy as np

"""Buffer classes."""

class BufferFoundationM2(object):
    """Trajectory buffer for m2 on Foundation."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.obs_flat = []
        self.obs_tensor = []
        self.action = []
        self.action_mask = []
        self.reward = []
        self.obs_flat_next = []
        self.obs_tensor_next = []
        self.done = []
        self.total_endowment_coin = []
        self.last_coin = []
        self.escrow_coin = []
        self.util_prev = []
        self.inventory_coin = []
        self.total_labor = []
        self.curr_rate_max = []
        self.enact_tax = []
        self.completions = []
        self.noise = []
        self.actor_state = []
        self.obs_flat_constant = []
        self.obs_tensor_constant = []

    def add(self, transition):
        self.obs_flat.append(transition[0])
        self.obs_tensor.append(transition[1])
        self.action.append(transition[2])
        self.action_mask.append(transition[3])
        self.reward.append(transition[4])
        self.obs_flat_next.append(transition[5])
        self.obs_tensor_next.append(transition[6])
        self.done.append(transition[7])
        self.noise.append(transition[8])

    def add_tax_info(self, transition):
        self.total_endowment_coin.append(transition[0])
        self.last_coin.append(transition[1])
        self.escrow_coin.append(transition[2])
        self.util_prev.append(transition[3])
        self.inventory_coin.append(transition[4])
        self.total_labor.append(transition[5])
        self.curr_rate_max.append(transition[6])
        self.enact_tax.append(transition[7])
        self.completions.append(transition[8])

    def add_lstm(self, actor_state):
        self.actor_state.append(actor_state)

    def add_constant(self, obs_flat, obs_tensor):
        self.obs_flat_constant.append(obs_flat)
        self.obs_tensor_constant.append(obs_tensor)
        

class Buffer(object):
    """An Agent's trajectory buffer."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.obs_flat = []
        self.obs_tensor = []
        self.action = []
        self.action_mask = []
        self.reward = []
        self.obs_flat_next = []
        self.obs_tensor_next = []
        self.done = []
        self.log_probs = []
        self.actor_state = []

    def add(self, transition):
        self.obs_flat.append(transition[0])
        self.obs_tensor.append(transition[1])
        self.action.append(transition[2])
        self.action_mask.append(transition[3])
        self.reward.append(transition[4])
        self.obs_flat_next.append(transition[5])
        self.obs_tensor_next.append(transition[6])
        self.done.append(transition[7])
    
    def add_ppo(self, log_prob, actor_state):
        self.log_probs.append(log_prob)
        self.actor_state.append(actor_state)
    
    def add_lstm(self, actor_state):
        self.actor_state.append(actor_state)
        
    def compute_advantages(self, values, gamma, gae_lambda=.98, reward=None):
        # values are in shape: (self.n_agents, timesteps + 1, 1)
        values = np.swapaxes(values, 0, 1)
        gae = 0
        advantages = [] 
        
        if reward is None:
            reward = self.reward
            
        for step in reversed(range(len(reward))):
            rewards_step = reward[step]
            
            if isinstance(rewards_step, float):
                rewards_step = np.array([rewards_step])
            
            rewards = np.expand_dims(rewards_step, axis=1)
            delta = rewards + gamma * values[step + 1] - values[step]
            gae = delta + gamma * gae_lambda * gae
            advantages.insert(0, gae)
        
        # Output shape: (self.n_agents, timesteps, 1)
        advantages = np.array(advantages)
        return np.swapaxes(advantages, 0, 1)
        

class Designer_SSD(object):
    """Designer's trajectory buffer for SSD."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.obs_tensor = []
        self.action_agents = []
        self.reward = []
        self.r_sample = []
        self.obs_tensor_next = []
        self.done = []

    def add(self, transition):
        self.obs_tensor.append(transition[0])
        self.action_agents.append(transition[1])
        self.reward.append(transition[2])
        self.obs_tensor_next.append(transition[3])
        self.done.append(transition[4])

    def add_r_sample(self, r):
        self.r_sample.append(r)

class Agents_SSD(object):
    """Agents' trajectory buffer for SSD."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.obs_tensor = []
        self.action = []
        self.action_type = []
        self.reward = []
        self.r_env = []
        self.obs_tensor_next = []
        self.done = []
        # For PPO
        self.log_probs = []
        self.actor_state = []

    def add(self, transition):
        self.obs_tensor.append(transition[0])
        self.action.append(transition[1])
        self.action_type.append(transition[2])
        self.reward.append(transition[3])
        self.r_env.append(transition[4])
        self.obs_tensor_next.append(transition[5])
        self.done.append(transition[6])

    def add_ppo(self, log_prob, actor_state):
        self.log_probs.append(log_prob)
        self.actor_state.append(actor_state)
