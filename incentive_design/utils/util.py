import numpy as np

def convert_batch_action_int_to_1hot(actions, l_action):
    """Converts a batch of integer actions to 1-hot.

    Args:
        actions: list of shape [batch_size]
        l_action: int size of action space
    
    Returns: np.array of shape [batch_size, l_action]
    """
    n_steps = len(actions)
    actions_1hot = np.zeros([n_steps, l_action], dtype=int)
    actions_1hot[np.arange(n_steps), actions] = 1

    return actions_1hot


def convert_batch_actions_int_to_1hot(list_action_all, l_action):
    """
    Args:
        list_action_all: list over time steps of list of agents' integer actions
        l_action: int size of action space

    Returns: np.array [n_steps, l_action*n_agents]
    """
    n_steps = len(list_action_all)
    n_agents = len(list_action_all[0])
    matrix = np.stack(list_action_all)  # [n_steps, n_agents]
    actions_1hot = np.zeros([n_steps, n_agents, l_action], dtype=np.float32)
    grid = np.indices((n_steps, n_agents))
    actions_1hot[grid[0], grid[1], matrix] = 1
    actions_1hot = np.reshape(actions_1hot, [n_steps, l_action*n_agents])

    return actions_1hot


def compute_returns(rewards, gamma):
    n_steps = len(rewards)
    gamma_prod = np.cumprod(np.ones(n_steps) * gamma)
    returns = np.cumsum((rewards * gamma_prod)[::-1])[::-1]
    returns = returns / gamma_prod

    return returns


def compute_returns_batch_time(rewards, gamma):
    """
    rewards: np.array shape [batch, time]

    returns np.array shape [batch, time]
    """
    n_batch, n_steps = rewards.shape
    ones = np.ones((n_batch, n_steps))
    gamma_prod = np.cumprod(ones * gamma, axis=1)
    returns = np.cumsum(
        (rewards * gamma_prod)[:, ::-1], axis=1)[:, ::-1]
    returns = returns / gamma_prod

    return returns


def compute_advantages(reward, values, gamma, gae_lambda=0.98):
    """
    Args:
        values: np.array [T+1]
        reward: np.array [T]

    Returns: np.array [T]
    """
    gae = 0
    advantages = []
    for step in reversed(range(len(reward))):
        delta = reward[step] + gamma * values[step + 1] - values[step]
        gae = delta + gamma * gae_lambda * gae
        advantages.insert(0, gae)

    return advantages


def compute_advantages_batch(values, rewards, gamma, gae_lambda):
    """
    Args:
    values: [t+1, batch]
    rewards: [t, batch]

    Returns: [t, batch]]
    """
    gae = 0
    advantages = []

    for step in reversed(range(len(rewards))):
        rewards_step = rewards[step] # [B]

        delta = (rewards_step + gamma * values[step + 1]
                 - values[step])
        gae = delta + gamma * gae_lambda * gae
        advantages.insert(0, gae)

    advantages = np.array(advantages)

    return advantages


class OU:

    def __init__(self, dim, x0=1., theta=1., mu=1., sigma=.3, dt=.5):
        self.dim = dim
        self.x = np.array([x0 for i in range(self.dim)])
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
    
    def step(self, use_noise):
        if not use_noise:
            return np.zeros(self.dim)
        
        self.x += self.theta * (self.mu - self.x) * self.dt + self.sigma * np.random.normal(0, self.dt, size=(self.dim))
        return self.x
