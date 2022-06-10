import numpy as np

class Actor(object):

    def __init__(self, agent_id, n_agents, l_obs):

        self.agent_id = agent_id
        self.l_obs = l_obs
        self.n_agents = n_agents
        self.position = 1

    def act(self, action):

        self.position = action

    def get_obs(self, state):
        obs = np.zeros(self.l_obs)
        # position of self
        obs[state[self.agent_id]] = 1
        list_others = list(range(0, self.n_agents))
        del list_others[self.agent_id]
        # positions of other agents
        for idx, other_id in enumerate(list_others):
            obs[3*(idx + 1) + state[other_id]] = 1

        return obs

    def reset(self, randomize=False):
        if randomize:
            self.position = np.random.randint(3)
        else:
            self.position = 1  # 'start' state
