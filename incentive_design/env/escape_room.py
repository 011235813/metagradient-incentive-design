"""Escape room game ER(N,M).

N = number of agents
M = number of agents who must be at the lever for door to be open

Agents can be at positions 0 (lever), 1 (start), and 2 (door).
"""
import numpy as np
from incentive_design.env import room_agent


class Env(object):

    def __init__(self, config_env):

        self.config = config_env

        self.n_agents = self.config.n_agents  # N
        self.name = 'er'
        self.l_action = 3
        # Observe self position (1-hot) and
        # other agents' positions (1-hot for each other agent)
        self.l_obs = 3 + 3*(self.n_agents - 1)

        self.list_discrete_incentives = self.config.list_discrete_incentives
        self.n_discrete_incentives = len(self.list_discrete_incentives)
        self.l_action_discrete_dualRL = self.n_discrete_incentives**self.l_action

        self.max_steps = self.config.max_steps
        self.min_at_lever = self.config.min_at_lever  # M
        self.randomize = self.config.randomize

        self.actors = [room_agent.Actor(idx, self.n_agents, self.l_obs)
                       for idx in range(self.n_agents)]

    def get_door_status(self, actions):
        n_going_to_lever = actions.count(0)
        return n_going_to_lever >= self.min_at_lever

    def calc_reward(self, actions, door_open):
        """
        Args:
            actions: list of integer actions

        Returns: np.array of rewards for ID and agents
        """
        assert len(actions) == self.n_agents
        rewards = np.zeros(1+self.n_agents)

        for agent_id in range(0, self.n_agents):
            if door_open and actions[agent_id] == 2:
                rewards[1+agent_id] = 10
            elif actions[agent_id] == self.actors[agent_id].position:
                # no penalty for staying at current position
                rewards[1+agent_id] = 0
            else:
                rewards[1+agent_id] = -1

        # ID's reward is the sum of agents' rewards
        rewards[0] = np.sum(rewards[1:])

        return rewards

    def get_obs(self):
        list_obs = []
        central_obs = np.zeros(3*self.n_agents)
        for idx, actor in enumerate(self.actors):
            list_obs.append(actor.get_obs(self.state))
            # populate central observation
            central_obs[3*idx + self.state[idx]] = 1

        # Insert central observation at index 0
        list_obs.insert(0, central_obs)

        return list_obs

    def step(self, actions):
        """Takes 1 step in the environment.

        Args:
            actions: list of integer actions

        Returns: list of next obs, list of rewards, done indicator
        """
        door_open = self.get_door_status(actions)
        rewards = self.calc_reward(actions, door_open)
        for idx, actor in enumerate(self.actors):
            actor.act(actions[idx])
        self.steps += 1
        self.state = [actor.position for actor in self.actors]
        list_obs_next = self.get_obs()

        # Terminate if (door is open and some agent ended up at door)
        # or reach max_steps
        done = (door_open and 2 in self.state) or self.steps == self.max_steps

        return list_obs_next, rewards, done

    def reset(self):
        for actor in self.actors:
            actor.reset(self.randomize)
        self.state = [actor.position for actor in self.actors]
        self.steps = 0
        list_obs = self.get_obs()

        return list_obs

    def map_discrete_action_to_incentives(self, action_designer,
                                          action_agents):
        """Maps an action integer to vector of incentive values.

        Action space is d^{l_action}, where
        d = n_discrete_incentives
        Example: d = 5, l = 3, discretization given above
        then there are 5^3 choices

        Args:
        action_designer: int ID's action
        action_agents: list of ints, actions taken by agents

        Returns: np.array of incentive values, one per agent
        """
        incentives = np.zeros(self.n_agents)
        incentive_for_action = [0] * self.l_action
        # Decode action_designer to incentive for each possible agent action
        for i in range(self.l_action):
            idx = action_designer % self.n_discrete_incentives
            incentive_for_action[i] = self.list_discrete_incentives[idx]
            action_designer = action_designer // self.n_discrete_incentives
            
        for agent in range(self.n_agents):
            incentives[agent] = incentive_for_action[action_agents[agent]]

        return incentives
