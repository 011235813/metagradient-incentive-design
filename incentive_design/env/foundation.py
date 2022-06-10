"""Wrapper around the Foundation environment.

Interfaces with the metagradient algorithm.
"""

from ai_economist import foundation
import json
import numpy as np


class Env(object):

    def __init__(self, config):

        self.env = foundation.make_env_instance(**config)
        self.periodic_bracket_tax = self.env._components_dict.get('PeriodicBracketTax')
        self.n_agents = self.env.n_agents

        # Assumes that all agents have the same action space, and
        # that multi_action_mode_agents is false.
        # Make it a list of size 1, as we use a general policy
        # that handles multi-mode action spaces.
        self.agent_action_space = [self.env.get_agent('0').action_spaces]
        # This is a list. If env_config contains 'PeriodicBracketTax'
        # component, then length of list is greater than 1.
        self.p_action_spaces = self.env.get_agent('p').action_spaces

        # Perform a reset to get observation dimensions
        obs = self.reset()
        self.obs_dim_agent_t = obs['0']['tensor'].shape
        self.obs_dim_agent_flat = obs['0']['flat'].shape
        self.obs_dim_p_t = obs['p']['tensor'].shape
        self.obs_dim_p_flat = obs['p']['flat'].shape

    def reset(self):
        """Resets the environment and returns observation dict."""

        obs = self.env.reset()
        obs = self.reformat_obs(obs)

        return obs

    def reformat_obs(self, obs):
        """Combines observation channels.

        obs is a dict with keys '0',...,'n_agents-1', 'p'
        Each obs[key] is another dict with the following keys:
        agents have 'world-map', 'world-idx-map', 'time', 'flat', 
                    'action_mask'
        planner has 'world-map', 'world-idx-map', 'time', 'flat', 
                    'p0', ..., 'p<n-1>', 'action_mask'
        
        For agents, simply combine 'world-map' and 'world-idx-map' into
                    a 'tensor' key
        For planner, combine 'world-map' and 'world-idx-map', also
        combine 'flat' with the 'p?' into a new 'flat'.
        Then agents and planner only have 'tensor', 'flat', and 'action_mask'
        """
        obs_new = {}
        for idx in range(self.env.n_agents):
            d_orig = obs[str(idx)]
            d_new = {}
            # channel is first dim
            concat = np.concatenate(
                [d_orig['world-map'], d_orig['world-idx_map']], axis=0)
            d_new['tensor'] = np.swapaxes(concat, 0, 2) # channel last
            d_new['flat'] = d_orig['flat']
            d_new['action_mask'] = d_orig['action_mask']
            obs_new[str(idx)] = d_new

        d_orig = obs['p']
        d_new = {}
        concated = np.concatenate(
            [d_orig['world-map'], d_orig['world-idx_map']], axis=0)
        d_new['tensor'] = np.swapaxes(concat, 0, 2)
        d_new['flat'] = np.concatenate(
            [d_orig['p'+str(idx)] for idx in range(self.env.n_agents)])
        d_new['action_mask'] = d_orig['action_mask']
        obs_new['p'] = d_new

        return obs_new

    def step(self, actions):
        """One time step of the environment

        actions: dict with keys {'0',...'<n_agents>-1', 'p'}
                 If multi_action_mode_agent/planner is True,
                 each value is a list of integers, otherwise
                 value is an int.

        Returns: obs dict of (agent, dict)
                 rew dict of (agent, float)
                 done Bool
                 info dict of (agent, dict)
        """
        # Ensure that actions is a dictionary with keys
        # {'0', '1',...,'p'} and type of value is either int or
        # list of ints (for designer with multi-action mode)
        actions = dict(actions)  # copy to a new object
        for k, v in actions.items():
            if k == 'p' and type(v) == list:
                pass  # assume that list is properly formatted
            elif k == 'p' and type(v) == np.ndarray and len(v.shape) >= 1:
                # convert to flat list
                actions[k] = list(np.squeeze(v))
            else:
                actions[k] = int(v)

        # Note that action masks are applied inside the policy networks
        obs, rew, done, info = self.env.step(actions)
    
        # Obs is a dict with keys '0','1',...,'n_agents-1', 'p'
        # where 'p' is the incentive designer
        # obs[key] is another dictionary
        obs = self.reformat_obs(obs)

        return obs, rew, done['__all__'], info

    def get_agent(self, i):

        return self.env.get_agent(i)
