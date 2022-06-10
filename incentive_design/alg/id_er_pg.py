"""A policy gradient agent.

This implements an RL incentive designer.

In the case of a discrete-action ID, the ID's action is an integer
that gets mapped to a choice of incentive for each possible agent action.
"""
import sys

import numpy as np
import tensorflow as tf
from scipy.special import expit

from incentive_design.alg import networks
from incentive_design.utils import util


class DualRLPG(object):

    def __init__(self, agent_id, config, l_action_agent, l_obs, n_agents,
                 nn, r_multiplier=2, l_action_ID=0):
        """
        Args:
            agent_id: int
            config: ConfigDict object
            l_action_agent: int size of each agent's discrete action space
            l_obs: int size of incentive designer's observation space
            n_agents: int
            nn: ConfigDict object containing neural net params
            r_multiplier: float
            l_action_ID: int size of ID's discrete action space
                         (only used if action_space=='discrete')
        """
        self.agent_id = agent_id
        self.agent_name = 'agent_%d' % self.agent_id
        self.alg_name = 'dual_RL'
        self.l_action_agent = l_action_agent
        self.l_obs = l_obs
        self.n_agents = n_agents
        self.nn = nn
        self.r_multiplier = r_multiplier
        self.l_action_ID = l_action_ID

        self.action_space = config.action_space
        self.entropy_coeff = config.entropy_coeff
        self.gamma = config.gamma
        self.lr_actor = config.lr_actor

        self.create_networks()
        self.create_policy_gradient_op()

    def create_networks(self):
        """Creates neural network part of the TF graph."""
        self.obs = tf.placeholder(tf.float32, [None, self.l_obs], 'l_obs')
        self.actions_by_agents = tf.placeholder(
            tf.float32, [None, self.l_action_agent * self.n_agents],
            'action_by_agents')

        with tf.variable_scope(self.agent_name):
            with tf.variable_scope('eta'):
                if self.action_space == 'continuous':
                    mean = networks.incentive_mlp(
                        self.obs, self.actions_by_agents, self.nn,
                        self.n_agents, None)
                else:
                    concated = tf.concat([self.obs, self.actions_by_agents],
                                         axis=1)
                    self.probs = networks.actor_mlp(
                        concated, self.l_action_ID, self.nn)

        if self.action_space == 'continuous':
            stddev = tf.ones_like(mean)
            self.reward_dist = tf.distributions.Normal(
                loc=mean, scale=stddev)
            self.reward_sample = self.reward_dist.sample()
        else:
            self.log_probs = tf.log(self.probs + 1e-15)
            self.action_samples = tf.multinomial(self.log_probs, 1)

    def compute_incentive(self, obs, action_all, sess):
        """
        Args:
            obs: np.array of ID's observation
            action_all: list of action integers
            sess: TF session

        Returns: 
        if continuous action space: np.array of incentives, length n_agents
                                    value in [0, r_multiplier]
        if discrete action space: a single integer in [0,...,l_action_ID-1]
        """        
        actions_by_agents_1hot = util.convert_batch_action_int_to_1hot(
            action_all, self.l_action_agent).flatten()
        feed = {self.obs: np.array([obs]),
                self.actions_by_agents: np.array([actions_by_agents_1hot])}

        if self.action_space == 'continuous':
            reward_sample = sess.run(self.reward_sample, feed_dict=feed).flatten()
            reward = self.r_multiplier * expit(reward_sample)
            return reward, reward_sample
        else:
            action = sess.run(self.action_samples, feed_dict=feed)[0][0]
            return action

    def create_policy_gradient_op(self):
        """Supports either discrete or continuous action space."""
        self.returns = tf.placeholder(tf.float32, [None], 'returns')

        if self.action_space == 'continuous':
            self.r_sampled = tf.placeholder(
                tf.float32, [None, self.n_agents], 'r_sampled')
            log_probs = self.reward_dist.log_prob(self.r_sampled)
            # change of variables for the transformation:
            # x ~ Gaussian(f_eta(obs))
            # action y = r_multiplier * sigmoid(u)
            # The formula implemented here is p(y) = p(x) |det dx/dy | = p(x) |det 1/(dy/dx)|
            sigmoid_derivative = tf.math.sigmoid(self.r_sampled) * (
                    1 - tf.math.sigmoid(self.r_sampled)) * self.r_multiplier
            log_probs = (tf.reduce_sum(log_probs, axis=1) -
                         tf.reduce_sum(tf.math.log(sigmoid_derivative), axis=1))
            self.loss = -tf.reduce_sum(tf.multiply(log_probs, self.returns))
        else:
            self.action_taken = tf.placeholder(
                tf.float32, [None, self.l_action_ID], 'action_taken')
            log_probs = tf.log(tf.reduce_sum(
                tf.multiply(self.probs, self.action_taken), axis=1) + 1e-15)

            entropy = -tf.reduce_sum(tf.multiply(self.probs, self.log_probs))

            self.policy_loss = -tf.reduce_sum(
                tf.multiply(log_probs, self.returns))
            self.loss = self.policy_loss - self.entropy_coeff * entropy

        self.policy_opt = tf.train.AdamOptimizer(self.lr_actor)
        self.policy_op = self.policy_opt.minimize(self.loss)

    def train(self, sess, buf):

        n_steps = len(buf.reward)
        ones = np.ones(n_steps)
        feed = {}
        feed[self.obs] = buf.obs
        feed[self.actions_by_agents] = util.convert_batch_actions_int_to_1hot(
            buf.action_all, self.l_action_agent)
        # feed[self.ones] = ones
        feed[self.returns] = util.compute_returns(buf.reward, self.gamma)
        if self.action_space == 'continuous':
            feed[self.r_sampled] = buf.r_sample
        else:
            actions_1hot = util.convert_batch_action_int_to_1hot(
                buf.action, self.l_action_ID)
            feed[self.action_taken] = actions_1hot

        _ = sess.run(self.policy_op, feed_dict=feed)
