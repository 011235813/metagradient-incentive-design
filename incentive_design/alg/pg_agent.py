"""A policy gradient agent."""

import sys

import numpy as np
import tensorflow as tf

from incentive_design.alg import networks
from incentive_design.utils import util


class Agent(object):

    def __init__(self, agent_id, config, l_action, l_obs, n_agents,
                 nn, r_multiplier=2):
        """
        Args:
            agent_id: int
            config: ConfigDict object
            l_action: int
            l_obs: int
            n_agents: int
            nn: ConfigDict object containing neural net params
            r_multiplier: float
        """
        self.agent_id = agent_id
        self.agent_name = 'agent_%d' % self.agent_id
        self.alg_name = 'pg'
        self.l_action = l_action
        self.l_obs = l_obs
        self.n_agents = n_agents
        self.nn = nn
        self.r_multiplier = r_multiplier

        self.entropy_coeff = config.entropy_coeff
        self.gamma = config.gamma
        self.lr_actor = config.lr_actor

        self.create_networks()
        self.create_policy_gradient_op()

    def create_networks(self):
        """Creates neural network part of the TF graph."""
        self.obs = tf.placeholder(tf.float32, [None, self.l_obs], 'l_obs')
        self.epsilon = tf.placeholder(tf.float32, None, 'epsilon')

        with tf.variable_scope(self.agent_name):
            with tf.variable_scope('policy_main'):
                with tf.variable_scope('policy'):
                    probs = networks.actor_mlp(self.obs, self.l_action, self.nn)
                self.probs = (1 - self.epsilon) * probs + self.epsilon / self.l_action
                self.log_probs = tf.log(self.probs + 1e-15)
                self.action_samples = tf.multinomial(self.log_probs, 1)

        self.policy_params = tf.trainable_variables(
            self.agent_name + '/policy_main/policy')

    def run_actor(self, obs, sess, epsilon):
        
        feed = {self.obs: np.array([obs]), self.epsilon: epsilon}
        action = sess.run(self.action_samples, feed_dict=feed)[0][0]

        return action

    def create_policy_gradient_op(self):

        # Reward defined by the environment
        self.returns = tf.placeholder(tf.float32, [None], 'returns')
        self.action_taken = tf.placeholder(tf.float32, [None, self.l_action],
                                           'action_taken')
        self.log_probs_taken = tf.log(tf.reduce_sum(
            tf.multiply(self.probs, self.action_taken), axis=1) + 1e-15)

        self.entropy = -tf.reduce_sum(tf.multiply(self.probs, self.log_probs))

        self.policy_loss = -tf.reduce_sum(
            tf.multiply(self.log_probs_taken, self.returns))
        self.loss = self.policy_loss - self.entropy_coeff * self.entropy

        self.policy_grads = tf.gradients(self.loss, self.policy_params)
        grads_and_vars = list(zip(self.policy_grads, self.policy_params))
        self.policy_opt = tf.train.GradientDescentOptimizer(self.lr_actor)
        self.policy_op = self.policy_opt.apply_gradients(grads_and_vars)

    def train(self, sess, buf, epsilon):

        n_steps = len(buf.obs)
        actions_1hot = util.convert_batch_action_int_to_1hot(buf.action, self.l_action)
        # ones = np.ones(n_steps)
        r_total = np.array(buf.r_env) + np.array(buf.incentive_received)
        returns = util.compute_returns(r_total, self.gamma)
        feed = {self.obs: buf.obs,
                self.action_taken: actions_1hot,
                self.returns: returns,
                # self.ones: ones,
                self.epsilon: epsilon}

        _ = sess.run(self.policy_op, feed_dict=feed)
