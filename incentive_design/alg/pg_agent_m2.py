"""A policy gradient agent.

This implementation allows an oracle incentive designer to conduct 1-step meta-gradient.
Only contains one policy, which is used to generate both the first 
and second (validation) trajectory.
"""
import sys

import numpy as np
import tensorflow as tf

from incentive_design.alg import networks
from incentive_design.alg import pg_agent_m1
from incentive_design.utils import util


class Agent(pg_agent_m1.Agent):

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
        super().__init__(agent_id, config, l_action, l_obs, n_agents,
                         nn, r_multiplier=2)

    def create_networks(self):
        """Creates neural network part of the TF graph."""
        self.obs = tf.placeholder(tf.float32, [None, self.l_obs], 'l_obs')
        self.epsilon = tf.placeholder(tf.float32, None, 'epsilon')

        with tf.variable_scope(self.agent_name):
            with tf.variable_scope('policy_main'):
                with tf.variable_scope('policy'):
                    probs = networks.actor_mlp(self.obs, self.l_action, self.nn)
                self.probs = (1 - self.epsilon) * probs + self.epsilon / self.l_action
                self.log_probs = tf.log(self.probs)
                self.action_samples = tf.multinomial(self.log_probs, 1)

        self.policy_params = tf.trainable_variables(
            self.agent_name + '/policy_main/policy')

    def create_update_op(self):
        """Set up TF graph for 1-step policy parameter update."""
        self.incentive_received = tf.placeholder(tf.float32, [None], 'incentive_received')
        r_total = self.r_env + self.incentive_received
        returns_val = tf.reverse(tf.math.cumsum(
            tf.reverse(r_total * self.gamma_prod, axis=[0])), axis=[0])
        returns_val = returns_val / self.gamma_prod

        log_probs_taken = tf.log(tf.reduce_sum(
            tf.multiply(self.probs, self.action_taken), axis=1) + 1e-15)
        entropy = -tf.reduce_sum(
            tf.multiply(self.probs, self.log_probs))
        policy_loss = -tf.reduce_sum(
            tf.multiply(log_probs_taken, returns_val))
        loss = policy_loss - self.entropy_coeff * entropy

        policy_opt = tf.train.GradientDescentOptimizer(self.lr_actor)
        self.policy_op = policy_opt.minimize(loss)

    def update(self, sess, buf, epsilon):
        """Runs 1-step policy parameter update."""

        n_steps = len(buf.obs)
        actions_1hot = util.convert_batch_action_int_to_1hot(buf.action, self.l_action)
        ones = np.ones(n_steps)
        feed = {self.obs: buf.obs,
                self.action_taken: actions_1hot,
                self.r_env: buf.r_env,
                self.ones: ones,
                self.epsilon: epsilon}

        feed[self.incentive_received] = buf.incentive_received

        _ = sess.run(self.policy_op, feed_dict=feed)
