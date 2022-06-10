"""Algorithm(s) for the top-level incentive designer."""
import sys

import numpy as np
import tensorflow as tf

import networks

from incentive_design.utils import util


class MetaGrad1Step(object):

    def __init__(self, agent_id, config, l_action, l_obs, n_agents,
                 nn, r_multiplier):
        """
        Args:
            agent_id: int
            config: ConfigDict object
            l_action: int size of each agent's discrete action space
            l_obs: int size of incentive designer's observation space
            n_agents: int
            nn: ConfigDict object containing neural net params
            r_multiplier: float
        """
        self.agent_id = agent_id
        self.agent_name = 'agent_%d' % self.agent_id
        self.alg_name = 'meta-1step'
        self.l_action = l_action
        self.l_obs = l_obs
        self.n_agents = n_agents
        self.nn = nn
        self.r_multiplier = r_multiplier

        self.gamma = config.gamma
        self.lr_cost = config.lr_cost
        self.lr_incentive = config.lr_incentive
        if 'optimizer' in config:
            self.optimizer = config.optimizer
        else:
            self.optimizer = 'sgd'
        self.output_type = config.output_type
        self.reg_coeff = config.reg_coeff
        self.separate_cost_optimizer = config.separate_cost_optimizer

        self.agent_spec = config.agent_spec
        self.lr_spec = config.lr_spec
        self.spec_coeff = config.spec_coeff 

        self.create_networks()
        
    def create_networks(self):
        """Creates neural network part of the TF graph."""
        self.obs = tf.placeholder(tf.float32, [None, self.l_obs], 'l_obs')
        self.actions_by_agents = tf.placeholder(
            tf.float32, [None, self.l_action * self.n_agents])

        n_outputs = (self.n_agents if self.output_type=='agent'
                     else self.l_action)
        with tf.variable_scope(self.agent_name):
            with tf.variable_scope('eta'):
                self.incentive_function = networks.incentive_mlp(
                    self.obs, self.actions_by_agents,
                    self.nn, n_outputs=n_outputs)

        self.eta_vars = tf.trainable_variables(self.agent_name + '/eta')

    def receive_list_of_agents(self, list_of_agents):
        self.list_of_agents = list_of_agents

    def compute_incentive(self, obs, action_all, sess):
        """
        Args:
            obs: np.array of ID's observation
            action_all: list of action integers
            sess: TF session
        Returns: np.array of incentives
        """
        # [n_agents * l_action]
        actions_by_agents_1hot = util.convert_batch_action_int_to_1hot(
            action_all, self.l_action).flatten()
        feed = {self.obs: np.array([obs]),
                self.actions_by_agents: np.array([actions_by_agents_1hot])}
        incentive = sess.run(self.incentive_function, feed_dict=feed)
        incentive = incentive.flatten() * self.r_multiplier

        return incentive

    def create_incentive_train_op(self):
        """Set up TF graph for training the incentive function."""
        list_incentive_loss = []
        self.list_policy_new = [0 for _ in range(self.n_agents)]
        # return of the incentive designer on the valdition trajectory
        self.returns = tf.placeholder(tf.float32, [None], 'returns')

        for agent in self.list_of_agents:
            policy_params_new = {}
            # \hat{\theta} <-- \theta + \Delta \theta
            for grad, var in zip(agent.policy_grads, agent.policy_params):
                policy_params_new[var.name] = var - agent.lr_actor * grad
            # Store into dummy model to preserve dependence on \eta
            policy_new = agent.policy_new(
                policy_params_new, agent.l_obs, agent.l_action,
                agent.agent_name)
            self.list_policy_new[agent.agent_id-1] = policy_new

            log_probs_taken = tf.log(
                tf.reduce_sum(tf.multiply(policy_new.probs,
                                          policy_new.action_taken), axis=1))
            loss_term = -tf.reduce_sum(tf.multiply(log_probs_taken, self.returns))
            list_incentive_loss.append(loss_term)

        self.ones = tf.placeholder(tf.float32, [None], 'ones')
        self.gamma_prod = tf.math.cumprod(self.ones * self.gamma)
        given_each_step = tf.reduce_sum(tf.abs(self.incentive_function), axis=1)
        total_given = tf.reduce_sum(tf.multiply(
            given_each_step, self.gamma_prod/self.gamma))
        if self.separate_cost_optimizer:
            self.incentive_loss = tf.reduce_sum(list_incentive_loss)
        else:
            self.incentive_loss = (tf.reduce_sum(list_incentive_loss) +
                                self.reg_coeff * total_given)

        if self.agent_spec:
            self.incentive_loss += self.spec_coeff * self.create_agent_specialization_op() 

        if self.optimizer == 'sgd':
            incentive_opt = tf.train.GradientDescentOptimizer(self.lr_incentive)
            if self.separate_cost_optimizer:
                cost_opt = tf.train.GradientDescentOptimizer(self.lr_cost)
        elif self.optimizer == 'adam':
            incentive_opt = tf.train.AdamOptimizer(self.lr_incentive)
            if self.separate_cost_optimizer:
                cost_opt = tf.train.AdamOptimizer(self.lr_cost)
        self.incentive_op = incentive_opt.minimize(
            self.incentive_loss, var_list=self.eta_vars)
        if self.separate_cost_optimizer:
            self.cost_op = cost_opt.minimize(
                total_given, var_list=self.eta_vars)

    def create_agent_specialization_op(self):
        individual_agent_loss = []
        for policy_new in self.list_policy_new:
            log_probs_taken = tf.log(
                tf.reduce_sum(tf.multiply(policy_new.probs,
                                          policy_new.action_taken), axis=1))
            loss_term = tf.reduce_sum(log_probs_taken)
            individual_agent_loss.append(loss_term)

        # Calculate the sum over timestep of log_probs for the joint policy
        joint_policy_loss = tf.reduce_sum(individual_agent_loss)

        # Calculate pairwise specialization reward
        specialization_reward = []
        for i in range(len(self.list_policy_new)):
            for j in range(i + 1, len(self.list_policy_new)):
                obs_i = self.list_policy_new[i].obs
                action_i = self.list_policy_new[i].action_taken

                obs_j = self.list_policy_new[j].obs
                action_j = self.list_policy_new[j].action_taken

                obs_distance = tf.norm(obs_i - obs_j, ord=2, axis=-1)
                action_distance = tf.norm(action_i - action_j, ord=2, axis=-1)

                loss_term = tf.reduce_sum(tf.tanh(tf.multiply(obs_distance, action_distance)))
                """
                agent_distance = tf.multiply(obs_distance, action_distance)
                agent_distance = tf.cond(tf.count_nonzero(tf.reduce_sum(agent_distance)) <= 0, lambda: tf.constant(0.01), lambda: agent_distance)
                loss_term = tf.reduce_sum(-tf.reciprocal(agent_distance))
                loss_term = tf.cond(tf.is_nan(loss_term), lambda: loss_term, lambda: tf.constant(-1 / .01))
                """
                specialization_reward.append(loss_term)
        
        specialization_loss = tf.reduce_sum(specialization_reward)
        
        return -tf.multiply(joint_policy_loss, specialization_loss)

    def train_incentive(self, sess, list_buf, list_buf_new, epsilon):
        """1-step of incentive function.

        Args:
            sess: TF session
            list_buf: list of agents' trajectory buffers (using \theta)
            list_buf_new: list of agents' new trajectory buffers (using \hat{\theta})
            epsilon: float exploration parameter
        """
        # 1st and 2nd trajectories experienced by the incentive designer
        buf_self = list_buf[self.agent_id]
        buf_self_new = list_buf_new[self.agent_id]

        n_steps = len(buf_self.obs)
        ones = np.ones(n_steps)

        feed = {}
        for agent in self.list_of_agents:
            agent_id = agent.agent_id
            buf_agent = list_buf[agent_id]
            actions_1hot = util.convert_batch_action_int_to_1hot(
                buf_agent.action, self.l_action)
            feed[agent.obs] = buf_agent.obs
            feed[agent.action_taken] = actions_1hot
            feed[agent.r_env] = buf_agent.r_env
            feed[agent.epsilon] = epsilon

            if agent.alg_name == 'ac':
                v_next = np.reshape(sess.run(
                    agent.v, feed_dict={agent.obs: buf_agent.obs_next}), [n_steps])
                v = np.reshape(sess.run(
                    agent.v, feed_dict={agent.obs: buf_agent.obs}), [n_steps])
                feed[agent.v_ph] = v
                feed[agent.v_next_ph] = v_next
                feed[agent.done] = buf_agent.done
            else:
                feed[agent.ones] = ones

            buf_agent_new = list_buf_new[agent_id]
            actions_1hot_new = util.convert_batch_action_int_to_1hot(
                buf_agent_new.action, self.l_action)
            policy_new = self.list_policy_new[agent_id-1] # agent_id goes from 1 to n
            feed[policy_new.obs] = buf_agent_new.obs
            feed[policy_new.action_taken] = actions_1hot_new

        n_steps = len(buf_self_new.obs)
        reward_id = buf_self_new.reward
        returns_id = util.compute_returns(reward_id, self.gamma)
        feed[self.obs] = buf_self.obs
        feed[self.actions_by_agents] = util.convert_batch_actions_int_to_1hot(
            buf_self.action_all, self.l_action)
        feed[self.ones] = ones
        feed[self.returns] = returns_id

        if self.separate_cost_optimizer:
            _ = sess.run([self.incentive_op, self.cost_op], feed_dict=feed)
        else:
            _ = sess.run(self.incentive_op, feed_dict=feed)
