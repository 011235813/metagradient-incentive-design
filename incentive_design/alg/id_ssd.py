"""Incentive designer with PPO-based metagradient loss function."""

import sys

import numpy as np
import tensorflow as tf

from incentive_design.alg import actor_critic_ssd_m1
from incentive_design.alg import networks
from incentive_design.utils import util


class MetaGrad1Step(object):

    def __init__(self, designer_id, agent_name, config, dim_action,
                 dim_obs_tensor, n_agents, nn, dim_action_agents=2,
                 r_multiplier=2.0):
        """Initialization.

        Args:
            designer_id: int
            agent_name: str
            config: ConfigDict
            dim_action: int, number of output nodes of incentive function
            dim_obs_tensor: list, if obs has an image part, else None
            n_agents: int
            nn: ConfigDict
            dim_action_agents: int, size of each agent's 1-hot action
                               input to incentive function
            r_multiplier: float
        """
        self.designer_id = designer_id
        self.agent_name = agent_name
        self.dim_action = dim_action
        self.dim_obs_tensor = dim_obs_tensor
        self.n_agents = n_agents
        self.nn = nn
        self.dim_action_agents = dim_action_agents
        self.r_multiplier = r_multiplier

        self.gae_lambda = config.gae_lambda
        self.gamma = config.gamma
        self.grad_clip = config.grad_clip
        self.lr_incentive = config.lr_incentive
        self.lr_v = config.lr_v
        self.ppo_epsilon = config.ppo_epsilon
        self.tau = config.tau
        self.use_critic = config.use_critic

        self.create_networks()

    def create_networks(self):
        """Creates neural network part of the TF graph."""
        self.obs_tensor = tf.placeholder(
            tf.float32, [None]+list(self.dim_obs_tensor),'obs_tensor')
        self.action_agents = tf.placeholder(
            tf.float32, [None, self.dim_action_agents * self.n_agents])

        incentive_function = networks.incentive_ssd
        value_net = networks.vnet_ssd

        with tf.variable_scope(self.agent_name):
            with tf.variable_scope('eta'):
                self.incentive_function = incentive_function(
                    self.obs_tensor, self.action_agents,
                    self.nn, self.dim_action)

            with tf.variable_scope('v_main'):
                self.v = value_net(self.obs_tensor, self.nn)
            with tf.variable_scope('v_target'):
                self.v_target = value_net(self.obs_tensor, self.nn)
        self.eta_vars = tf.trainable_variables(self.agent_name + '/eta')

        self.v_params = tf.trainable_variables(
            self.agent_name + '/v_main')
        self.v_target_params = tf.trainable_variables(
            self.agent_name + '/v_target')
        self.list_initialize_v_ops = []
        self.list_update_v_ops = []
        for idx, var in enumerate(self.v_target_params):
            # target <- main
            self.list_initialize_v_ops.append(
                var.assign(self.v_params[idx]))
            # target <- tau * main + (1-tau) * target
            self.list_update_v_ops.append(
                var.assign(self.tau*self.v_params[idx] + (1-self.tau)*var))

    def receive_agents(self, agents):
        """Oracle access to agent parameters

        agents: an agent class with parameter-sharing
        """
        assert isinstance(agents, actor_critic_ssd_m1.ActorCritic)
        self.agents = agents

    def compute_incentive(self, obs_tensor, action_agents, sess):
        """Computes incentive for the current time step.

        Args:
            obs: np.array of ID's observation
            action_agents: list of ints
            sess: TF session

        Returns: np.array of incentive values for each action type
        """
        action_agents_1hot = util.convert_batch_actions_int_to_1hot(
            [action_agents], self.dim_action_agents)
        feed = {self.obs_tensor: np.array([obs_tensor]),
                self.action_agents: action_agents_1hot}
        incentive = sess.run(self.incentive_function, feed_dict=feed)
        incentive = incentive.flatten()

        return incentive

    def create_critic_train_op(self):
        self.v_target_next = tf.placeholder(tf.float32, [None], 'v_target_next')
        self.reward = tf.placeholder(tf.float32, [None], 'reward')
        td_target = self.reward + self.gamma * self.v_target_next
        self.loss_v = tf.reduce_mean(tf.square(
            td_target - tf.squeeze(self.v)))
        self.v_opt = tf.train.AdamOptimizer(self.lr_v)
        self.v_op = self.v_opt.minimize(self.loss_v)

    def create_incentive_train_op(self):
        """Set up TF graph for metagradient."""
        
        # [time*n_agents]
        self.advantages = tf.placeholder(tf.float32, [None], 'advantages')
        
        # Assumes parameter sharing at the agents' level
        policy_params_new = {}
        # \hat{\theta} <-- \theta + \Delta \theta
        for grad, var in zip(self.agents.policy_grads,
                             self.agents.policy_params):
            policy_params_new[var.name] = var - self.agents.lr_actor * grad
        
        # Store into dummy model to preserve dependence on \eta
        self.policy_new = self.agents.policy_new(
            policy_params_new, self.agents.dim_obs_tensor,
            self.agents.dim_action, self.agents.agent_name)

        # policy_new.probs has shape [time*n_agents, dim_action]
        # shape: [time*n_agents] due to parameter-sharing
        log_probs_taken = tf.log(tf.reduce_sum(
            tf.multiply(self.policy_new.probs,
                        self.policy_new.action_taken), axis=1) + 1e-15)

        # shape: [time]
        sum_agents_log_pi = tf.reduce_sum(
            tf.reshape(log_probs_taken, [-1, self.n_agents]), axis=1)

        # [time]
        sum_agents_log_pi_old = tf.stop_gradient(sum_agents_log_pi)
        
        ratio = tf.exp(sum_agents_log_pi - sum_agents_log_pi_old)

        surr_1 = tf.multiply(ratio, self.advantages)
        surr_2 = tf.multiply(tf.clip_by_value(
            ratio, 1.0-self.ppo_epsilon, 1.0+self.ppo_epsilon),
                             self.advantages)
        self.incentive_loss = - tf.reduce_mean(tf.minimum(surr_1, surr_2))

        incentive_opt = tf.train.AdamOptimizer(self.lr_incentive)
        if self.grad_clip:
            grad_var = incentive_opt.compute_gradients(
                self.incentive_loss, var_list=self.eta_vars)
            grad_var = [(tf.clip_by_norm(tup[0], self.grad_clip) if (not tup[0] is None) else tup[0], tup[1]) for tup in grad_var]
            self.incentive_op = incentive_opt.apply_gradients(grad_var)
        else:
            self.incentive_op = incentive_opt.minimize(
                self.incentive_loss, var_list=self.eta_vars)
            self.grad_var = incentive_opt.compute_gradients(
                self.incentive_loss, var_list=self.eta_vars)

    def train_critic(self, sess, buf):
        """Update value network."""
        batch_size = len(buf.reward)

        feed = {self.obs_tensor: buf.obs_tensor_next}
        v_target_next = sess.run(self.v_target, feed_dict=feed)
        v_target_next = np.reshape(v_target_next, [batch_size])

        feed = {self.obs_tensor: buf.obs_tensor,
                self.v_target_next: v_target_next,
                self.reward: buf.reward}
        _ = sess.run(self.v_op, feed_dict=feed)

        # Update target network
        sess.run(self.list_update_v_ops)

    def get_agent_action_type(self, action_type_int, n_types=3):
        """
        action_type_int: a list over time steps of 1D np.array of
        action ints in the set {0,1,2}

        Returns: [time*n_agents, 1-hot indicator of action type]
        """
        action = np.vstack(action_type_int)
        n_steps = len(action_type_int)
        action_1hot = np.zeros([n_steps, self.n_agents, n_types],
                               dtype=int)
        grid = np.indices((n_steps, self.n_agents))
        action_1hot[grid[0], grid[1], action] = 1
        action_1hot.shape = (n_steps*self.n_agents, n_types)

        return action_1hot

    def train(self, sess, list_buf, list_buf_new, epsilon):
        """1-step of incentive function.

        Args:
            sess: TF session
            list_buf: list of designer and agents' trajectory buffers
            list_buf_new: list of designer and agents' validation trajectory buffers
            epsilon: float exploration parameter
        """
        # 1st and 2nd trajectories experienced by the incentive designer
        buf_self = list_buf[self.designer_id]
        buf_self_new = list_buf_new[self.designer_id]
        
        buf_agent = list_buf[self.agents.agent_id]
        (obs_tensor, action_1hot, reward, r_env, obs_tensor_next,
         done) = self.agents.reshape_batch(buf_agent)

        n_steps = len(buf_self.reward)

        # Get agents' value function outputs
        feed = {self.agents.obs_tensor: obs_tensor_next}
        v_next = np.reshape(sess.run(self.agents.v, feed_dict=feed),
                            [n_steps*self.n_agents])
        feed = {self.agents.obs_tensor: obs_tensor}
        v = np.reshape(sess.run(self.agents.v, feed_dict=feed),
                       [n_steps*self.n_agents])

        # Feed placeholders for the fictitious policy update step by agents
        # This requires data from the first trajectory
        feed = {}
        feed[self.agents.obs_tensor] = obs_tensor
        feed[self.agents.action_taken] = action_1hot
        feed[self.agents.r_env] = r_env
        feed[self.agents.epsilon] = epsilon
        feed[self.agents.v_next_ph] = v_next
        feed[self.agents.v_ph] = v

        # Feed placeholders for the ID's gradient.
        # This requires data from the second trajectory
        buf_agent_new = list_buf_new[self.agents.agent_id]
        (obs_tensor_new, action_1hot_new, reward_new, r_env_new,
         obs_tensor_next_new,
         done_new) = self.agents.reshape_batch(buf_agent_new)
        feed[self.policy_new.obs] = obs_tensor_new
        feed[self.policy_new.action_taken] = action_1hot_new
        
        # Fictitious policy update step by agents requires output from
        # the ID's tax function, which requires data from ID's first traj
        feed[self.obs_tensor] = buf_self.obs_tensor
        feed[self.action_agents] = util.convert_batch_actions_int_to_1hot(
            buf_self.action_agents, self.dim_action_agents)

        # Feed everything required to duplicate the computation of agents'
        # total rewards during the first trajectory
        feed[self.agents.action_type] = self.get_agent_action_type(
            buf_agent.action_type)

        # ------------ Compute advantages for ID --------------- #
        # V(s_0),...,V(s_{T-1})
        v_new = sess.run(self.v, feed_dict={
            self.obs_tensor: buf_self_new.obs_tensor})
        v_new = np.reshape(v_new, [-1])

        # V(s_T)
        v_next_new_last = sess.run(self.v, feed_dict={
            self.obs_tensor: [buf_self_new.obs_tensor_next[-1]]})
        v_next_new_last = np.reshape(v_next_new_last, [-1])

        # [1, T+1, 1]
        value_rollouts = np.concatenate((v_new, v_next_new_last))
        # [T]
        advantages = util.compute_advantages(
            buf_self_new.reward, value_rollouts, self.gamma, self.gae_lambda)
        feed[self.advantages] = advantages
        # ----------- End advantages --------------------------- #
        
        _ = sess.run(self.incentive_op, feed_dict=feed)
