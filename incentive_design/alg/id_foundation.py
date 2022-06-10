"""Algorithm(s) for the top-level incentive designer."""
import sys

import numpy as np
import tensorflow as tf

from incentive_design.alg import actor_critic_ps
from incentive_design.alg import ppo_agent
from incentive_design.alg import ppo_agent_m1
from incentive_design.alg import ppo_agent_m2
from incentive_design.alg import networks
from incentive_design.utils import util


class MetaGrad1Step(object):

    def __init__(self, designer_id, agent_name, config, dim_action,
                 dim_obs_flat, dim_obs_tensor, n_agents, nn):
        """Initialization.

        Args:
            designer_id: int
            agent_name: str
            config: ConfigDict
            dim_action: int
            dim_obs_flat: int, if obs has a flat part, else None
            dim_obs_tensor: list, if obs has an image part, else None
            n_agents: int
            nn: ConfigDict
        """
        assert not (dim_obs_tensor is None and dim_obs_flat is None)
        
        self.designer_id = designer_id
        # self.agent_name = 'agent_%d' % self.agent_id
        self.agent_name = agent_name
        self.dim_action = dim_action
        self.dim_obs_flat = dim_obs_flat
        self.dim_obs_tensor = dim_obs_tensor
        self.tensor_and_flat = (self.dim_obs_flat and self.dim_obs_tensor)
        self.n_agents = n_agents
        self.nn = nn

        self.gamma = config.gamma
        self.grad_clip = config.grad_clip
        self.lr_incentive = config.lr_incentive
        self.lr_v = config.lr_v
        self.r_scalar = config.r_scalar
        self.tau = config.tau
        self.use_critic = config.use_critic
        self.use_noise = config.use_noise
        self.ou = util.OU(self.dim_action)

        self.create_networks()

    def create_networks(self):
        """Creates neural network part of the TF graph."""
        if self.tensor_and_flat:
            self.obs_tensor = tf.placeholder(
                tf.float32, [1, None]+list(self.dim_obs_tensor),'obs_tensor')
            self.obs_flat = tf.placeholder(
                tf.float32, [1, None, self.dim_obs_flat[0]], 'obs_flat')
            self.obs_tensor_constant = tf.placeholder(
                tf.float32, [1, None]+list(self.dim_obs_tensor),
                'obs_tensor_constant')
            self.obs_flat_constant = tf.placeholder(
                tf.float32, [1, None, self.dim_obs_flat[0]],
                'obs_flat_constant')
            
            tax_function = networks.tax_image_vec
            value_net = networks.vnet_image_vec

            tax_lstm = tf.nn.rnn_cell.LSTMCell(self.nn.n_lstm)
            self.new_tax_state = tf.placeholder(
                tf.float32, [2, 1, self.nn.n_lstm], 'new_actor_state')
        
            v_main_lstm = tf.nn.rnn_cell.LSTMCell(self.nn.n_lstm)
            self.new_v_main_state = tf.placeholder(
                tf.float32, [2, 1, self.nn.n_lstm], 'new_v_main_state')  
            v_target_lstm = tf.nn.rnn_cell.LSTMCell(self.nn.n_lstm)            
            self.new_v_target_state = tf.placeholder(
                tf.float32, [2, 1, self.nn.n_lstm], 'new_v_target_state')

        else:
            raise NotImplementedError

        self.noise = tf.placeholder(tf.float32, [None, self.dim_action], 'noise')

        with tf.variable_scope(self.agent_name):
            with tf.variable_scope('eta'):
                tax_tuple_state = tf.nn.rnn_cell.LSTMStateTuple(self.new_tax_state[0], self.new_tax_state[1])
                self.tax_function, self.tax_state = tax_function(
                    self.obs_tensor, self.obs_flat,
                    self.dim_action, self.nn, self.noise, tax_lstm, tax_tuple_state)
            with tf.variable_scope('eta', reuse=True):
                (self.tax_function_duplicate,
                 self.tax_state_duplicate) = tax_function(
                    self.obs_tensor_constant, self.obs_flat_constant,
                    self.dim_action, self.nn, self.noise, tax_lstm, tax_tuple_state)

            if self.use_critic:
                with tf.variable_scope('v_main'):
                    v_main_tuple_state = tf.nn.rnn_cell.LSTMStateTuple(
                        self.new_v_main_state[0], self.new_v_main_state[1]
                    )
                    v, v_main_state = value_net(
                        self.obs_tensor, self.obs_flat, self.nn, v_main_lstm, v_main_tuple_state)
                    # Remove the n_agents dimension from output
                    self.v = v[0]
                with tf.variable_scope('v_target'):
                    v_target_tuple_state = tf.nn.rnn_cell.LSTMStateTuple(
                        self.new_v_target_state[0], self.new_v_target_state[1]
                    )
                    v_target, v_target_state = value_net(
                        self.obs_tensor, self.obs_flat, self.nn, v_target_lstm, v_target_tuple_state)
                    # Remove the n_agents dimension from output
                    self.v_target = v_target[0]
        self.eta_vars = tf.trainable_variables(self.agent_name + '/eta')

        if self.use_critic:
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
        assert isinstance(agents, actor_critic_ps.ActorCritic) or isinstance(agents, ppo_agent.Agent)
        self.agents = agents

    def compute_tax(self, obs_tensor, obs_flat, sess, apply_noise=False, lstm_state=None):
        """Computes tax rates for the current time step.

        Args:
            obs: np.array of ID's observation
            sess: TF session
        Returns: np.array of tax rates
        """

        noise = self.ou.step(self.use_noise and apply_noise)
        noise = np.expand_dims(noise, axis=0)

        feed = {self.obs_tensor: np.array([[obs_tensor]]),
                self.obs_flat: np.array([[obs_flat]]),
                self.noise: noise}
        
        if lstm_state:
            feed[self.new_tax_state] = lstm_state
        else:
            feed[self.new_tax_state] = np.zeros([2, 1, self.nn.n_lstm])

        tax, tax_state = sess.run([self.tax_function, self.tax_state], feed_dict=feed)
        tax = tax.flatten()

        return tax, noise[0], tax_state

    def create_critic_train_op(self):
        self.v_target_next = tf.placeholder(tf.float32, [None], 'v_target_next')
        self.reward = tf.placeholder(tf.float32, [None], 'reward')
        td_target = self.reward + self.gamma * self.v_target_next
        self.loss_v = tf.reduce_mean(tf.square(
            td_target - tf.squeeze(self.v)))
        self.v_opt = tf.train.AdamOptimizer(self.lr_v)
        self.v_op = self.v_opt.minimize(self.loss_v)

    def create_tax_train_op(self):
        """Set up TF graph for metagradient."""
        # td error or return of the ID on the valdition trajectory
        if self.use_critic:
            self.v_td_error = tf.placeholder(tf.float32, [None], 'v_td_error')
        else:
            self.returns = tf.placeholder(tf.float32, [None], 'returns')

        # Assumes parameter sharing at the agents' level
        policy_params_new = {}
        # \hat{\theta} <-- \theta + \Delta \theta
        for grad, var in zip(self.agents.policy_grads,
                             self.agents.policy_params):
            if isinstance(self.agents, ppo_agent.Agent):
                beta1_power, beta2_power = self.agents.actor_opt._get_beta_accumulators()
                lr_ = self.agents.lr_actor * tf.sqrt(1 - beta2_power) / (1 - beta1_power)
                m, v = self.agents.actor_opt.get_slot(var, 'm'), self.agents.actor_opt.get_slot(var, 'v')
                # print(m, v)
                # input('here')
                m = m + (grad - m) * (1 - .9)
                v = v + (tf.square(tf.stop_gradient(grad)) - v) * (1 - .999)
                policy_params_new[var.name] = var - m * lr_ / (tf.sqrt(v) + 1E-8)
            else:
                policy_params_new[var.name] = var - self.agents.lr_actor * grad
        
        # Store into dummy model to preserve dependence on \eta
        if isinstance(self.agents, ppo_agent.Agent):
            self.policy_new = self.agents.policy_new(
                policy_params_new, self.agents.dim_obs_tensor,
                self.agents.dim_obs_flat, self.agents.dim_action,
                self.agents.agent_name, self.agents.nn, self.agents.n_agents)
        else:
            self.policy_new = self.agents.policy_new(
                policy_params_new, self.agents.dim_obs_tensor,
                self.agents.dim_obs_flat, self.agents.dim_action,
                self.agents.agent_name, self.agents.nn)

        # shape: [time*n_agents] due to parameter-sharing
        log_probs_taken = tf.log(tf.reduce_sum(
            tf.multiply(self.policy_new.probs,
                        self.policy_new.action_taken), axis=1) + 1e-15)

        # shape: [time]
        sum_agents_log_pi = tf.reduce_sum(
            tf.reshape(log_probs_taken, [-1, self.n_agents]), axis=1)
        if self.use_critic:
            self.incentive_loss = -tf.reduce_mean(tf.multiply(
                sum_agents_log_pi, self.v_td_error))
        else:
            self.incentive_loss = -tf.reduce_mean(tf.multiply(
                sum_agents_log_pi, self.returns))

        incentive_opt = tf.train.AdamOptimizer(self.lr_incentive)
        if self.grad_clip:
            grad_var = incentive_opt.compute_gradients(
                self.incentive_loss, var_list=self.eta_vars)
            grad_var = [(tf.clip_by_norm(tup[0], self.grad_clip) if (not tup[0] is None) else tup[0], tup[1]) for tup in grad_var]
            self.incentive_op = incentive_opt.apply_gradients(grad_var)
        else:
            self.incentive_op = incentive_opt.minimize(
                self.incentive_loss, var_list=self.eta_vars)

    def reshape_batch_tax_info(self, buf):
        """Reshapes info into [time*n_agents, original_dim]."""
        
        # Each buf.* is list over time of list over agents of scalar
        total_endowment_coin = np.array(buf.total_endowment_coin).flatten()
        last_coin = np.array(buf.last_coin).flatten()
        escrow_coin = np.array(buf.escrow_coin).flatten()
        util_prev = np.array(buf.util_prev).flatten()
        inventory_coin = np.array(buf.inventory_coin).flatten()
        total_labor = np.array(buf.total_labor).flatten()
        curr_rate_max = np.array(buf.curr_rate_max).flatten()

        # buf.enact_tax is a list over time of int
        # duplicate for parameter sharing
        enact_tax = np.repeat(np.array(buf.enact_tax), self.n_agents, axis=0)

        # buf.idx_episode is a list over time of int
        # duplicate for parameter sharing
        completions = np.repeat(np.array(buf.completions),
                                self.n_agents, axis=0)

        return (total_endowment_coin, last_coin, escrow_coin,
                util_prev, inventory_coin, total_labor,
                curr_rate_max, enact_tax, completions)

    def train_critic(self, sess, buf):
        """Update value network."""
        batch_size = len(buf.reward)

        feed = {self.obs_tensor: [buf.obs_tensor_next],
                self.obs_flat: [buf.obs_flat_next]}
        feed[self.new_v_target_state] = np.zeros([2, 1, self.nn.n_lstm])
        v_target_next = sess.run(self.v_target, feed_dict=feed)
        v_target_next = np.reshape(v_target_next, [batch_size])

        feed = {self.obs_tensor: [buf.obs_tensor],
                self.obs_flat: [buf.obs_flat],
                self.v_target_next: v_target_next,
                self.reward: buf.reward}
        feed[self.new_v_main_state] = np.zeros([2, 1, self.nn.n_lstm])
        _ = sess.run(self.v_op, feed_dict=feed)

        # Update target network
        sess.run(self.list_update_v_ops)

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
        
        if isinstance(self.agents, ppo_agent.Agent):
            buf_agent = list_buf[self.agents.agent_id]
            (obs_flat, obs_tensor, reward, obs_flat_next, obs_tensor_next) = self.agents.reshape_batch(buf_agent)

            n_steps = len(buf_self.reward)
            zeros = np.zeros((2, self.agents.n_agents, self.agents.nn.n_lstm))

            # Get agents' value function outputs
            feed = {self.agents.obs_tensor: obs_tensor_next,
                    self.agents.obs_flat: obs_flat_next,
                    self.agents.new_v_main_state: zeros}
            v_next = sess.run(self.agents.v, feed_dict=feed)
            feed = {self.agents.obs_tensor: obs_tensor,
                    self.agents.obs_flat: obs_flat,
                    self.agents.new_v_main_state: zeros}
            v = sess.run(self.agents.v, feed_dict=feed)

            # Feed placeholders for the fictitious policy update step by agents
            # This requires data from the first trajectory
            feed = {}
            feed[self.agents.obs_tensor] = obs_tensor
            feed[self.agents.obs_flat] = obs_flat
            feed[self.agents.reward] = reward
            feed[self.agents.epsilon] = epsilon
            # Feeding in values
            feed[self.agents.v_ph] = v
            feed[self.agents.v_next_ph] = v_next
            
            
            # Feed placeholders for the ID's gradient.
            # This requires data from the second trajectory
            buf_agent_new = list_buf_new[self.agents.agent_id]
            (obs_flat_new, obs_tensor_new, reward_new, 
             obs_flat_next_new, obs_tensor_next_new) = self.agents.reshape_batch(buf_agent_new)
            feed[self.policy_new.obs_tensor] = obs_tensor_new
            feed[self.policy_new.obs_flat] = obs_flat_new
        
        else:
            buf_agent = list_buf[self.agents.agent_id]
            (obs_flat, obs_tensor, action_1hot, action_mask, reward,
             obs_flat_next, obs_tensor_next, done) = self.agents.reshape_batch(buf_agent)

            n_steps = len(buf_self.reward)

            # Get agents' value function outputs
            feed = {self.agents.obs_tensor: obs_tensor_next,
                    self.agents.obs_flat: obs_flat_next}
            v_next = np.reshape(sess.run(self.agents.v, feed_dict=feed),
                                [n_steps*self.n_agents])
            feed = {self.agents.obs_tensor: obs_tensor,
                    self.agents.obs_flat: obs_flat}
            v = np.reshape(sess.run(self.agents.v, feed_dict=feed),
                           [n_steps*self.n_agents])

            # Feed placeholders for the fictitious policy update step by agents
            # This requires data from the first trajectory
            feed = {}
            feed[self.agents.obs_tensor] = obs_tensor
            feed[self.agents.obs_flat] = obs_flat
            feed[self.agents.action_taken] = action_1hot
            feed[self.agents.action_mask] = action_mask
            feed[self.agents.reward] = reward
            feed[self.agents.epsilon] = epsilon
            feed[self.agents.v_next_ph] = v_next
            feed[self.agents.v_ph] = v

            # Feed placeholders for the ID's gradient.
            # This requires data from the second trajectory
            buf_agent_new = list_buf_new[self.agents.agent_id]
            (obs_flat_new, obs_tensor_new, action_1hot_new, action_mask_new,
             reward_new, obs_flat_next_new, obs_tensor_next_new,
             done_new) = self.agents.reshape_batch(buf_agent_new)
            feed[self.policy_new.obs_tensor] = obs_tensor_new
            feed[self.policy_new.obs_flat] = obs_flat_new
            feed[self.policy_new.action_taken] = action_1hot_new
            feed[self.policy_new.action_mask] = action_mask_new
        
        # Fictitious policy update step by agents requires output from
        # the ID's tax function, which requires data from ID's first traj
        feed[self.obs_tensor] = [buf_self.obs_tensor]
        feed[self.obs_flat] = [buf_self.obs_flat]
        feed[self.obs_tensor_constant] = [buf_self.obs_tensor_constant]
        feed[self.obs_flat_constant] = [buf_self.obs_flat_constant]
        feed[self.noise] = buf_self.noise
        feed[self.new_tax_state] = np.zeros((2, 1, self.nn.n_lstm))

        # Feed everything required to duplicate the computation of tax
        # and agents' rewards during the first trajectory
        (total_endowment_coin, last_coin, escrow_coin, util_prev,
         inventory_coin, total_labor, curr_rate_max, enact_tax,
         completions) = self.reshape_batch_tax_info(buf_self)
        feed[self.agents.total_endowment_coin] = total_endowment_coin
        feed[self.agents.last_coin] = last_coin
        feed[self.agents.escrow_coin] = escrow_coin
        feed[self.agents.util_prev] = util_prev
        feed[self.agents.inventory_coin] = inventory_coin
        feed[self.agents.total_labor] = total_labor
        feed[self.agents.curr_rate_max] = curr_rate_max
        feed[self.agents.enact_tax] = enact_tax
        feed[self.agents.completions] = completions

        # ID's critic or returns is used in the outer loss function
        reward_id = buf_self_new.reward
        if self.use_critic:
            zeros = np.zeros((2, 1, self.nn.n_lstm))
            v_new = np.reshape(sess.run(
                self.v, feed_dict={self.obs_tensor: [buf_self_new.obs_tensor],
                                   self.obs_flat: [buf_self_new.obs_flat],
                                   self.new_v_main_state: zeros}),
                               [n_steps])
            v_next_new = np.reshape(sess.run(
                self.v, feed_dict={
                    self.obs_tensor: buf_self_new.obs_tensor_next,
                    self.obs_flat: buf_self_new.obs_flat_next,
                    self.new_v_main_state: zeros}),
                                    [n_steps])
            feed[self.v_td_error] = reward_id + self.gamma*v_next_new - v_new
        else:
            returns_id = util.compute_returns(reward_id, self.gamma)
            feed[self.returns] = returns_id
        
        if isinstance(self.agents, ppo_agent.Agent):

            buf = list_buf[self.agents.agent_id]
            buf_new = list_buf_new[self.agents.agent_id]
            
            self.agents.feed_actions(feed,
                buf.log_probs,
                buf.action,
                buf.action_mask
            )

            self.policy_new.feed_actions(feed,
                buf_new.action,
                buf_new.action_mask
            )

            zeros = np.zeros((2, self.agents.n_agents, self.agents.nn.n_lstm))
            feed[self.agents.new_actor_state] = zeros
            feed[self.policy_new.new_actor_state] = zeros

            _ = sess.run(self.incentive_op, feed_dict=feed)
        else:
            _ = sess.run(self.incentive_op, feed_dict=feed)
