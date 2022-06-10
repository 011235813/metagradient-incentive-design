"""Incentive designer with PPO-type loss function."""

import sys

import numpy as np
import tensorflow as tf

from incentive_design.alg.id_foundation import MetaGrad1Step
from incentive_design.alg import ppo_agent
from incentive_design.alg import ppo_agent_m1
from incentive_design.alg import ppo_agent_m2
from incentive_design.alg import networks
from incentive_design.utils import util


class MetaGrad1Step(MetaGrad1Step):

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
        super().__init__(designer_id, agent_name, config, dim_action,
                         dim_obs_flat, dim_obs_tensor, n_agents, nn)
        self.ppo_epsilon = config.ppo_epsilon
        self.gae_lambda = config.gae_lambda
        assert self.use_critic

    def create_tax_train_op(self):
        """Set up TF graph for metagradient."""
        
        # [time*n_agents]
        self.advantages = tf.placeholder(tf.float32, [None], 'advantages')
        
        # Assumes parameter sharing at the agents' level
        policy_params_new = {}
        # \hat{\theta} <-- \theta + \Delta \theta
        for grad, var in zip(self.agents.policy_grads,
                             self.agents.policy_params):
            if isinstance(self.agents, ppo_agent.Agent):
                beta1_power, beta2_power = self.agents.actor_opt._get_beta_accumulators()
                lr_ = self.agents.lr_actor * tf.sqrt(1 - beta2_power) / (1 - beta1_power)
                m, v = self.agents.actor_opt.get_slot(var, 'm'), self.agents.actor_opt.get_slot(var, 'v')
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

        # ------------ Compute advantages for ID --------------- #
        # V(s_0),...,V(s_{T-1})
        zeros = np.zeros((2, 1, self.nn.n_lstm))
        v_new = sess.run(self.v, feed_dict={
            self.obs_tensor: [buf_self_new.obs_tensor],
            self.obs_flat: [buf_self_new.obs_flat],
            self.new_v_main_state: zeros})
        v_new = np.reshape(v_new, [-1])

        # V(s_T)
        v_next_new_last = sess.run(self.v, feed_dict={
            self.obs_tensor: [[buf_self_new.obs_tensor_next[-1]]],
            self.obs_flat: [[buf_self_new.obs_flat_next[-1]]],
            self.new_v_main_state: zeros})
        v_next_new_last = np.reshape(v_next_new_last, [-1])

        # [1, T+1, 1]
        value_rollouts = np.concatenate((v_new, v_next_new_last))
        # [T]
        advantages = util.compute_advantages(
            buf_self_new.reward, value_rollouts, self.gamma, self.gae_lambda)
        feed[self.advantages] = advantages
        # ----------- End advantages --------------------------- #
        
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
