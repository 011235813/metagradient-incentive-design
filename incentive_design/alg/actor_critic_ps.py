"""Actor critic with parameter sharing.

Unlike actor_critic.py, this does not support a factored action space,
This is meant to be used only for the agents, not the planner,
in Foundation.
"""

import numpy as np
import tensorflow as tf

from incentive_design.alg import actor_critic
from incentive_design.alg import networks
from incentive_design.utils import util


class ActorCritic(actor_critic.ActorCritic):

    def __init__(self, agent_name, config, dim_action,
                 dim_obs_flat, dim_obs_tensor,  n_agents, nn):
        """Initialization.

        Args:
            agent_name: string
            config: ConfigDict
            dim_action: int
            dim_obs_flat: int, if obs has a flat part, else None
            dim_obs_tensor: list, if obs has an image part, else None
            n_agents: int
            nn: ConfigDict
        """
        super().__init__(agent_name, config, dim_action,
                         dim_obs_flat, dim_obs_tensor,  nn)
        # self.dim_action = dim_action
        self.n_agents = n_agents

    def create_networks(self):
        """Placeholders and neural nets."""
        if self.tensor_and_flat:
            self.obs_tensor = tf.placeholder(
                tf.float32, [None]+list(self.dim_obs_tensor), 'obs_tensor')
            self.obs_flat = tf.placeholder(
                tf.float32, [None, self.dim_obs_flat[0]], 'obs_flat')
            actor_net = networks.actor_image_vec
            value_net = networks.vnet_image_vec

        self.epsilon = tf.placeholder(tf.float32, [], 'epsilon')
        self.action_mask = tf.placeholder(
            tf.float32, [None, self.dim_action], 'action_mask')

        with tf.variable_scope(self.agent_name):
            with tf.variable_scope('policy'):
                if self.tensor_and_flat:
                    probs = actor_net(self.obs_tensor, self.obs_flat,
                                      self.dim_action, self.nn)

            # Apply action mask and normalize
            prob = tf.multiply(probs, self.action_mask)
            prob = prob / tf.reshape(tf.reduce_sum(prob, axis=1), [-1,1])
            # Exploration lower bound
            self.probs = ((1 - self.epsilon) * prob +
                          self.epsilon / self.dim_action)
            self.log_probs = tf.log(self.probs + 1e-15)
            self.action_samples = tf.multinomial(self.log_probs, 1)

            with tf.variable_scope('v_main'):
                if self.tensor_and_flat:
                    self.v = value_net(
                        self.obs_tensor, self.obs_flat, self.nn)
            with tf.variable_scope('v_target'):
                if self.tensor_and_flat:
                    self.v_target = value_net(
                        self.obs_tensor, self.obs_flat, self.nn)

        self.policy_params = tf.trainable_variables(
            self.agent_name + '/policy')
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

    def run_actor(self, obs_tensor, obs_flat, action_masks, sess, epsilon):
        """Gets actions of all agents from shared policy.

        Args:
            obs_tensor: list of np.array image part of obs
            obs_flat: list of np.array flat part of obs
            action_masks: list of binary np.array
                          We assume each agent's action mask 
                          is an np.array, not [np.array].
                          corresponding to multi_action_mode_agents=False
            sess: TF session
            epsilon: float

        Returns: np.array of action ints
        """        
        # Convert to np.array with agents on the batch dimension
        if self.tensor_and_flat:
            feed = {self.obs_tensor: np.array(obs_tensor),
                    self.obs_flat: np.array(obs_flat),
                    self.epsilon: epsilon}
        feed[self.action_mask] = np.array(action_masks)

        actions = sess.run(self.action_samples, feed_dict=feed)
        
        return actions.flatten()

    def create_policy_gradient_op(self):

        self.v_next_ph = tf.placeholder(tf.float32, [None], 'v_next_ph')
        self.v_ph = tf.placeholder(tf.float32, [None], 'v_ph')
        v_td_error = self.reward + self.gamma*self.v_next_ph - self.v_ph

        self.action_taken = tf.placeholder(
            tf.float32, [None, self.dim_action])
        # list of Tensors with shape [batch, 1]
        log_probs_taken = tf.log(tf.reduce_sum(tf.multiply(
            self.probs, self.action_taken), axis=1) + 1e-15)

        self.entropy = -tf.reduce_mean(tf.reduce_sum(tf.multiply(
            self.probs, self.log_probs), axis=1))

        self.policy_loss = -tf.reduce_mean(
            tf.multiply(log_probs_taken, v_td_error))
        self.loss = self.policy_loss - self.entropy_coeff * self.entropy

        self.policy_grads = tf.gradients(self.loss, self.policy_params)
        grads = ([tf.clip_by_norm(grad, self.grad_clip) for grad in
                 self.policy_grads] if self.grad_clip else self.policy_grads)
        grads_and_vars = list(zip(grads, self.policy_params))
        self.policy_opt = tf.train.GradientDescentOptimizer(self.lr_actor)
        self.policy_op = self.policy_opt.apply_gradients(grads_and_vars)

    def reshape_batch(self, buf):
        """Reshapes agents' experiences into [time*agents, original dims]."""
        n_steps = len(buf.reward)
        # buf.obs_flat and buf.obs_tensor: list over time steps of
        # list over agents of obs
        obs_flat = np.vstack(buf.obs_flat)
        obs_tensor = np.vstack(buf.obs_tensor)
        
        # buf.action is a list over time steps of 1D np.array of action ints
        action = np.vstack(buf.action) # [time, n_agents]
        action_1hot = np.zeros([n_steps, self.n_agents, self.dim_action],
                                dtype=int)
        grid = np.indices((n_steps, self.n_agents))
        action_1hot[grid[0], grid[1], action] = 1
        action_1hot.shape = (n_steps*self.n_agents, self.dim_action)

        # buf.action_mask is a list over time steps of list over agents
        # of np.array
        action_mask = np.vstack(buf.action_mask)
        
        # buf.reward is a list over time of np.array of agents' rewards
        reward = np.vstack(buf.reward)
        reward.shape = (n_steps * self.n_agents)

        obs_flat_next = np.vstack(buf.obs_flat_next)
        obs_tensor_next = np.vstack(buf.obs_tensor_next)

        done = np.stack(buf.done)
        done = np.repeat(done, self.n_agents, axis=0)

        return (obs_flat, obs_tensor, action_1hot, action_mask, reward,
                obs_flat_next, obs_tensor_next, done)

    def train(self, sess, buf, epsilon):
        """Training step for policy and value function

        Args:
            sess: TF session
            buf: Buffer object
            epsilon: float
        """
        (obs_flat, obs_tensor, action_1hot, action_mask, reward,
         obs_flat_next, obs_tensor_next, done) = self.reshape_batch(buf)

        # Update value network
        if self.tensor_and_flat:
            feed = {self.obs_tensor: obs_tensor_next,
                    self.obs_flat: obs_flat_next}
        else:
            raise NotImplementedError
        v_target_next, v_next = sess.run([self.v_target, self.v],
                                         feed_dict=feed)
        v_target_next = np.squeeze(v_target_next)
        v_next = np.squeeze(v_next)

        feed = {self.v_target_next: v_target_next,
                self.reward: reward}
        if self.tensor_and_flat:
            feed[self.obs_tensor] = obs_tensor
            feed[self.obs_flat] = obs_flat
        else:
            raise NotImplementedError
        _, v = sess.run([self.v_op, self.v], feed_dict=feed)
        v = np.squeeze(v)

        feed = {self.action_taken: action_1hot}
        feed[self.action_mask] = action_mask
        feed[self.reward] = reward
        feed[self.epsilon] =  epsilon
        feed[self.v_next_ph] = v_next
        feed[self.v_ph] = v
        if self.tensor_and_flat:
            feed[self.obs_tensor] = obs_tensor
            feed[self.obs_flat] = obs_flat
        else:
            raise NotImplementedError

        _ = sess.run(self.policy_op, feed_dict=feed)

        # Update target network
        sess.run(self.list_update_v_ops)
