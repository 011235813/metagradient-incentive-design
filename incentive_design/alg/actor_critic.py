"""Actor critic with advantage function.

Advantage function is estimated by 1-step TD(0) error.
Supports factored action space.
"""

import numpy as np
import tensorflow as tf

import networks
from incentive_design.utils import util


class ActorCritic(object):

    def __init__(self, agent_name, config, dim_action,
                 dim_obs_flat, dim_obs_tensor,  nn):
        """Initialization.

        Args:
            agent_name: string
            config: ConfigDict
            dim_action: list of ints for each action subspace
            dim_obs_flat: int, if obs has a flat part, else None
            dim_obs_tensor: list, if obs has an image part, else None
            nn: ConfigDict
        """
        assert not (dim_obs_tensor is None and dim_obs_flat is None)
        
        self.agent_name = agent_name
        self.dim_action = dim_action
        self.dim_obs_flat = dim_obs_flat
        self.dim_obs_tensor = dim_obs_tensor
        self.tensor_and_flat = (self.dim_obs_flat and self.dim_obs_tensor)
        self.nn = nn

        self.grad_clip = config.grad_clip
        self.entropy_coeff = config.entropy_coeff
        self.gamma = config.gamma
        self.lr_actor = config.lr_actor
        self.lr_v = config.lr_v
        self.tau = config.tau

        self.create_networks()
        self.create_critic_train_op()
        self.create_policy_gradient_op()

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
        self.action_masks = [tf.placeholder(
            tf.float32, [None, dim], 'action_mask_%d'%idx)
                             for idx, dim in enumerate(self.dim_action)]

        with tf.variable_scope(self.agent_name):
            with tf.variable_scope('policy'):
                if self.tensor_and_flat:
                    # List of Tensors
                    probs = actor_net(self.obs_tensor, self.obs_flat,
                                      self.dim_action, self.nn)
            self.probs = []
            self.log_probs = []
            self.action_samples = []
            for idx in range(len(self.dim_action)):
                # Apply action mask and normalize
                prob = tf.multiply(probs[idx], self.action_masks[idx])
                prob = prob / tf.reshape(tf.reduce_sum(prob, axis=1), [-1,1])
                # Exploration lower bound
                self.probs.append((1 - self.epsilon) * prob +
                                  self.epsilon / self.dim_action[idx])
                self.log_probs.append(tf.log(self.probs[idx] + 1e-15))
                self.action_samples.append(
                    tf.multinomial(self.log_probs[idx], 1))

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
        """Gets action from policy.

        Args:
            obs_tensor: np.array image part of obs
            obs_flat: np.array flat part of obs
            action_masks: list of binary np.array
            sess: TF session
            epsilon: float

        Returns: int if len(dim_action)==1, otherwise a list
        """
        if self.tensor_and_flat:
            feed = {self.obs_tensor: np.array([obs_tensor]),
                    self.obs_flat: np.array([obs_flat]),
                    self.epsilon: epsilon}
        for idx in range(len(self.dim_action)):
            feed[self.action_masks[idx]] = np.array([action_masks[idx]])

        # output is a list
        actions = sess.run(self.action_samples, feed_dict=feed)

        if len(actions) == 1:
            return actions[0][0][0]
        else:
            actions = [action[0][0] for action in actions]
            return actions

    def create_critic_train_op(self):

        self.v_target_next = tf.placeholder(tf.float32, [None], 'v_target_next')
        self.reward = tf.placeholder(tf.float32, [None], 'reward')
        td_target = self.reward + self.gamma * self.v_target_next
        self.loss_v = tf.reduce_mean(tf.square(td_target - tf.squeeze(self.v)))
        self.v_opt = tf.train.AdamOptimizer(self.lr_v)
        self.v_op = self.v_opt.minimize(self.loss_v)

    def create_policy_gradient_op(self):

        self.v_next_ph = tf.placeholder(tf.float32, [None], 'v_next_ph')
        self.v_ph = tf.placeholder(tf.float32, [None], 'v_ph')
        v_td_error = self.reward + self.gamma*self.v_next_ph - self.v_ph

        self.action_taken = [tf.placeholder(tf.float32, [None, dim])
                             for dim in self.dim_action]
        # list of Tensors with shape [batch, 1]
        log_probs_taken = [tf.reshape(tf.log(tf.reduce_sum(
            tf.multiply(prob, action_taken), axis=1) + 1e-15), [-1,1]) for
                           prob, action_taken
                           in zip(self.probs, self.action_taken)]

        # Factored action space
        # log pi(a|s) = \sum_i log \pi_i(a_i|s)
        log_probs_taken = tf.reduce_sum(tf.concat(log_probs_taken, axis=1),
                                        axis=1)

        # Not implemented: entropy requires summing over
        # combinatorial number of actions due to factored action space...
        # self.entropy = -tf.reduce_sum(tf.multiply(self.probs, self.log_probs))

        self.policy_loss = -tf.reduce_sum(
            tf.multiply(log_probs_taken, v_td_error))
        # self.loss = self.policy_loss - self.entropy_coeff * self.entropy
        self.loss = self.policy_loss

        self.policy_grads = tf.gradients(self.loss, self.policy_params)
        grads = ([tf.clip_by_norm(grad, self.grad_clip) for grad in
                 self.policy_grads] if self.grad_clip else self.policy_grads)
        # grads_and_vars = list(zip(self.policy_grads, self.policy_params))
        grads_and_vars = list(zip(grads, self.policy_params))
        self.policy_opt = tf.train.GradientDescentOptimizer(self.lr_actor)
        self.policy_op = self.policy_opt.apply_gradients(grads_and_vars)

    def train(self, sess, buf, epsilon):
        """Training step for policy and value function

        Args:
            sess: TF session
            buf: Buffer object
            epsilon: float
        """
        batch_size = len(buf.reward)
        # Update value network
        if self.tensor_and_flat:
            feed = {self.obs_tensor: buf.obs_tensor_next,
                    self.obs_flat: buf.obs_flat_next}
        else:
            raise NotImplementedError
        v_target_next, v_next = sess.run([self.v_target, self.v],
                                         feed_dict=feed)
        v_target_next = np.reshape(v_target_next, [batch_size])
        v_next = np.reshape(v_next, [batch_size])

        feed = {self.v_target_next: v_target_next,
                self.reward: buf.reward}
        if self.tensor_and_flat:
            feed[self.obs_tensor] = buf.obs_tensor
            feed[self.obs_flat] = buf.obs_flat
        else:
            raise NotImplementedError
        _, v = sess.run([self.v_op, self.v], feed_dict=feed)
        v = np.reshape(v, [batch_size])

        # buf.action is a list over time of list of integers
        # where each inner list is the action(s) at a time step
        # The inner list has length==1 if not multi-mode action
        action_np = np.array(buf.action)
        # list, each entry is for one action dimension
        actions_1hot = [util.convert_batch_action_int_to_1hot(
            action_np[:,idx], self.dim_action[idx])
                        for idx in range(len(self.dim_action))]
        feed = {action_taken : action_1hot for (action_taken, action_1hot)
                in zip(self.action_taken, actions_1hot)}
        feed[self.reward] = buf.reward
        feed[self.epsilon] =  epsilon
        feed[self.v_next_ph] = v_next
        feed[self.v_ph] = v
        if self.tensor_and_flat:
            feed[self.obs_tensor] = buf.obs_tensor
            feed[self.obs_flat] = buf.obs_flat
        else:
            raise NotImplementedError
            
        # Feed action masks
        for idx in range(len(self.dim_action)):
            # buf.action_mask is a list of list of np.array
            # Shape [time, number of subspaces, size of subspace]
            action_mask_np = np.array(buf.action_mask)
            feed[self.action_masks[idx]] = action_mask_np[:,idx,:]

        _ = sess.run(self.policy_op, feed_dict=feed)

        # Update target network
        sess.run(self.list_update_v_ops)
