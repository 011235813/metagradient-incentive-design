"""Actor critic with parameter sharing for method 1.

This implementation allows an oracle incentive designer to
differentiate through the agents' policy update.

This does not support a factored action space,
This is meant to be used only for the agents, not the planner,
in Foundation.
"""

import numpy as np
import tensorflow as tf

from incentive_design.alg import actor_critic_ps_m2
from incentive_design.alg import networks
from incentive_design.utils import util


class ActorCritic(actor_critic_ps_m2.ActorCritic):

    def __init__(self, agent_id, agent_name, config, dim_action,
                 dim_obs_flat, dim_obs_tensor,  n_agents, nn,
                 tax, utility):
        """Initialization.

        Args:
            agent_id: int
            agent_name: string
            config: ConfigDict
            dim_action: int
            dim_obs_flat: int, if obs has a flat part, else None
            dim_obs_tensor: list, if obs has an image part, else None
            n_agents: int
            nn: ConfigDict
        """
        super().__init__(agent_id, agent_name, config, dim_action,
                         dim_obs_flat, dim_obs_tensor,  n_agents, nn,
                         tax, utility)

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

            with tf.variable_scope('policy_prime'):
                if self.tensor_and_flat:
                    probs = actor_net(self.obs_tensor, self.obs_flat,
                                      self.dim_action, self.nn)
            probs = tf.multiply(probs, self.action_mask)
            probs = probs / tf.reshape(tf.reduce_sum(probs, axis=1), [-1,1])
            self.probs_prime = ((1 - self.epsilon) * probs +
                                self.epsilon / self.dim_action)
            self.log_probs_prime = tf.log(self.probs_prime + 1e-15)
            self.action_samples_prime = tf.multinomial(self.log_probs_prime, 1)

            with tf.variable_scope('v_main'):
                if self.tensor_and_flat:
                    self.v = value_net(
                        self.obs_tensor, self.obs_flat, self.nn)
            with tf.variable_scope('v_target'):
                if self.tensor_and_flat:
                    self.v_target = value_net(
                        self.obs_tensor, self.obs_flat, self.nn)

        self.policy_params = tf.trainable_variables(
            self.agent_name + '/policy/')
        self.policy_prime_params = tf.trainable_variables(
            self.agent_name + '/policy_prime')

        self.list_copy_main_to_prime_ops = []
        for idx, var in enumerate(self.policy_prime_params):
            self.list_copy_main_to_prime_ops.append(
                var.assign(self.policy_params[idx]))

        self.list_copy_prime_to_main_ops = []
        for idx, var in enumerate(self.policy_params):
            self.list_copy_prime_to_main_ops.append(
                var.assign(self.policy_prime_params[idx]))
                        
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
    def run_actor(self, obs_tensor, obs_flat, action_masks, sess, epsilon,
                  prime=False):
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

        if prime:
            actions = sess.run(self.action_samples_prime, feed_dict=feed)
        else:
            actions = sess.run(self.action_samples, feed_dict=feed)
        
        return actions.flatten()    

    def create_update_op(self):
        """Set up TF graph for 1-step prime parameter update."""
        
        # No need to do any modification to reward because
        # self.reward placeholder will be fed with the complete reward
        # that includes taxation and difference of utilities
        v_td_error = self.reward + self.gamma*self.v_next_ph - self.v_ph

        # list of Tensors with shape [batch, 1]
        log_probs_taken = tf.log(tf.reduce_sum(tf.multiply(
            self.probs_prime, self.action_taken), axis=1) + 1e-15)

        entropy = -tf.reduce_mean(tf.reduce_sum(tf.multiply(
            self.probs_prime, self.log_probs_prime), axis=1))

        policy_loss = -tf.reduce_mean(
            tf.multiply(log_probs_taken, v_td_error))
        loss = policy_loss - self.entropy_coeff * entropy

        grads = tf.gradients(loss, self.policy_prime_params)
        grads = ([tf.clip_by_norm(grad, self.grad_clip) for grad in grads]
                 if self.grad_clip else grads)
        grads_and_vars = list(zip(grads, self.policy_prime_params))
        policy_prime_opt = tf.train.GradientDescentOptimizer(self.lr_actor)
        self.policy_prime_op = policy_prime_opt.apply_gradients(grads_and_vars)

    def update(self, sess, buf, epsilon):
        """Runs 1-step prime parameter update.

        Args:
            sess: TF session
            buf: Buffer object
            epsilon: float
        """
        sess.run(self.list_copy_main_to_prime_ops)
        n_steps = len(buf.reward)

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

        # Update prime policy
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

        _ = sess.run(self.policy_prime_op, feed_dict=feed)

        # Update target network
        sess.run(self.list_update_v_ops)

    def update_main(self, sess):
        sess.run(self.list_copy_prime_to_main_ops)
