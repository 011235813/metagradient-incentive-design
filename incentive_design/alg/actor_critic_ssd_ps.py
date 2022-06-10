"""Actor-critic agents with parameter sharing, used by dual-RL in SSD."""

import numpy as np
import tensorflow as tf

from incentive_design.alg import networks
from incentive_design.utils import util


class ActorCritic(object):

    def __init__(self, agent_id, agent_name, config, dim_action,
                 dim_obs_tensor, n_agents, nn):
        """Initialization.

        Args:
            agent_id: int
            agent_name: string
            config: ConfigDict
            dim_action: int
            dim_obs_tensor: list, if obs has an image part, else None
            n_agents: int
            nn: ConfigDict
        """
        assert not (dim_obs_tensor is None)
        
        self.agent_id = agent_id
        self.agent_name = agent_name
        self.dim_action = dim_action
        self.dim_obs_tensor = dim_obs_tensor
        self.n_agents = n_agents
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
        self.obs_tensor = tf.placeholder(
            tf.float32, [None]+list(self.dim_obs_tensor), 'obs_tensor')
        actor_net = networks.actor_ssd
        value_net = networks.vnet_ssd

        self.epsilon = tf.placeholder(tf.float32, [], 'epsilon')

        with tf.variable_scope(self.agent_name):
            with tf.variable_scope('policy'):
                probs = actor_net(self.obs_tensor, self.dim_action, self.nn)
            # Exploration lower bound
            self.probs = ((1 - self.epsilon) * probs +
                          self.epsilon / self.dim_action)
            self.log_probs = tf.log(self.probs + 1e-15)
            self.action_samples = tf.multinomial(self.log_probs, 1)

            with tf.variable_scope('v_main'):
                self.v = value_net(self.obs_tensor, self.nn)
            with tf.variable_scope('v_target'):
                self.v_target = value_net(self.obs_tensor, self.nn)

        self.policy_params = tf.trainable_variables(
            self.agent_name + '/policy/')
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

    def run_actor(self, obs_tensor, sess, epsilon):
        """Gets actions of all agents from shared policy.

        Args:
            obs_tensor: list of np.array image part of obs
            sess: TF session
            epsilon: float

        Returns: np.array of action ints
        """        
        # Convert to np.array with agents on the batch dimension
        feed = {self.obs_tensor: np.array(obs_tensor),
                self.epsilon: epsilon}

        actions = sess.run(self.action_samples, feed_dict=feed)
        
        return actions.flatten()

    def create_critic_train_op(self):

        self.v_target_next = tf.placeholder(tf.float32, [None],
                                            'v_target_next')
        self.reward = tf.placeholder(tf.float32, [None], 'reward')
        td_target = self.reward + self.gamma * self.v_target_next
        self.loss_v = tf.reduce_mean(tf.square(
            td_target - tf.squeeze(self.v)))
        self.v_opt = tf.train.AdamOptimizer(self.lr_v)
        self.v_op = self.v_opt.minimize(self.loss_v)

    def create_policy_gradient_op(self):
        """Used by the ID for its own parameter update.

        Agents' policy parameter update is computed by create_update_op()
        and update_step(), not here.
        """
        self.v_next_ph = tf.placeholder(tf.float32, [None], 'v_next_ph')
        self.v_ph = tf.placeholder(tf.float32, [None], 'v_ph')
        v_td_error = self.reward + self.gamma*self.v_next_ph - self.v_ph

        self.action_taken = tf.placeholder(
            tf.float32, [None, self.dim_action], 'action_taken')
        log_probs_taken = tf.log(tf.reduce_sum(
            tf.multiply(self.probs, self.action_taken), axis=1) + 1e-15)

        self.entropy = -tf.reduce_sum(tf.multiply(
            self.probs, self.log_probs))

        self.policy_loss = -tf.reduce_sum(
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
        # buf.obs_tensor: list over time steps of list over agents of obs
        obs_tensor = np.vstack(buf.obs_tensor)
        
        # buf.action is a list over time steps of 1D np.array of action ints
        action = np.vstack(buf.action) # [time, n_agents]
        action_1hot = np.zeros([n_steps, self.n_agents, self.dim_action],
                                dtype=int)
        grid = np.indices((n_steps, self.n_agents))
        action_1hot[grid[0], grid[1], action] = 1
        action_1hot.shape = (n_steps*self.n_agents, self.dim_action)

        # buf.reward is a list over time of np.array of agents' total rewards
        reward = np.vstack(buf.reward)
        reward.shape = (n_steps * self.n_agents)

        obs_tensor_next = np.vstack(buf.obs_tensor_next)

        done = np.stack(buf.done)
        done = np.repeat(done, self.n_agents, axis=0)

        return (obs_tensor, action_1hot, reward, obs_tensor_next, done)

    def train(self, sess, buf, epsilon):
        """Runs policy parameter update.

        Args:
            sess: TF session
            buf: Buffer object
            epsilon: float
        """
        n_steps = len(buf.reward)

        (obs_tensor, action_1hot, reward,
         obs_tensor_next, done) = self.reshape_batch(buf)

        # Update value network
        feed = {self.obs_tensor: obs_tensor_next}
        v_target_next, v_next = sess.run([self.v_target, self.v],
                                         feed_dict=feed)
        v_target_next = np.squeeze(v_target_next)
        v_next = np.squeeze(v_next)

        feed = {self.v_target_next: v_target_next,
                self.reward: reward}
        feed[self.obs_tensor] = obs_tensor
        _, v = sess.run([self.v_op, self.v], feed_dict=feed)
        v = np.squeeze(v)

        feed = {self.action_taken: action_1hot}
        feed[self.reward] = reward
        feed[self.epsilon] =  epsilon
        feed[self.v_next_ph] = v_next
        feed[self.v_ph] = v
        feed[self.obs_tensor] = obs_tensor

        _ = sess.run(self.policy_op, feed_dict=feed)

        # Update target network
        sess.run(self.list_update_v_ops)
