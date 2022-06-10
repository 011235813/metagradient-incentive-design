"""Actor critic with 1-step TD(0).

Used for comparison to AMD (Baumann) in Escape Room.
"""

import numpy as np
import tensorflow as tf

import networks
from incentive_design.utils import util
from incentive_design.alg import pg_agent_m1


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
        self.alg_name = 'ac'
        self.lr_v = config.lr_v

    def create_networks(self):
        """Placeholders and neural nets."""
        self.obs = tf.placeholder(tf.float32, [None, self.l_obs], 'l_obs')
        self.epsilon = tf.placeholder(tf.float32, [], 'epsilon')

        with tf.variable_scope(self.agent_name):
            with tf.variable_scope('policy_main'):
                with tf.variable_scope('policy'):
                    self.theta = tf.Variable(tf.random_normal(
                        [self.l_obs, self.l_action], stddev=0.5), name='theta')
                probs = tf.nn.softmax(tf.matmul(self.obs, self.theta))
                self.probs = (1 - self.epsilon) * probs + self.epsilon / self.l_action
                self.log_probs = tf.log(self.probs + 1e-15)
                self.action_samples = tf.multinomial(self.log_probs, 1)

            with tf.variable_scope('critic'):
                l1 = tf.layers.dense(
                    inputs=self.obs,
                    units=self.nn.n_h1,
                    activation=tf.nn.relu,
                    kernel_initializer=tf.random_normal_initializer(0., 0.1),
                    bias_initializer=tf.constant_initializer(0.1),
                    name='l1'+str(self.agent_id)
                )
                
                self.v = tf.layers.dense(
                    inputs=l1,
                    units=1,
                    activation=None,
                    kernel_initializer=tf.random_normal_initializer(0., 0.1),
                    bias_initializer=tf.constant_initializer(0.1),
                    name='V'+str(self.agent_id)
                )

            with tf.variable_scope('policy_prime'):
                with tf.variable_scope('policy'):
                    theta_prime = tf.Variable(tf.random_normal(
                        [self.l_obs, self.l_action], stddev=0.5), name='theta')
                probs = tf.nn.softmax(tf.matmul(self.obs, theta_prime))
                self.probs_prime = (1-self.epsilon)*probs + self.epsilon/self.l_action
                self.log_probs_prime = tf.log(self.probs_prime + 1e-15)
                self.action_samples_prime = tf.multinomial(self.log_probs_prime, 1)

        self.policy_params = tf.trainable_variables(
            self.agent_name + '/policy_main')
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

    def create_policy_gradient_op(self):
        """This will be used by the incentive designer for 1-step meta-gradient."""
        # Reward defined by the environment
        self.r_env = tf.placeholder(tf.float32, [None], 'r_ext')
        self.done = tf.placeholder(tf.float32, [None], 'done')

        # agent_id is in [1, n_agents], so need to -1
        this_agent_1hot = tf.one_hot(indices=self.agent_id-1, depth=self.n_agents)
        # Total reward
        r_total = self.r_env + self.r_multiplier * tf.reduce_sum(
            tf.multiply(self.designer.incentive_function,
                        this_agent_1hot), axis=1)

        self.v_next_ph = tf.placeholder(tf.float32, [None], 'v_next_ph')
        self.v_ph = tf.placeholder(tf.float32, [None], 'v_ph')
        v_td_error = r_total + (1-self.done) * self.gamma * self.v_next_ph - self.v_ph

        self.action_taken = tf.placeholder(tf.float32, [None, self.l_action],
                                           'action_taken')
        self.log_probs_taken = tf.log(tf.reduce_sum(
            tf.multiply(self.probs, self.action_taken), axis=1) + 1e-15)

        self.policy_loss = -tf.reduce_sum(
            tf.multiply(self.log_probs_taken, v_td_error))
        self.loss = self.policy_loss

        # Just need these grads to perform 1-step meta-gradient.
        # Do not need a tf.optimizer opt for these main parameter gradients
        # because the prime parameter will already have been updated
        # with the same training data and will be copied into main params.
        self.policy_grads = tf.gradients(self.loss, self.policy_params)

    def create_critic_train_op(self):

        self.reward = tf.placeholder(tf.float32, [None], 'reward')
        td_target = self.reward + (1-self.done) * self.gamma * self.v_next_ph
        self.loss_v = tf.reduce_mean(tf.square(td_target - tf.squeeze(self.v)))
        self.v_opt = tf.train.AdamOptimizer(self.lr_v)
        self.v_op = self.v_opt.minimize(self.loss_v)

    def create_update_op(self):
        """Set up TF graph for 1-step prime parameter update."""
        self.incentive_received = tf.placeholder(tf.float32, [None], 'incentive_received')
        r_total = self.r_env + self.incentive_received
        v_td_error = r_total + (1 - self.done) * self.gamma * self.v_next_ph - self.v_ph

        log_probs_taken = tf.log(tf.reduce_sum(
            tf.multiply(self.probs_prime, self.action_taken), axis=1) + 1e-15)
        policy_loss = -tf.reduce_sum(
            tf.multiply(log_probs_taken, v_td_error))
        loss = policy_loss

        policy_opt_prime = tf.train.GradientDescentOptimizer(self.lr_actor)
        self.policy_op_prime = policy_opt_prime.minimize(loss)

    def update(self, sess, buf, epsilon):
        """Updates value function and prime parameter.

        Args:
            sess: TF session
            buf: Buffer object
            epsilon: float
        """
        batch_size = len(buf.obs)
        # Update value network
        feed = {self.obs: buf.obs_next}
        v_next = sess.run(self.v, feed_dict=feed)
        v_next = np.reshape(v_next, [batch_size])
        total_reward = [buf.r_env[idx] + buf.incentive_received[idx]
                        for idx in range(batch_size)]

        feed = {self.obs: buf.obs,
                self.v_next_ph: v_next,
                self.reward: total_reward,
                self.done: buf.done}
        _, v = sess.run([self.v_op, self.v], feed_dict=feed)
        v = np.reshape(v, [batch_size])

        # Update prime policy
        actions_1hot = util.convert_batch_action_int_to_1hot(buf.action, self.l_action)
        feed = {self.obs:  buf.obs,
                self.action_taken: actions_1hot,
                self.r_env: buf.r_env,
                self.incentive_received: buf.incentive_received,
                self.epsilon:  epsilon,
                self.v_next_ph: v_next,
                self.v_ph: v,
                self.done: buf.done}

        _ = sess.run(self.policy_op_prime, feed_dict=feed)
