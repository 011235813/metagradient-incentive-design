"""A policy gradient agent.

This implementation allows an oracle incentive designer to conduct 1-step meta-gradient.
Maintains a prime policy, which gets updated using the first trajectory.
The prime policy is used to generate the validation trajectory.
"""
import sys

import numpy as np
import tensorflow as tf

import networks

from incentive_design.utils import util


class Agent(object):

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
        self.agent_id = agent_id
        self.agent_name = 'agent_%d' % self.agent_id
        self.alg_name = 'pg'
        self.l_action = l_action
        self.l_obs = l_obs
        self.n_agents = n_agents
        self.nn = nn
        self.r_multiplier = r_multiplier

        self.entropy_coeff = config.entropy_coeff
        self.gamma = config.gamma
        self.lr_actor = config.lr_actor

        self.create_networks()
        if self.nn.use_single_layer:
            self.policy_new = PolicyNew1Layer
        else:
            self.policy_new = PolicyNew
        
    def create_networks(self):
        """Creates neural network part of the TF graph."""
        self.obs = tf.placeholder(tf.float32, [None, self.l_obs], 'l_obs')
        self.epsilon = tf.placeholder(tf.float32, None, 'epsilon')

        with tf.variable_scope(self.agent_name):
            with tf.variable_scope('policy_main'):
                with tf.variable_scope('policy'):
                    if self.nn.use_single_layer:
                        probs = networks.actor_single_layer(
                            self.obs, self.l_obs, self.l_action)
                    else:
                        probs = networks.actor_mlp(
                            self.obs, self.l_action, self.nn)
                self.probs = (1 - self.epsilon) * probs + self.epsilon / self.l_action
                self.log_probs = tf.log(self.probs)
                self.action_samples = tf.multinomial(self.log_probs, 1)

            # Explanation: a set of 'prime' policy parameters is needed to conduct 
            # \theta' <-- \theta' + \Delta \theta'
            # while the main \theta are unchanged.
            # \theta' is used to collect the validation trajectory
            # for use in meta-gradient. After a meta-gradient step,
            # \theta' is copied into the main \theta
            with tf.variable_scope('policy_prime'):
                with tf.variable_scope('policy'):
                    if self.nn.use_single_layer:
                        probs = networks.actor_single_layer(
                            self.obs, self.l_obs, self.l_action)
                    else:
                        probs = networks.actor_mlp(
                            self.obs, self.l_action, self.nn)
                self.probs_prime = (1-self.epsilon)*probs + self.epsilon/self.l_action
                self.log_probs_prime = tf.log(self.probs_prime)
                self.action_samples_prime = tf.multinomial(self.log_probs_prime, 1)

        self.policy_params = tf.trainable_variables(
            self.agent_name + '/policy_main/policy')
        self.policy_prime_params = tf.trainable_variables(
            self.agent_name + '/policy_prime/policy')

        self.list_copy_main_to_prime_ops = []
        for idx, var in enumerate(self.policy_prime_params):
            self.list_copy_main_to_prime_ops.append(
                var.assign(self.policy_params[idx]))

        self.list_copy_prime_to_main_ops = []
        for idx, var in enumerate(self.policy_params):
            self.list_copy_prime_to_main_ops.append(
                var.assign(self.policy_prime_params[idx]))

    def receive_designer(self, designer):
        """Stores the incentive designer object for use in oracle meta-gradient."""
        self.designer = designer

    def run_actor(self, obs, sess, epsilon, prime=False):
        """Gets action given observation.

        Args:
            obs: np.array
            sess: TF session
            epsilon: float
            prime: Bool, whether to run main or prime policy

        Returns: integer
        """
        feed = {self.obs: np.array([obs]), self.epsilon: epsilon}
        if prime:
            action = sess.run(self.action_samples_prime, feed_dict=feed)[0][0]
        else:
            action = sess.run(self.action_samples, feed_dict=feed)[0][0]
        return action

    def create_policy_gradient_op(self):
        """This will be used by the incentive designer for 1-step meta-gradient."""
        # Reward defined by the environment
        self.r_env = tf.placeholder(tf.float32, [None], 'r_ext')

        self.action_taken = tf.placeholder(tf.float32, [None, self.l_action],
                                           'action_taken')

        if self.designer.output_type == 'agent':
            # agent_id is in [1, n_agents], so need to -1
            # this_agent_1hot = tf.one_hot(indices=self.agent_id-1, depth=self.n_agents)
            index_array = tf.one_hot(indices=self.agent_id-1, depth=self.n_agents)
        else:
            index_array = self.action_taken
        # Total reward
        r_total = self.r_env + self.r_multiplier * tf.reduce_sum(
            tf.multiply(self.designer.incentive_function,
                        index_array), axis=1)

        self.ones = tf.placeholder(tf.float32, [None], 'ones')
        self.gamma_prod = tf.math.cumprod(self.ones * self.gamma)
        returns = tf.reverse(tf.math.cumsum(
            tf.reverse(r_total * self.gamma_prod, axis=[0])), axis=[0])
        returns = returns / self.gamma_prod


        self.log_probs_taken = tf.log(tf.reduce_sum(
            tf.multiply(self.probs, self.action_taken), axis=1) + 1e-15)

        self.entropy = -tf.reduce_sum(tf.multiply(self.probs, self.log_probs))

        self.policy_loss = -tf.reduce_sum(
            tf.multiply(self.log_probs_taken, returns))
        self.loss = self.policy_loss - self.entropy_coeff * self.entropy

        # Just need these grads to perform 1-step meta-gradient.
        # Do not need a tf.optimizer opt for these main parameter gradients
        # because the prime parameter will already have been updated
        # with the same training data and will be copied into main params.
        self.policy_grads = tf.gradients(self.loss, self.policy_params)

    def create_update_op(self):
        """Set up TF graph for 1-step prime parameter update."""
        self.incentive_received = tf.placeholder(tf.float32, [None], 'incentive_received')
        r_total = self.r_env + self.incentive_received
        returns_val = tf.reverse(tf.math.cumsum(
            tf.reverse(r_total * self.gamma_prod, axis=[0])), axis=[0])
        returns_val = returns_val / self.gamma_prod

        log_probs_taken = tf.log(tf.reduce_sum(
            tf.multiply(self.probs_prime, self.action_taken), axis=1) + 1e-15)
        entropy = -tf.reduce_sum(
            tf.multiply(self.probs_prime, self.log_probs_prime))
        policy_loss = -tf.reduce_sum(
            tf.multiply(log_probs_taken, returns_val))
        loss = policy_loss - self.entropy_coeff * entropy

        policy_opt_prime = tf.train.GradientDescentOptimizer(self.lr_actor)
        self.policy_op_prime = policy_opt_prime.minimize(loss)

    def update(self, sess, buf, epsilon):
        """Runs 1-step prime parameter update."""
        sess.run(self.list_copy_main_to_prime_ops)

        n_steps = len(buf.obs)
        actions_1hot = util.convert_batch_action_int_to_1hot(buf.action, self.l_action)
        ones = np.ones(n_steps)
        feed = {self.obs: buf.obs,
                self.action_taken: actions_1hot,
                self.r_env: buf.r_env,
                self.ones: ones,
                self.epsilon: epsilon}

        feed[self.incentive_received] = buf.incentive_received

        _ = sess.run(self.policy_op_prime, feed_dict=feed)

    def update_main(self, sess):
        """Recall that prime parameters have already been updated
        immediately before collecting the validation trajectory for
        meta-gradient (using the main params)."""
        sess.run(self.list_copy_prime_to_main_ops)

        
class PolicyNew(object):
    """A replicate of the agent's policy.

    Used during the 1-step meta-gradient to store \hat{\theta} and
    retain in-graph dependency on \eta.
    """
    def __init__(self, params, l_obs, l_action, agent_name):
        self.obs = tf.placeholder(tf.float32, [None, l_obs], 'obs_new')
        self.action_taken = tf.placeholder(tf.float32, [None, l_action],
                                           'action_taken')
        prefix = agent_name + '/policy_main/policy/'
        with tf.variable_scope('policy_new'):
            h1 = tf.nn.relu(
                tf.nn.xw_plus_b(self.obs, params[prefix + 'actor_h1/kernel:0'],
                                params[prefix + 'actor_h1/bias:0']))
            h2 = tf.nn.relu(
                tf.nn.xw_plus_b(h1, params[prefix + 'actor_h2/kernel:0'],
                                params[prefix + 'actor_h2/bias:0']))
            out = tf.nn.xw_plus_b(h2, params[prefix + 'actor_out/kernel:0'],
                                params[prefix + 'actor_out/bias:0'])
        self.probs = tf.nn.softmax(out)


class PolicyNew1Layer(object):
    """A replicate of the agent's policy.

    Used during the 1-step meta-gradient to store \hat{\theta} and
    retain in-graph dependency on \eta.
    """
    def __init__(self, params, l_obs, l_action, agent_name):
        self.obs = tf.placeholder(tf.float32, [None, l_obs], 'obs_new')
        self.action_taken = tf.placeholder(tf.float32, [None, l_action],
                                           'action_taken')
        prefix = agent_name + '/policy_main/policy/'
        with tf.variable_scope('policy_new'):
            # theta = tf.Variable(params[prefix + 'theta:0'])
            out = tf.nn.xw_plus_b(self.obs, params[prefix + 'theta:0'],
                                  tf.zeros(l_action))

        self.probs = tf.nn.softmax(out)
