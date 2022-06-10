"""Standard PPO incentive designer for SSD."""

import numpy as np
import tensorflow as tf

from scipy.special import expit
from incentive_design.alg import networks
from incentive_design.utils import util


class Agent(object):
    
    def __init__(self, designer_id, agent_name, config, dim_action,
                 dim_obs_tensor, n_agents, nn, dim_action_agents=2,
                 r_multiplier=2.0):
        """Initialization.

        Args:
            designer_id: int
            agent_name: str
            config: ConfigDict
            dim_action: int, number of output nodes of incentive function
            dim_obs_tensor: list of ints
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
        self.nn = nn
        self.n_agents = n_agents
        self.dim_action_agents = dim_action_agents
        self.r_multiplier = r_multiplier

        self.entropy_coeff = config.entropy_coeff
        self.gae_lambda = config.gae_lambda
        self.gamma = config.gamma
        self.grad_clip = config.grad_clip
        self.lr_actor = config.lr_actor
        self.lr_v = config.lr_v
        self.ppo_epsilon = config.ppo_epsilon
        self.tau = config.tau
        
        self.create_networks()
        self.create_critic_train_op()
        self.create_policy_gradient_op()
   
    def create_networks(self):
        """Placeholders and neural nets."""

        # [t, ...obs dim...]
        self.obs_tensor = tf.placeholder(
            tf.float32, [None]+list(self.dim_obs_tensor), 'obs_tensor')
        self.action_agents = tf.placeholder(
            tf.float32, [None, self.dim_action_agents * self.n_agents])

        actor_net = networks.incentive_ssd
        value_net = networks.vnet_ssd
            
        self.epsilon = tf.placeholder(tf.float32, [], 'epsilon')
        
        with tf.variable_scope(self.agent_name):
            with tf.variable_scope('policy'):
                # probs is list of [n, t, |A|],
                # length of list = number of action subspaces
                incentive_mean = actor_net(
                    self.obs_tensor, self.action_agents,
                    self.nn, self.dim_action, output_nonlinearity=None)
                stddev = tf.ones_like(incentive_mean)
                # Will be squashed by sigmoid later
                self.incentive_dist = tf.distributions.Normal(
                    loc=incentive_mean, scale=stddev)
                self.incentive_sample = self.incentive_dist.sample()
                    
            with tf.variable_scope('v_main'):
                self.v = value_net(self.obs_tensor, self.nn)
            
            with tf.variable_scope('v_target'):
                self.v_target = value_net(self.obs_tensor, self.nn)

        self.policy_vars = tf.trainable_variables(self.agent_name+'/policy')
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
    
    def run_actor(self, obs_tensor, action_agents, sess):
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
        incentive_sample = sess.run(self.incentive_sample, feed_dict=feed)
        incentive_sample = incentive_sample.flatten()
        incentive = self.r_multiplier * expit(incentive_sample)

        return incentive, incentive_sample
    
    def create_critic_train_op(self):
        
        self.v_target_next = tf.placeholder(tf.float32, [None], 'v_target_next')
        self.reward = tf.placeholder(tf.float32, [None], 'reward')
        td_target = self.reward + self.gamma * self.v_target_next
        self.loss_v = tf.reduce_mean(tf.math.square(
            td_target - tf.squeeze(self.v)))
        self.v_opt = tf.train.AdamOptimizer(self.lr_v)
        self.v_op = self.v_opt.minimize(self.loss_v)
    
    def create_policy_gradient_op(self):
        self.advantages = tf.placeholder(tf.float32, [None], 'advantages')

        self.r_sampled = tf.placeholder(tf.float32, [None, self.dim_action],
                                        'r_sampled')
        self.log_probs_incentive = self.incentive_dist.log_prob(
            self.r_sampled)
        # Account for the change of variables due to passing through
        # sigmoid and scaling
        # p(y) = p(x) |det dx/dy | = p(x) |det 1/(dy/dx)|
        sigmoid_derivative = tf.math.sigmoid(self.r_sampled) * (
            1 - tf.math.sigmoid(self.r_sampled)) * self.r_multiplier
        log_probs = (
            tf.reduce_sum(self.log_probs_incentive, axis=1) -
            tf.reduce_sum(tf.math.log(sigmoid_derivative), axis=1))

        log_probs_old = tf.stop_gradient(log_probs)
        
        ratio = tf.exp(log_probs - log_probs_old)
        surr_1 = tf.multiply(ratio, self.advantages)
        surr_2 = tf.multiply(tf.clip_by_value(
            ratio, 1.0-self.ppo_epsilon, 1.0+self.ppo_epsilon),
                             self.advantages)
        self.policy_loss = - tf.reduce_mean(tf.minimum(surr_1, surr_2))

        # Compute entropy
        # p(y) = p(x) |det dx/dy | = p(x) |det 1/(dy/dx)|
        #      = p(x) 1/| prod_{i=1}^d dy_i/dx_i|
        probs = self.incentive_dist.prob(self.r_sampled) # [t,|A|]
        # [t]
        denominator = tf.reduce_prod(tf.math.log(sigmoid_derivative), axis=1)
        denominator = tf.expand_dims(denominator, axis=1) # [t,1]
        # element-wise division of each row by scalar
        probs = probs / denominator # [t, |A|]
        # log p(y) = log p(x) - log \prod_{i=1}^d dy_i/dx_i
        log_probs_subtract = tf.reduce_sum(tf.math.log(sigmoid_derivative),
                                           axis=1)
        log_probs_subtract = tf.expand_dims(log_probs_subtract, axis=1)#[t,1]
        # element-wise subtraction of each row by scalar
        log_probs = self.log_probs_incentive - log_probs_subtract # [t,|A|]
        entropy = -tf.reduce_sum(tf.multiply(probs, log_probs))
        self.loss = self.policy_loss - self.entropy_coeff * entropy

        policy_opt = tf.train.AdamOptimizer(self.lr_actor)
        if self.grad_clip:
            grad_var = policy_opt.compute_gradients(
                self.loss, var_list=self.policy_vars)
            grad_var = [(tf.clip_by_norm(tup[0], self.grad_clip) if (not tup[0] is None) else tup[0], tup[1]) for tup in grad_var]
            self.policy_op = policy_opt.apply_gradients(grad_var)
        else:
            self.policy_op = policy_opt.minimize(
                self.loss, var_list=self.policy_vars)
    
    def train(self, sess, buf, epsilon):
        """Training step for policy and value function

        Args:
            sess: TF session
            buf: Buffer object
            epsilon: float
        """
        # Update value network
        obs_tensor_next = np.array(buf.obs_tensor_next) # [t,L,W,C]
        feed = {self.obs_tensor: obs_tensor_next}
        v_target_next, v_next = sess.run([self.v_target, self.v],
                                         feed_dict=feed) # [t,1]

        feed[self.v_target_next] = np.squeeze(v_target_next)
        feed[self.reward] = np.array(buf.reward) # [t]
        feed[self.obs_tensor] = np.array(buf.obs_tensor) # [t,L,W,C]
        _, v = sess.run([self.v_op, self.v], feed_dict=feed) # [t,1]
        v = np.squeeze(v) # V(s_0),....,V(s_{t-1})
        # Update target network
        sess.run(self.list_update_v_ops)
    
        # Update policy network
        # Compute advantages
        v_last = sess.run(self.v, feed_dict={
            self.obs_tensor: [buf.obs_tensor_next[-1]]})
        v_last = np.reshape(v_last, [-1])
        v_rollouts = np.concatenate((v, v_last))
        advantages = util.compute_advantages(
            buf.reward, v_rollouts, self.gamma, self.gae_lambda)
        feed[self.advantages] = advantages

        feed[self.action_agents] = util.convert_batch_actions_int_to_1hot(
            buf.action_agents, self.dim_action_agents)
        feed[self.r_sampled] = buf.r_sample
        
        _ = sess.run(self.policy_op, feed_dict=feed)
