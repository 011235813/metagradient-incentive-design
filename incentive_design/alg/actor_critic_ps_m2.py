"""Actor critic with parameter sharing for method 2.

This implementation allows an oracle incentive designer to
differentiate through the agents' policy update.

This does not support a factored action space,
This is meant to be used only for the agents, not the planner,
in Foundation.
"""

import numpy as np
import tensorflow as tf

from incentive_design.alg import actor_critic_ps
from incentive_design.alg import networks
from incentive_design.utils import util


class ActorCritic(actor_critic_ps.ActorCritic):

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
        assert not (dim_obs_tensor is None and dim_obs_flat is None)
        
        self.agent_id = agent_id
        self.agent_name = agent_name
        self.dim_action = dim_action
        self.dim_obs_flat = dim_obs_flat
        self.dim_obs_tensor = dim_obs_tensor
        self.tensor_and_flat = (self.dim_obs_flat and self.dim_obs_tensor)
        self.n_agents = n_agents
        self.nn = nn
        self.tax = tax
        self.utility = utility

        self.grad_clip = config.grad_clip
        self.entropy_coeff = config.entropy_coeff
        self.gamma = config.gamma
        self.lr_actor = config.lr_actor
        self.lr_v = config.lr_v
        self.tau = config.tau

        # use inherited method in actor_critic_ps.py
        self.create_networks()  
        self.create_critic_train_op()

        self.policy_new = PolicyNew

    def receive_designer(self, designer):
        """Stores the ID object for use in oracle metagradient."""
        self.designer = designer

    def create_policy_gradient_op(self):
        """Used by the ID for its own parameter update.

        Agents' policy parameter update is computed by create_update_op()
        and update_step(), not here.
        """
        self.total_endowment_coin = tf.placeholder(
            tf.float32, [None], 'total_endowment_coin')
        self.last_coin = tf.placeholder(tf.float32, [None], 'last_coin')
        self.escrow_coin = tf.placeholder(tf.float32, [None], 'escrow_coin')
        self.util_prev = tf.placeholder(tf.float32, [None], 'util_prev')
        self.inventory_coin = tf.placeholder(
            tf.float32, [None], 'inventory_coin')
        self.total_labor = tf.placeholder(tf.float32, [None], 'total_labor')
        self.curr_rate_max = tf.placeholder(tf.float32, [None], 'rate_max')
        self.enact_tax = tf.placeholder(tf.float32, [None], 'enact_tax')
        self.completions = tf.placeholder(tf.int32, [None], 'completions')
        income = self.total_endowment_coin - self.last_coin
        redistribution_minus_tax = self.tax.compute_tax_and_redistribution(
            income, self.designer.tax_function, self.curr_rate_max,
            self.inventory_coin)
        # Apply tax on inventory coins
        inventory_coin = self.inventory_coin + self.enact_tax * redistribution_minus_tax
        total_endowment_coin = inventory_coin + self.escrow_coin
        util_current = self.utility.isoelastic_coin_minus_labor(
            total_endowment_coin, self.total_labor, self.completions)
        r_total = util_current - self.util_prev

        # Check gradients: bypass tax and let tax_function
        # directly affect agents' rewards
        # Duplicate to get [time*n_agents, num_brackets]
        # tax_rates = tf.reshape(tf.tile(self.designer.tax_function,
        #                                [1, self.n_agents]),
        #                        [-1, 7])
        # r_total = tf.reduce_sum(tax_rates, axis=1)

        self.v_next_ph = tf.placeholder(tf.float32, [None], 'v_next_ph')
        self.v_ph = tf.placeholder(tf.float32, [None], 'v_ph')
        v_td_error = r_total + self.gamma*self.v_next_ph - self.v_ph

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

        # These grads are use by 1-step meta-gradient
        policy_grads = tf.gradients(self.loss, self.policy_params)
        self.policy_grads = ([tf.clip_by_norm(grad, self.grad_clip) for
                              grad in policy_grads] if self.grad_clip
                             else policy_grads)

    def create_update_op(self):
        """Set up TF graph for 1-step policy update."""
        
        # self.redistribution_minus_tax = tf.placeholder(
        #     tf.float32, [None], 'redistribution_minus_tax')
        # r_total = self.r_env + self.redistribution_minus_tax
        # No need to do any modification to reward because
        # self.reward placeholder will be fed with the complete reward
        # that includes taxation and difference of utilities
        v_td_error = self.reward + self.gamma*self.v_next_ph - self.v_ph

        # list of Tensors with shape [batch, 1]
        log_probs_taken = tf.log(tf.reduce_sum(tf.multiply(
            self.probs, self.action_taken), axis=1) + 1e-15)

        entropy = -tf.reduce_mean(tf.reduce_sum(tf.multiply(
            self.probs, self.log_probs), axis=1))

        policy_loss = -tf.reduce_mean(
            tf.multiply(log_probs_taken, v_td_error))
        loss = policy_loss - self.entropy_coeff * entropy

        grads = tf.gradients(loss, self.policy_params)
        grads = ([tf.clip_by_norm(grad, self.grad_clip) for grad in grads]
                 if self.grad_clip else grads)
        grads_and_vars = list(zip(grads, self.policy_params))
        policy_opt = tf.train.GradientDescentOptimizer(self.lr_actor)
        self.policy_op = policy_opt.apply_gradients(grads_and_vars)

    def train(self, sess, buf, epsilon):
        """Runs 1-step policy update.

        Args:
            sess: TF session
            buf: Buffer object
            epsilon: float
        """
        n_steps = len(buf.reward)

        (obs_flat, obs_tensor, action_1hot, action_mask, reward,
         obs_flat_next, obs_tensor_next, done) = self.reshape_batch(buf)

        # buf.incentive_recieved is a list over time of np.array
        # containing each agent's redistribution minus tax
        # redistribution_minus_tax = np.vstack(buf.incentive_received)
        # redistribution_minus_tax.shape = (n_steps * self.n_agents)

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

        # Update policy
        feed = {self.action_taken: action_1hot}
        feed[self.action_mask] = action_mask
        feed[self.reward] = reward
        # feed[self.redistribution_minus_tax] = redistribution_minus_tax
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


class PolicyNew(object):

    def __init__(self, params, dim_obs_tensor, dim_obs_flat,
                 dim_action, agent_name, nn):
        """Initalize a new policy network to store the updated policy params.

        Architecture matches networks.actor_image_vec

        Args:
            params: TF variables
            dim_obs_tensor: list of ints
            dim_obs_flat: [int]
            dim_action: int
            agent_name: str
            nn: ConfigDict
        """
        self.obs_tensor = tf.placeholder(
            tf.float32, [None]+list(dim_obs_tensor), 'obs_tensor_new')
        self.obs_flat = tf.placeholder(
                tf.float32, [None, dim_obs_flat[0]], 'obs_flat_new')
        self.action_taken = tf.placeholder(tf.float32, [None, dim_action],
                                           'action_taken_new')
        self.action_mask = tf.placeholder(
            tf.float32, [None, dim_action], 'action_mask')
        prefix = agent_name + '/policy/'

        h = self.obs_tensor
        with tf.variable_scope('policy_new'):
            # convolutional layer(s)
            for idx in range(1, len(nn.n_filters)+1):
                h = tf.nn.relu(
                    tf.nn.conv2d(h, params[prefix + ('conv_%d/w:0' % idx)],
                                 strides=[1, 1, 1, 1], padding='SAME',
                                 data_format='NHWC')
                    + params[prefix + ('conv_%d/b:0' % idx)])
            size = np.prod(h.get_shape().as_list()[1:])
            conv_flat = tf.reshape(h, [-1, size])
            
            conv_flat_dense = tf.nn.relu(
                tf.nn.xw_plus_b(conv_flat,
                                params[prefix + 'conv_flat_dense/kernel:0'],
                                params[prefix + 'conv_flat_dense/bias:0']))

            # Concatenate with flat part of obs
            h = tf.concat([conv_flat_dense, self.obs_flat], axis=1)

            # FC layers
            for idx in range(1, len(nn.n_fc)+1):
                h = tf.nn.relu(tf.nn.xw_plus_b(
                    h, params[prefix + ('fc_%d/kernel:0'%idx)],
                    params[prefix + ('fc_%d/bias:0'%idx)]))

            out = tf.nn.xw_plus_b(h, params[prefix + 'out/kernel:0'],
                                  params[prefix + 'out/bias:0'])

        probs = tf.nn.softmax(out)
        probs = tf.multiply(probs, self.action_mask)
        self.probs = probs / tf.reshape(tf.reduce_sum(probs, axis=1), [-1,1])
