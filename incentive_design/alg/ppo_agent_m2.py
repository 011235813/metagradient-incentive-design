import numpy as np
import tensorflow as tf

import networks
from incentive_design.utils import util
from incentive_design.alg import ppo_agent

class Agent(ppo_agent.Agent):
    
    def __init__(self, agent_id, agent_name, config, dim_action,
                 dim_obs_flat, dim_obs_tensor, n_agents, objective, nn,
                 tax, utility):
        """Initialization.

        Args:
            agent_id: int
            agent_name: string
            config: ConfigDict
            dim_action: list of ints for each action subspace
            dim_obs_flat: int, if obs has a flat part, else None
            dim_obs_tensor: list, if obs has an image part, else None
            nn: ConfigDict
        """
        assert not (dim_obs_tensor is None and dim_obs_flat is None)
        
        self.agent_id = agent_id
        self.agent_name = agent_name
        self.dim_action = dim_action
        self.dim_obs_flat = dim_obs_flat
        self.dim_obs_tensor = dim_obs_tensor
        self.tensor_and_flat = (self.dim_obs_flat and self.dim_obs_tensor)
        self.nn = nn
        self.n_agents = n_agents
        self.objective = objective
        self.tax = tax
        self.utility = utility

        self.grad_clip = config.grad_clip
        self.entropy_coeff = config.entropy_coeff
        self.gamma = config.gamma
        self.lr_actor = config.lr_actor
        self.lr_v = config.lr_v
        self.tau = config.tau
        self.gae_lambda = config.gae_lambda
        self.ppo_epsilon = config.ppo_epsilon
        self.entropy_coeff = config.entropy_coeff
        
        self.create_networks()
        self.create_critic_train_op()
        
        self.policy_new = PolicyNew
    
    def reshape_batch(self, buf=None):
        """Reshapes buffer to match ppo_agent
        
        Outputs: obs_flat, obs_tensor, reward, obs_flat_next, obs_tensor_next
                 Shapes are [n,t, ...object dims...]
        """
        if not buf:
            buf = self.buffer
            
        obs_tensor = np.array(buf.obs_tensor) # [t,n,L,W,C]
        obs_flat = np.array(buf.obs_flat)

        # If obs_tensors don't have dimension for n_agents:
        if len(obs_tensor.shape) == 4:  # should never happen for m1/m2
            obs_tensor = np.expand_dims(obs_tensor, axis=1)
            obs_flat = np.expand_dims(obs_flat, axis=1)

        obs_tensor = np.swapaxes(obs_tensor, 0, 1) # [n,t,L,W,C]
        obs_flat = np.swapaxes(obs_flat, 0, 1)

        obs_tensor_next = np.array(buf.obs_tensor_next) # [t,n,L,W,C]
        obs_flat_next = np.array(buf.obs_flat_next)

        # If obs_tensors don't have dimension for n_agents:
        if len(obs_tensor_next.shape) == 4:
            obs_tensor_next = np.expand_dims(obs_tensor_next, axis=1)
            obs_flat_next = np.expand_dims(obs_flat_next, axis=1)

        obs_tensor_next = np.swapaxes(obs_tensor_next, 0, 1) # [n,t,L,W,C]
        obs_flat_next = np.swapaxes(obs_flat_next, 0, 1)
        
        reward = np.array(buf.reward) # [t,n]
        
        # If reward doesn't have dimension for n_agents:
        if len(reward.shape) == 1: # should never happen for m1/m2
            reward = np.expand_dims(reward, axis=1)
            
        reward = np.swapaxes(reward, 0, 1)
        reward = np.expand_dims(reward, axis=2) # [n,t,1]
    
        return (obs_flat, obs_tensor, reward, obs_flat_next, obs_tensor_next)
    
    def receive_designer(self, designer):
        """Stores the ID object for use in oracle metagradient."""
        self.designer = designer
   
    def create_policy_gradient_op(self):
        """Used by the ID for its own parameter update.

        Agents' policy parameter update is computed by create_update_op()
        and update_step(), not here.
        """
        # Shape is assumed to be [time*n_agents]
        # This means [agents' values at t_1,...,agents' values at t_T]
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
            income, self.designer.tax_function_duplicate, self.curr_rate_max,
            self.inventory_coin)
        # Apply tax on inventory coins
        inventory_coin = self.inventory_coin + self.enact_tax * redistribution_minus_tax
        total_endowment_coin = inventory_coin + self.escrow_coin
        util_current = self.utility.isoelastic_coin_minus_labor(
            total_endowment_coin, self.total_labor, self.completions)
        r_total = util_current - self.util_prev
        r_total = tf.reshape(r_total, [-1, self.n_agents])  # [t, n]
        r_total = tf.transpose(r_total, [1, 0])  # [n, t]
        r_total = tf.reshape(r_total, [self.n_agents, -1, 1])  # [n, t, 1]
        
        self.v_ph = tf.placeholder(tf.float32, [self.n_agents, None, 1], 'v')
        self.v_next_ph = tf.placeholder(tf.float32, [self.n_agents, None, 1], 'v_next')
        
        # Calculate advantages here so that gradients propogate through self.designer.tax_function
        timesteps = tf.shape(r_total)[1]
        gae = r_total + self.gamma * self.v_next_ph - self.v_ph
        
        """
        # Build an array with 1 on the diagonal and horizon coefficents on the vertical axis
        # [1,   0,   0,   0]
        # [a,   1,   0,   0]
        # [a^2, a,   1,   0]
        # [a^3, a^2, a,   1]
        # where a = self.gae_lambda * self.gamma
        # Using np
        coeff = np.fromfunction(lambda i, j, k: (self.gae_lambda * self.gamma) ** (j - k), (self.n_agents, timesteps, timesteps))
        coeff = np.tril(coeff)
        """

        j = tf.cast(tf.range(timesteps), tf.float32)
        j = tf.tile(j, [timesteps])
        j = tf.reshape(j, [timesteps, timesteps])
        j = tf.transpose(j)

        k = tf.cast(tf.range(timesteps), tf.float32)
        k = tf.tile(k, [timesteps])
        k = tf.reshape(k, [timesteps, timesteps])

        powers = j - k
        powers = tf.expand_dims(powers, axis=0)
        powers = tf.tile(powers, [self.n_agents, 1, 1])
        
        coeff = tf.fill([self.n_agents, timesteps, timesteps], self.gae_lambda * self.gamma)
        coeff = tf.math.pow(coeff, powers)
        # Get lower triangular part
        # [n_agents, time, time]
        coeff = tf.matrix_band_part(coeff, -1, 0)

        # [n_agents, time, time]
        gae = tf.tile(gae, [1, 1, timesteps])
        
        # Element-wise multiplication, then for each agent dimension,
        # sum over the first time dimension (i.e., along a column in picture above)
        # [n_agents, 1, time]
        advantages = tf.reduce_sum(tf.multiply(coeff, gae), axis=1, keepdims=True)
        # [n_agents, time, 1]
        advantages = tf.transpose(advantages, [0, 2, 1])
        
        self.old_log_probs = [tf.placeholder(
            tf.float32, [self.n_agents, None, dim], 'log_probs')
                              for dim in self.dim_action]
        self.action_taken = [tf.placeholder(
            tf.float32, [self.n_agents, None, dim])
                             for dim in self.dim_action]
        
        # new_probs is a list of [n_agents, timesteps, |A|].
        # length of list should be 1 = #subspaces
        new_probs = self.probs
        batch_size = tf.shape(advantages)[1]
        old_log_probs = tf.zeros([self.n_agents, batch_size, 1])
        new_log_probs = tf.zeros([self.n_agents, batch_size, 1])
        
        old_prob_action = tf.multiply(self.old_log_probs[0], self.action_taken[0])
        old_log_probs += tf.reduce_sum(old_prob_action, axis=2, keepdims=True)

        new_prob_action = tf.multiply(new_probs[0], self.action_taken[0])
        new_prob_action = tf.reduce_sum(new_prob_action, axis=2, keepdims=True)
        new_log_probs += tf.log(new_prob_action + 1e-15)
        self.new_log_probs = new_log_probs
        
        ratio = tf.exp(self.new_log_probs - old_log_probs)
        
        # Ratio in shape: (self.n_agents, None, 1)
        if self.objective == "clipped_surrogate":            
            
            entropy = tf.zeros([self.n_agents, batch_size, 1])
            
            # Calculate the cartesian product of probs for all action dims
            cart_prod_probs = self.probs[0]
            self.cart_prod_probs = cart_prod_probs
            
            # Not implemented: combinatorial action space is too large
            if len(self.dim_action) == 1:
                mult_probs = tf.multiply(self.cart_prod_probs, tf.log(self.cart_prod_probs))
                entropy = - tf.reduce_sum(mult_probs, axis=2, keepdims=True)

                    
            surr_1 = tf.multiply(ratio, advantages)
            surr_2 = tf.multiply(tf.clip_by_value(
                ratio, 1 - self.ppo_epsilon, 1 + self.ppo_epsilon),
                                 advantages)
            self.loss = tf.math.reduce_mean(
                - tf.math.minimum(surr_1, surr_2)
                - self.entropy_coeff * entropy)
            # These grads are use by 1-step meta-gradient
            self.policy_grads = tf.gradients(self.loss, self.policy_params)
    
    def create_update_op(self):
        self.advantages = tf.placeholder(tf.float32, [self.n_agents, None, 1], 'advantages')
        
        # new_probs is a list of [n_agents, timesteps, |A|]
        # length of list = # action subspaces = 1
        new_probs = self.probs
        batch_size = tf.shape(self.advantages)[1]
        old_log_probs = tf.zeros([self.n_agents, batch_size, 1])
        new_log_probs = tf.zeros([self.n_agents, batch_size, 1])
        
        # Calculate the log_prob of taking the action
        for idx in range(len(self.dim_action)):
            old_prob_action = tf.multiply(self.old_log_probs[idx], self.action_taken[idx])
            old_log_probs += tf.reduce_sum(old_prob_action, axis=2, keepdims=True)

            new_prob_action = tf.multiply(new_probs[idx], self.action_taken[idx])
            new_prob_action = tf.reduce_sum(new_prob_action, axis=2, keepdims=True)
            new_log_probs += tf.log(new_prob_action + 1e-15)
            
        ratio = tf.exp(new_log_probs - old_log_probs)
        
        # Ratio in shape: (self.n_agents, None, 1)
        if self.objective == "clipped_surrogate":            
            
            entropy = tf.zeros([self.n_agents, batch_size, 1])
            
            # Calculate the cartesian product of probs for all action dims
            cart_prod_probs = self.probs[0]
            
            # Not implemented: combinatorial action space is too large
            if len(self.dim_action) == 1:
                mult_probs = tf.multiply(cart_prod_probs, tf.log(cart_prod_probs))
                entropy = - tf.reduce_sum(mult_probs, axis=2, keepdims=True)

            surr_1 = tf.multiply(ratio, self.advantages)
            surr_2 = tf.multiply(tf.clip_by_value(
                ratio, 1 - self.ppo_epsilon, 1 + self.ppo_epsilon),
                                 self.advantages)
            self.loss_actor = tf.math.reduce_mean(
                - tf.math.minimum(surr_1, surr_2)
                - self.entropy_coeff * entropy)
            self.actor_opt = tf.train.AdamOptimizer(self.lr_actor)
            self.actor_op = self.actor_opt.minimize(self.loss_actor)

    def train(self, sess, buf, epsilon):
        """Training step for policy and value function

        Args:
            sess: TF session
            buf: Buffer object
            epsilon: float
        """
        # Input shape is (n_agents, timesteps, obs_shape)
        # Calculate values
        if self.tensor_and_flat:
            obs_tensor_next = np.array(buf.obs_tensor_next)
            obs_flat_next = np.array(buf.obs_flat_next)
            
            # If obs_tensors don't have dimension for n_agents:
            if len(obs_tensor_next.shape) == 4:
                obs_tensor_next = np.expand_dims(obs_tensor_next, axis=1)
                obs_flat_next = np.expand_dims(obs_flat_next, axis=1)
            
            obs_tensor_next = np.swapaxes(obs_tensor_next, 0, 1)
            obs_flat_next = np.swapaxes(obs_flat_next, 0, 1)
            
            feed = {self.obs_tensor: obs_tensor_next,
                    self.obs_flat: obs_flat_next}
        else:
            raise NotImplementedError
        
        # Initialize the v_main and v_target LSTMCells
        feed[self.new_v_main_state] = np.zeros([2, self.n_agents, self.nn.n_lstm])
        feed[self.new_v_target_state] = np.zeros([2, self.n_agents, self.nn.n_lstm])
        
        v_target_next, v_next = sess.run([self.v_target, self.v],
                                         feed_dict=feed)
        
        # Update value network
        feed[self.v_target_next] = v_target_next
        
        reward = np.array(buf.reward)
        
        # If reward doesn't have dimension for n_agents:
        if len(reward.shape) == 1:
            reward = np.expand_dims(reward, axis=1)
            
        reward = np.swapaxes(reward, 0, 1)
        reward = np.expand_dims(reward, axis=2)
        feed[self.reward] = reward
        
        if self.tensor_and_flat:
            obs_tensor = np.array(buf.obs_tensor)
            obs_flat = np.array(buf.obs_flat)
            
            # If obs_tensors don't have dimension for n_agents:
            if len(obs_tensor.shape) == 4:
                obs_tensor = np.expand_dims(obs_tensor, axis=1)
                obs_flat = np.expand_dims(obs_flat, axis=1)
            
            obs_tensor = np.swapaxes(obs_tensor, 0, 1)
            obs_flat = np.swapaxes(obs_flat, 0, 1)
            
            feed[self.obs_tensor] = obs_tensor
            feed[self.obs_flat] = obs_flat
        else:
            raise NotImplementedError
            
        v = sess.run(self.v, feed_dict=feed)
       
        _ = sess.run(self.v_op, feed_dict=feed)
        
        # Update target network
        sess.run(self.list_update_v_ops)
    
        # Update policy network
        
        # Feed advantages
        self.feed_advantages(buf, v, v_next, feed)
        
        feed[self.epsilon] = epsilon
        
        self.feed_actions(feed,
            buf.log_probs,
            buf.action,
            buf.action_mask
        )
        
        initial_actor_state = np.zeros((2, self.n_agents, self.nn.n_lstm))
        feed[self.new_actor_state] = initial_actor_state
        _ = sess.run(self.actor_op, feed_dict=feed)
        
    def get_minibatch(self, feed, idx):
        feed_idx = {}
        for key, value in feed.items():
            if key == self.obs_tensor or key == self.obs_flat or key == self.v or key == self.v_next:
                feed_idx[key] = value[:, idx : idx + self.nn.max_timesteps, :]
            else:
                feed_idx[key] = value
        return feed_idx 
    
    
class PolicyNew(object):
    
    def __init__(self, params, dim_obs_tensor, dim_obs_flat,
                 dim_action, agent_name, nn, n_agents):
        """
        dim_action: list of ints for each action subspace.
                    For parameter-sharing agents, length should be 1.
        """
        self.n_agents = n_agents
        self.dim_action = dim_action
        self.obs_tensor = tf.placeholder(
                tf.float32, [self.n_agents, None]+list(dim_obs_tensor), 'obs_tensor')
        self.obs_flat = tf.placeholder(
            tf.float32, [self.n_agents, None, dim_obs_flat[0]], 'obs_flat')
        self.action_taken_dim = [tf.placeholder(tf.float32, [self.n_agents, None, dim])
                             for dim in dim_action]
        
        self.action_masks = [tf.placeholder(
            tf.float32, [self.n_agents, None, dim], 'action_mask_%d'%idx)
                             for idx, dim in enumerate(dim_action)]
        
        self.new_actor_state = tf.placeholder(
                tf.float32, [2, self.n_agents, nn.n_lstm], 'new_actor_state')
        prefix = agent_name + '/policy/'
        
        h = self.obs_tensor
        h = tf.reshape(h, [-1] + list(dim_obs_tensor)) # [n*t,L,W,C]
        
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
            obs_flat = tf.reshape(self.obs_flat, [-1, dim_obs_flat[0]])
            h = tf.concat([conv_flat_dense, obs_flat], axis=1)

            # FC layers
            for idx in range(1, len(nn.n_fc)+1):
                h = tf.nn.relu(tf.nn.xw_plus_b(
                    h, params[prefix + ('fc_%d/kernel:0'%idx)],
                    params[prefix + ('fc_%d/bias:0'%idx)]))
            

            # [n,t,last hidden layer size]
            h = tf.reshape(h, [self.n_agents, -1] + h.get_shape().as_list()[1:])
            
            if nn.use_lstm_actor:
                kernel_initializer = constant_initializer(params[prefix + 'rnn/lstm_cell/kernel:0'])
                bias_initializer = constant_initializer(params[prefix + 'rnn/lstm_cell/bias:0'])

                lstm_cell = LSTMCell(nn.n_lstm, kernel_initializer, bias_initializer)
                lstm_state = tf.nn.rnn_cell.LSTMStateTuple(self.new_actor_state[0], self.new_actor_state[1])

                h, state = tf.nn.dynamic_rnn(cell=lstm_cell, inputs=h, 
                                             dtype=tf.float32, 
                                             initial_state=lstm_state)
            
            # [n*t,...]
            h = tf.reshape(h, [-1] + h.get_shape().as_list()[2:])
            out = tf.nn.xw_plus_b(h, params[prefix + 'out_0/kernel:0'],
                                  params[prefix + 'out_0/bias:0'])
            # [n,t,|A|]
            out = tf.reshape(out, [self.n_agents, -1] + out.get_shape().as_list()[1:])
        
        probs = tf.nn.softmax(out)
        probs = tf.multiply(probs, self.action_masks[0])
        probs = probs / tf.reshape(tf.reduce_sum(probs, axis=2), [self.n_agents, -1, 1])
        probs = probs + 1e-15 * tf.one_hot(0, dim_action[0]) # [n,t,|A|]

        # Reshape probs and action_taken to match id_foundation.Metagrad1Step
        # id_foundation expects [t*n, |A|].
        # t*n means ordering is: all agents at t_1,...,all agents at t_T
        probs = tf.transpose(probs, [1, 0, 2]) # [t,n,|A|]
        self.probs = tf.reshape(probs, [-1, dim_action[0]]) # [t*n, |A|]
        # self.action_taken_dim[0] is [n,t,|A|]
        action_taken = tf.transpose(self.action_taken_dim[0], [1,0,2])
        self.action_taken = tf.reshape(action_taken, [-1, dim_action[0]]) # [t*n, |A|]
                                       
    def feed_actions(self, feed, buf_action, buf_action_mask):
        # buf_action is a list of [n, 1], length of list = t
        actions = np.array(buf_action)  # [t, n, 1]

        if len(actions.shape) == 3:
            actions = np.expand_dims(actions, axis=1) # [t, 1, n, 1]
        actions = np.swapaxes(actions, 0, 1) # [1, t, n, 1]
        actions = np.swapaxes(actions, 1, 2) # [1, n, t, 1]
        # actions has shape: (dim_action, n_agents, timesteps, 1)
        for idx, action_taken in enumerate(self.action_taken_dim):
            action = []
            for i in range(self.n_agents):
                action_1hot = util.convert_batch_actions_int_to_1hot(actions[idx][i], self.dim_action[idx]) # [t, |A|]
                action.append(action_1hot)
            feed[action_taken] = np.array(action) # [n, t, |A|]
        
        # buf_action_mask is a list over time of list over agents of [|A|]
        action_mask = buf_action_mask
        
        for idx in range(len(self.dim_action)):
            if len(self.dim_action) == 1:
                dim_action_mask = np.array(action_mask) # [t,n,|A|]
                dim_action_mask = np.swapaxes(dim_action_mask, 0, 1)
                feed[self.action_masks[idx]] = dim_action_mask # [n,t,|A|]
            else:
                dim_action_mask = []
                for step in range(len(action_mask)):
                    dim_action_mask.append(action_mask[step][idx])
                dim_action_mask = np.array(dim_action_mask)
                
                # action_mask has no dimension for n_agents:
                if len(dim_action_mask.shape) == 2:
                    dim_action_mask = np.expand_dims(dim_action_mask, axis=1)
                
                dim_action_mask = np.swapaxes(dim_action_mask, 0, 1)
                feed[self.action_masks[idx]] = dim_action_mask
        
"""Extends tf.constant_initializer to accept values of type Tensor"""
class constant_initializer(tf.constant_initializer):
    def __init__(self, value, dtype=tf.float32, verify_shape=False):
        self.value = value
        self.dtype = dtype
        self._verify_shape = verify_shape
    
    def __call__(self, shape, dtype=None, partition_info=None, verify_shape=None):
        return self.value
                                       
"""Extends bias_intialization to tf.nn.rnn_cell.LSTMCell"""
class LSTMCell(tf.nn.rnn_cell.LSTMCell):
    
    def __init__(self, num_units, kernel_initializer, bias_initializer):
        super().__init__(num_units, initializer=kernel_initializer)
            
        self._bias_initializer = bias_initializer
    
    def build(self, inputs_shape):
        def fixed_size_partitioner(num_shards, axis=0):
            """Partitioner to specify a fixed number of shards along given axis.
            Args:
            num_shards: `int`, number of shards to partition variable.
            axis: `int`, axis to partition on.
            Returns:
            A partition function usable as the `partitioner` argument to
            `variable_scope`, `get_variable`, and `get_partitioned_variable_list`.
            """
            def _partitioner(shape, **unused_args):
                partitions_list = [1] * len(shape)
                partitions_list[axis] = min(num_shards, shape[axis].value)
                return partitions_list
            
            return _partitioner
    
        if inputs_shape[1].value is None:
            raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                           % inputs_shape)

        input_depth = inputs_shape[1].value
        h_depth = self._num_units if self._num_proj is None else self._num_proj
        maybe_partitioner = (
            fixed_size_partitioner(self._num_unit_shards)
            if self._num_unit_shards is not None
            else None)
        self._kernel = self.add_variable(
            "kernel",
            shape=[input_depth + h_depth, 4 * self._num_units],
            initializer=self._initializer,
            partitioner=maybe_partitioner)
        self._bias = self.add_variable(
            "bias",
            shape=[4 * self._num_units],
            initializer=self._bias_initializer)
        if self._use_peepholes:
            self._w_f_diag = self.add_variable("w_f_diag", shape=[self._num_units],
                                             initializer=self._initializer)
            self._w_i_diag = self.add_variable("w_i_diag", shape=[self._num_units],
                                             initializer=self._initializer)
            self._w_o_diag = self.add_variable("w_o_diag", shape=[self._num_units],
                                             initializer=self._initializer)

        if self._num_proj is not None:
            maybe_proj_partitioner = (
              fixed_size_partitioner(self._num_proj_shards)
              if self._num_proj_shards is not None
              else None)
            self._proj_kernel = self.add_variable(
              "projection/%s" % _WEIGHTS_VARIABLE_NAME,
              shape=[self._num_units, self._num_proj],
              initializer=self._initializer,
              partitioner=maybe_proj_partitioner)

        self.built = True
