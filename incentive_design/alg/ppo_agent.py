import numpy as np
import tensorflow as tf

import networks
from incentive_design.utils import util

class Agent(object):
    
    def __init__(self, agent_name, config, dim_action,
                 dim_obs_flat, dim_obs_tensor, n_agents, objective,  nn):
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
        self.n_agents = n_agents
        self.objective = objective

        self.entropy_coeff = config.entropy_coeff
        self.gamma = config.gamma
        self.grad_clip = config.grad_clip
        self.lr_actor = config.lr_actor
        self.lr_v = config.lr_v
        self.tau = config.tau
        self.gae_lambda = config.gae_lambda
        self.ppo_epsilon = config.ppo_epsilon
        self.entropy_coeff = config.entropy_coeff
        
        self.create_networks()
        self.create_critic_train_op()
        self.create_policy_gradient_op()
   
    def create_networks(self):
        """Placeholders and neural nets."""
        if self.tensor_and_flat:
            # [n, t, ...obs dim...]
            self.obs_tensor = tf.placeholder(
                tf.float32, [self.n_agents, None]+list(self.dim_obs_tensor), 'obs_tensor')
            self.obs_flat = tf.placeholder(
                tf.float32, [self.n_agents, None, self.dim_obs_flat[0]], 'obs_flat')
            actor_net = networks.actor_image_vec
            value_net = networks.vnet_image_vec
            
            actor_lstm = tf.nn.rnn_cell.LSTMCell(self.nn.n_lstm)
            # 2 for hidden state and output
            self.new_actor_state = tf.placeholder(
                tf.float32, [2, self.n_agents, self.nn.n_lstm], 'new_actor_state')
          
            v_main_lstm = tf.nn.rnn_cell.LSTMCell(self.nn.n_lstm)
            self.new_v_main_state = tf.placeholder(
                tf.float32, [2, self.n_agents, self.nn.n_lstm], 'new_v_main_state')  
            v_target_lstm = tf.nn.rnn_cell.LSTMCell(self.nn.n_lstm)            
            self.new_v_target_state = tf.placeholder(
                tf.float32, [2, self.n_agents, self.nn.n_lstm], 'new_v_target_state')
            
        self.epsilon = tf.placeholder(tf.float32, [], 'epsilon')
        self.action_masks = [tf.placeholder(
            tf.float32, [self.n_agents, None, dim], 'action_mask_%d'%idx)
                             for idx, dim in enumerate(self.dim_action)]
        
        with tf.variable_scope(self.agent_name):
     
            with tf.variable_scope('policy'):
                if self.tensor_and_flat:
                    actor_tuple_state = tf.nn.rnn_cell.LSTMStateTuple(
                        self.new_actor_state[0], self.new_actor_state[1])
                    # probs is list of [n, t, |A|],
                    # length of list = number of action subspaces
                    probs, self.actor_state = actor_net(
                        self.obs_tensor, self.obs_flat, self.dim_action,
                        self.nn, actor_lstm, actor_tuple_state)
             
            self.probs = []
            self.log_probs = []
            self.action_samples = []
            for idx in range(len(self.dim_action)):
                # Apply action mask and normalize
                prob = tf.multiply(probs[idx], self.action_masks[idx])
                prob = prob + 1e-15 * tf.one_hot(0, self.dim_action[idx])
                # element-wise div, [n, t, |A|]
                prob = prob / tf.reshape(tf.reduce_sum(prob, axis=2),
                                         [self.n_agents, -1, 1])
                # Exploration lower bound
                self.probs.append((1 - self.epsilon) * prob +
                                  self.epsilon / self.dim_action[idx])
                self.log_probs.append(tf.log(self.probs[idx] + 1e-15))
                log_probs = self.log_probs[idx]
                # [n*t, |A|].
                # n*t means order is: agent_1 for all t,...,agent_n for all t
                log_probs = tf.reshape(log_probs, [-1, self.dim_action[idx]])
                action_samples = tf.multinomial(log_probs, 1)
                self.action_samples.append(
                    tf.reshape(action_samples, [self.n_agents, -1, 1])
                )
                    
            with tf.variable_scope('v_main'):
                if self.tensor_and_flat:
                    v_main_tuple_state = tf.nn.rnn_cell.LSTMStateTuple(
                        self.new_v_main_state[0], self.new_v_main_state[1]
                    )
                    self.v, v_main_state = value_net(
                        self.obs_tensor, self.obs_flat, self.nn, v_main_lstm, v_main_tuple_state) 
            
            with tf.variable_scope('v_target'):
                if self.tensor_and_flat:
                    v_target_tuple_state = tf.nn.rnn_cell.LSTMStateTuple(
                        self.new_v_target_state[0], self.new_v_target_state[1]
                    )
                    self.v_target, v_target_state = value_net(
                        self.obs_tensor, self.obs_flat, self.nn, v_target_lstm, v_target_tuple_state)           

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
    
    def run_actor(self, obs_tensor, obs_flat, action_masks, sess, epsilon, actor_state=None):
        """Gets action from policy.

        Args:
            obs_tensor: if agents with parameter sharing,
                        this is a list of np.array images.
                        if designer, this is a single np.array image.
            obs_flat: flat part of obs, similar cases as obs_tensor
            action_masks: list of binary np.array
            sess: TF session
            epsilon: float

        Returns: np.array [n_agents, 1] if len(dim_action)==1,
                 otherwise [#subspaces, 1, 1]
                 and log_probs, new_actor_state
        """
        
        if self.tensor_and_flat:
            # For parameter sharing agents, new shape is [n_agents,1,L,W,C]
            # For designer, new shape is [L,1,W,C], which is corrected below
            obs_tensor = np.swapaxes(np.array([obs_tensor]), 0, 1)
            obs_flat = np.swapaxes(np.array([obs_flat]), 0, 1)
            # If obs_tensors don't have dimension for n_agents:
            if len(obs_tensor.shape) == 4:
                # New shape is [1,1,L,W,C]
                obs_tensor = np.swapaxes(obs_tensor, 0, 1)
                obs_tensor = np.expand_dims(obs_tensor, axis=1)
                obs_flat = np.swapaxes(obs_flat, 0, 1)
                obs_flat = np.expand_dims(obs_flat, axis=1)
            
            feed = {self.obs_tensor: obs_tensor,
                    self.obs_flat:  obs_flat,
                    self.epsilon: epsilon}
            
        if actor_state:
            feed[self.new_actor_state] = actor_state
        else:
            feed[self.new_actor_state] = np.zeros([2, self.n_agents, self.nn.n_lstm])
        
        # action_masks have shape: (n_agents, dim_action_value)
        # or shape: (dim_action, dim_action_value)
        for idx in range(len(self.dim_action)):
            if len(self.dim_action) == 1:  # agents have non-factorized space
                action_masks = np.array(action_masks)
                # [n,1,dim_action]
                action_masks = np.expand_dims(action_masks, axis=1)
                # Necessary for static_rnn
                # action_masks = np.repeat(action_masks, self.nn.max_timesteps, axis=1)
                feed[self.action_masks[idx]] = action_masks
            else:  # designer has factorized space
                action_masks_idx = np.array(action_masks[idx])
                action_masks_idx = np.expand_dims(action_masks_idx, axis=0)
                # [1,1,dimension of each subspace]
                action_masks_idx = np.expand_dims(action_masks_idx, axis=1)
                feed[self.action_masks[idx]] = action_masks_idx

        actions, log_probs, new_actor_state = sess.run([self.action_samples, self.log_probs, self.actor_state], feed_dict=feed)
        
        # actions is a list with length = # action subspaces,
        # each element is np.array with shape [n_agents, timesteps, 1)
        if len(actions) == 1:  # parameter sharing agents
            # result: (n_agents, timesteps, 1)
            action = actions[0]

            # If only 1 timestep (which should always be the case)
            if action.shape[1] == 1:
                # result: [n_agents, 1]
                action = np.squeeze(action, axis=1)
        else:  # designer with multiple action subspaces
            # result: [#subspaces, n_agents, t, 1]
            actions = np.array(actions)
            
            # result: [n_agents, #subspaces, t, 1]
            action = np.swapaxes(actions, 0, 1)
            
            # If only 1 timestep
            if action.shape[1] == 1:  # this clause is never satisfied
                action = np.squeeze(action, axis=1)
        
        # If only 1 agent
        if action.shape[0] == 1:  # satisfied for designer, not agents
            # result: [#subspaces, t, 1]
            action = np.squeeze(action, axis=0)

        return action, log_probs, new_actor_state
    
    def create_critic_train_op(self):
        
        self.v_target_next = tf.placeholder(tf.float32, [self.n_agents, None, 1], 'v_target_next')
        self.reward = tf.placeholder(tf.float32, [self.n_agents, None, 1], 'reward')
        
        td_target = self.reward + self.gamma * self.v_target_next
        
        self.loss_v = tf.reduce_mean(tf.math.square(td_target - self.v))
        self.v_opt = tf.train.AdamOptimizer(self.lr_v)
        self.v_op = self.v_opt.minimize(self.loss_v)
    
    def create_policy_gradient_op(self):

        self.advantages = tf.placeholder(
            tf.float32, [self.n_agents, None, 1], 'advantages')
        self.old_log_probs = [tf.placeholder(
            tf.float32, [self.n_agents, None, dim], 'log_probs')
                              for dim in self.dim_action]
        self.action_taken = [tf.placeholder(
            tf.float32, [self.n_agents, None, dim])
                             for dim in self.dim_action]
        
        # new_probs is a list of [n_agents, timesteps, dim_action_value]
        # length of list = #action subspaces
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
                # [n,t,1]
                entropy = - tf.reduce_sum(mult_probs, axis=2, keepdims=True)

                    
            surr_1 = tf.multiply(ratio, self.advantages)
            surr_2 = tf.multiply(tf.clip_by_value(
                ratio, 1 - self.ppo_epsilon, 1 + self.ppo_epsilon),
                                 self.advantages)
            self.loss_actor = tf.math.reduce_mean(
                - tf.math.minimum(surr_1, surr_2)
                - self.entropy_coeff * entropy)
            self.actor_opt = tf.train.AdamOptimizer(self.lr_actor)
            if self.grad_clip:
                grad_var = self.actor_opt.compute_gradients(
                    self.loss_actor, var_list=self.policy_params)
                grad_var = [(tf.clip_by_norm(tup[0], self.grad_clip) if (not tup[0] is None) else tup[0], tup[1]) for tup in grad_var]
                self.actor_op = self.actor_opt.apply_gradients(grad_var)
            else:
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
            # Designer: [t,L,W,C]. Agents: [t,n,L,W,C]
            obs_tensor_next = np.array(buf.obs_tensor_next)
            obs_flat_next = np.array(buf.obs_flat_next)
            
            # If obs_tensors don't have dimension for n_agents:
            if len(obs_tensor_next.shape) == 4:
                obs_tensor_next = np.expand_dims(obs_tensor_next, axis=1)
                obs_flat_next = np.expand_dims(obs_flat_next, axis=1)
            
            # [n,t,L,W,C]
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
                                         feed_dict=feed) # [n,t,1]
        
        # Update value network
        feed[self.v_target_next] = v_target_next
        
        # Designer: [t]. Agents: [t,n]
        reward = np.array(buf.reward)

        # If reward doesn't have dimension for n_agents:
        if len(reward.shape) == 1:
            reward = np.expand_dims(reward, axis=1)
            
        reward = np.swapaxes(reward, 0, 1)
        # [n,t,1]
        reward = np.expand_dims(reward, axis=2)
        feed[self.reward] = reward
        
        if self.tensor_and_flat:
            # Designer: [t,L,W,C]. Agents: [t,n,L,W,C]
            obs_tensor = np.array(buf.obs_tensor)
            obs_flat = np.array(buf.obs_flat)
            
            # If obs_tensors don't have dimension for n_agents:
            if len(obs_tensor.shape) == 4:
                obs_tensor = np.expand_dims(obs_tensor, axis=1)
                obs_flat = np.expand_dims(obs_flat, axis=1)
            
            # [n,t,L,W,C]
            obs_tensor = np.swapaxes(obs_tensor, 0, 1)
            obs_flat = np.swapaxes(obs_flat, 0, 1)
            
            feed[self.obs_tensor] = obs_tensor
            feed[self.obs_flat] = obs_flat
        else:
            raise NotImplementedError
            
        v = sess.run(self.v, feed_dict=feed) # [n,t,1]
       
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
            if key == self.obs_tensor or key == self.obs_flat or key == self.advantages:
                feed_idx[key] = value[:, idx : idx + self.nn.max_timesteps, :]
            else:
                feed_idx[key] = value
        return feed_idx
    
    def feed_advantages(self, buf, v, v_next, feed):
        # v, v_next: [n,t,1]
        v_next_last = np.expand_dims(v_next[:, -1, :], axis=1) # [n,1,1]
        value_rollouts = np.concatenate((v, v_next_last), axis=1) # [n,t+1,1]

        advantages = buf.compute_advantages(
            value_rollouts, self.gamma, self.gae_lambda) # [n,t,1]
        feed[self.advantages] = advantages
    
    def feed_actions(self, feed, buf_log_probs, buf_action, buf_action_mask):
        # buf_log_probs have shape: (timesteps, # action subspaces,
        #                            n_agents, 1, dim_action_value)
        # time and #subspaces are lists
        for idx, old_log_probs in enumerate(self.old_log_probs):
            log_probs = []
            for step in range(len(buf_log_probs)):
                # list over t of [n, 1, action space]
                log_probs.append(buf_log_probs[step][idx])
            log_probs = np.squeeze(np.array(log_probs), axis=2) # [t, n, |A|]
            log_probs = np.swapaxes(log_probs, 0, 1)  # [n, t, |A|]
            # log_probs have shape: (n_agents, timesteps, dim_action_value)
            feed[old_log_probs] = log_probs
        
        # buf_action has shape:
        # (timesteps, #subspaces, n_agents=1, 1) for designer or
        # (timesteps, n_agents, 1) for parameter sharing agents
        actions = np.array(buf_action)

        if len(actions.shape) == 3:
            actions = np.expand_dims(actions, axis=1)  # [t, 1, n, 1]
        actions = np.swapaxes(actions, 0, 1) # [#subspaces, t, n, 1]
        actions = np.swapaxes(actions, 1, 2) # [#subspaces, n, t, 1]
        # actions has shape: (dim_action, n_agents, timesteps, 1)
        for idx, action_taken in enumerate(self.action_taken):
            action = []
            for i in range(self.n_agents):
                action_1hot = util.convert_batch_actions_int_to_1hot(actions[idx][i], self.dim_action[idx])
                action.append(action_1hot)
            feed[action_taken] = np.array(action)
        
        # buf_action_mask has shape
        # [t, #subspaces, action space size] for designer, or
        # [t, n_agents, action space size] for parameter sharing agents
        # where t dimension is a list
        action_mask = buf_action_mask
        
        for idx in range(len(self.dim_action)):
            if len(self.dim_action) == 1:  # non-factored action space
                dim_action_mask = np.array(action_mask) # [t, n, |A|]
                # [n, t, |A|]
                dim_action_mask = np.swapaxes(dim_action_mask, 0, 1)
                feed[self.action_masks[idx]] = dim_action_mask
            else:
                dim_action_mask = []
                for step in range(len(action_mask)):
                    dim_action_mask.append(action_mask[step][idx])
                dim_action_mask = np.array(dim_action_mask)  # [t, |A|]
                
                # action_mask has no dimension for n_agents:
                if len(dim_action_mask.shape) == 2:
                    # [t, 1, |A|]
                    dim_action_mask = np.expand_dims(dim_action_mask, axis=1)
                
                # [n=1, t, |A|]
                dim_action_mask = np.swapaxes(dim_action_mask, 0, 1)
                feed[self.action_masks[idx]] = dim_action_mask
