import numpy as np
import tensorflow as tf

import networks
from incentive_design.utils import util
from incentive_design.alg import ppo_agent_m2


class Agent(ppo_agent_m2.Agent):
    
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
        # agents don't have factored action space
        assert len(dim_action) == 1
        super().__init__(agent_id, agent_name, config, dim_action,
                 dim_obs_flat, dim_obs_tensor, n_agents, objective, nn,
                 tax, utility)
    
    def create_networks(self):
        """Placeholders and neural nets."""
        if self.tensor_and_flat:
            self.obs_tensor = tf.placeholder(
                tf.float32, [self.n_agents, None]+list(self.dim_obs_tensor), 'obs_tensor')
            self.obs_flat = tf.placeholder(
                tf.float32, [self.n_agents, None, self.dim_obs_flat[0]], 'obs_flat')
            actor_net = networks.actor_image_vec
            value_net = networks.vnet_image_vec
            
            actor_lstm = tf.nn.rnn_cell.LSTMCell(self.nn.n_lstm)
            self.new_actor_state = tf.placeholder(
                tf.float32, [2, self.n_agents, self.nn.n_lstm], 'new_actor_state')
          
            actor_prime_lstm = tf.nn.rnn_cell.LSTMCell(self.nn.n_lstm)
            self.new_actor_prime_state = tf.placeholder(
                tf.float32, [2, self.n_agents, self.nn.n_lstm], 'new_actor_prime_state')

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
                    probs, self.actor_state = actor_net(
                        self.obs_tensor, self.obs_flat, self.dim_action,
                        self.nn, actor_lstm, actor_tuple_state)
             
            self.probs = []
            self.log_probs = []
            self.action_samples = []
            for idx in range(len(self.dim_action)):
                # Apply action mask and normalize
                prob = tf.multiply(probs[idx], self.action_masks[idx])
                prob = prob / tf.reshape(tf.reduce_sum(prob, axis=2), [self.n_agents, -1, 1])
                # Exploration lower bound
                self.probs.append((1 - self.epsilon) * prob +
                                  self.epsilon / self.dim_action[idx])
                self.log_probs.append(tf.log(self.probs[idx] + 1e-15))
                log_probs = self.log_probs[idx]
                log_probs = tf.reshape(log_probs, [-1, self.dim_action[idx]])
                action_samples = tf.multinomial(log_probs, 1)
                self.action_samples.append(
                    tf.reshape(action_samples, [self.n_agents, -1, 1])
                )    
            
            with tf.variable_scope('policy_prime'):
                if self.tensor_and_flat:
                    actor_prime_tuple_state = tf.nn.rnn_cell.LSTMStateTuple(
                        self.new_actor_prime_state[0],
                        self.new_actor_prime_state[1])
                    probs_prime, self.actor_prime_state = actor_net(
                        self.obs_tensor, self.obs_flat, self.dim_action,
                        self.nn, actor_prime_lstm, actor_prime_tuple_state)
             
            self.probs_prime = []
            self.log_probs_prime = []
            self.action_samples_prime = []
            for idx in range(len(self.dim_action)):
                # Apply action mask and normalize
                prob = tf.multiply(probs_prime[idx], self.action_masks[idx])
                prob = prob + 1e-15 * tf.one_hot(0, self.dim_action[idx])
                prob = prob / tf.reshape(
                    tf.reduce_sum(prob, axis=2), [self.n_agents, -1, 1])
                # Exploration lower bound
                self.probs_prime.append((1 - self.epsilon) * prob +
                                  self.epsilon / self.dim_action[idx])
                self.log_probs_prime.append(
                    tf.log(self.probs_prime[idx] + 1e-15))
                log_probs = self.log_probs_prime[idx]
                log_probs = tf.reshape(log_probs, [-1, self.dim_action[idx]])
                action_samples = tf.multinomial(log_probs, 1)
                self.action_samples_prime.append(
                    tf.reshape(action_samples, [self.n_agents, -1, 1])
                )   
                    
            with tf.variable_scope('v_main'):
                if self.tensor_and_flat:
                    v_main_tuple_state = tf.nn.rnn_cell.LSTMStateTuple(
                        self.new_v_main_state[0], self.new_v_main_state[1]
                    )
                    self.v, critic_state = value_net(
                        self.obs_tensor, self.obs_flat, self.nn,
                        v_main_lstm, v_main_tuple_state) 
            
            with tf.variable_scope('v_target'):
                if self.tensor_and_flat:
                    v_target_tuple_state = tf.nn.rnn_cell.LSTMStateTuple(
                        self.new_v_target_state[0],
                        self.new_v_target_state[1]
                    )
                    self.v_target, critic_state_target = value_net(
                        self.obs_tensor, self.obs_flat, self.nn,
                        v_target_lstm, v_target_tuple_state)

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
                  actor_state=None, prime=False):
        """Gets action from policy.

        Args:
            obs_tensor: list of np.array image part of obs
            obs_flat: list of np.array flat part of obs
            action_masks: list of binary np.array
            sess: TF session
            epsilon: float

        Returns: np.array [n_agents, 1], log_probs, new_actor_state
        """
        
        if self.tensor_and_flat:
            obs_tensor = np.swapaxes(np.array([obs_tensor]), 0, 1)
            obs_flat = np.swapaxes(np.array([obs_flat]), 0, 1)
            feed = {self.obs_tensor: obs_tensor,
                    self.obs_flat:  obs_flat,
                    self.epsilon: epsilon}
            
        new_actor_state = self.new_actor_state
        if prime:
            new_actor_state = self.new_actor_prime_state
        if actor_state:
            feed[new_actor_state] = actor_state
        else:
            feed[new_actor_state] = np.zeros([2, self.n_agents, self.nn.n_lstm])
        
        # action_masks have shape: (n_agents, dim_action_value)
        # or shape: (dim_action, dim_action_value)
        for idx in range(len(self.dim_action)):
            action_masks = np.array(action_masks)  # [n, |A|]
            action_masks = np.expand_dims(action_masks, axis=1) # [n, 1, |A|]
            feed[self.action_masks[idx]] = action_masks

        if prime:
            actions, log_probs, new_actor_state = sess.run([self.action_samples_prime, self.log_probs_prime, self.actor_prime_state], feed_dict=feed)
        else:
            actions, log_probs, new_actor_state = sess.run([self.action_samples, self.log_probs, self.actor_state], feed_dict=feed)
        
        # actions is a list with length = # action subspaces,
        # each element is np.array with shape [n_agents, timesteps, 1)
        # result: (n_agents, timesteps, 1)
        action = actions[0]

        # If only 1 timestep (which should always be the case)
        if action.shape[1] == 1:
            # result: [n_agents, 1]
            action = np.squeeze(action, axis=1)
        
        # If only 1 agent
        if action.shape[0] == 1:
            action = np.squeeze(action, axis=0)
        
        return action, log_probs, new_actor_state

    def create_policy_gradient_op(self):
        """Used by the ID for its own parameter update.

        Agents' policy parameter update is computed by create_update_op()
        and update_step(), not here.
        """
        super().create_policy_gradient_op() # defines self.loss
        # This copy of AdamOptimizer is needed because m1 updates
        # the prime variables before the designer's update,
        # which changes the AdamOptimizer internal state.
        # id_foundation's update step needs the AdamOptimizer
        # internal state before the policy update.
        self.actor_opt = tf.train.AdamOptimizer(self.lr_actor)
        self.actor_op = self.actor_opt.minimize(self.loss)

    def create_update_op(self):
        self.advantages = tf.placeholder(tf.float32, [self.n_agents, None, 1], 'advantages')
        
        # new_probs is a list of [n_agents, timesteps, |A|]
        # length of list = # action subspaces = 1
        new_probs = self.probs_prime
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
            cart_prod_probs = self.probs_prime[0]
            
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
            grads = tf.gradients(self.loss_actor, self.policy_prime_params)
            grads_and_vars = list(zip(grads, self.policy_prime_params))
            self.actor_prime_opt = tf.train.AdamOptimizer(self.lr_actor)
            self.actor_prime_op = self.actor_prime_opt.apply_gradients(grads_and_vars)
        self.create_optimizer_internal_update_op()

    def create_optimizer_internal_update_op(self):
        """Creates ops that assign self.actor_prime_opt internals 
        to self.actor_opt internals.
        """
        self.list_update_optimizer_ops = []
        for var in self.policy_params:
            m = self.actor_opt.get_slot(var, 'm')
            v = self.actor_opt.get_slot(var, 'v')
            beta1_power, beta2_power = self.actor_opt._get_beta_accumulators()
            var_prime = tf.trainable_variables(
                var.name.replace('policy', 'policy_prime'))[0]
            m_prime = self.actor_prime_opt.get_slot(var_prime, 'm')
            v_prime = self.actor_prime_opt.get_slot(var_prime, 'v')
            beta1_power_prime, beta2_power_prime = self.actor_prime_opt._get_beta_accumulators()
            
            self.list_update_optimizer_ops.append(m.assign(m_prime))
            self.list_update_optimizer_ops.append(v.assign(v_prime))
            self.list_update_optimizer_ops.append(
                beta1_power.assign(beta1_power_prime))
            self.list_update_optimizer_ops.append(
                beta2_power.assign(beta2_power_prime))

    def update(self, sess, buf, epsilon):
        """Training step for policy and value function

        Args:
            sess: TF session
            buf: Buffer object
            epsilon: float
        """
        self.update_optimizer_internal(sess)

        # Input shape is (n_agents, timesteps, obs_shape)
        # Calculate values
        if self.tensor_and_flat:
            obs_tensor_next = np.array(buf.obs_tensor_next) # [t,n,L,W,C]
            obs_flat_next = np.array(buf.obs_flat_next)
            
            # If obs_tensors don't have dimension for n_agents:
            if len(obs_tensor_next.shape) == 4:
                obs_tensor_next = np.expand_dims(obs_tensor_next, axis=1)
                obs_flat_next = np.expand_dims(obs_flat_next, axis=1)
            
            obs_tensor_next = np.swapaxes(obs_tensor_next, 0, 1) #[n,t,L,W,C]
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
        
        reward = np.array(buf.reward) # [t,n]
        
        # If reward doesn't have dimension for n_agents:
        if len(reward.shape) == 1:
            reward = np.expand_dims(reward, axis=1)
            
        reward = np.swapaxes(reward, 0, 1) # [n,t]
        reward = np.expand_dims(reward, axis=2) # [n,t,1]
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
        _ = sess.run(self.actor_prime_op, feed_dict=feed)
        
    def update_main(self, sess):
        sess.run(self.list_copy_prime_to_main_ops)

    def update_optimizer_internal(self, sess, test=False):
        """Runs ops that assign actor_prime_opt internals to 
        actor_opt internals.
        Must be called before policy_prime update(), so that prime opt
        internals are copied before they are changed
        Designer needs the copied opt internals to replicate the opt step.
        """
        if test:
            input('Inside update_optimizer_internal, press key to continue: ')
            var = self.policy_params[1]
            var_prime = tf.trainable_variables(
                var.name.replace('policy', 'policy_prime'))[0]
            beta1_power_prev, beta2_power_prev = self.actor_opt._get_beta_accumulators()
            beta1_power_prime, beta2_power_prime = self.actor_prime_opt._get_beta_accumulators()
            m_prev, v_prev = [self.actor_opt.get_slot(var, s) for s in ['m', 'v']]
            m_prime, v_prime = [self.actor_prime_opt.get_slot(var_prime, s) for s in ['m', 'v']]
            print('Showing opt internals for variable', var)
            print('Before copy')
            print('main m, v', sess.run([m_prev, v_prev]))
            print('prime m, v', sess.run([m_prime, v_prime]))
            print('main beta', sess.run([beta1_power_prev, beta2_power_prev]))
            print('prime beta', sess.run([beta1_power_prime, beta2_power_prime]))

        sess.run(self.list_update_optimizer_ops)

        if test:
            m_after, v_after = [self.actor_opt.get_slot(var, s) for s in ['m', 'v']]
            beta1_power_after, beta2_power_after = self.actor_opt._get_beta_accumulators()
            print('After copy')
            print('main m, v', sess.run([m_after, v_after]))
            print('main beta', sess.run([beta1_power_after, beta2_power_after]))
