import tensorflow as tf
import numpy as np
# import logging
# logging.basicConfig(filename='Planning_Agent.log',level=logging.DEBUG,filemode='w')
from incentive_design.alg.baumann_rl.Agents import Agent


class Planning_Agent(Agent):

    def __init__(self, sess, env, underlying_agents, config):

        super().__init__(sess, env, config.lr, config.gamma)     

        self.underlying_agents = underlying_agents
        self.log = []
        self.grad_clip = config.grad_clip
        self.r_multiplier = config.r_multiplier
        n_players = len(underlying_agents)
        self.with_redistribution = config.with_redistribution
        self.value_fn_variant = config.value_fn_variant

        self.s = tf.placeholder(tf.float32, [1, env.l_obs], "state")
        self.a_players = tf.placeholder(tf.float32, [1, n_players], "player_actions")
        self.r_id = tf.placeholder(tf.float32, None, "player_rewards") # planner's reward
        self.inputs = tf.concat([self.s, self.a_players], 1)

        with tf.variable_scope('Policy_p'):
            self.l1 = tf.layers.dense(inputs=self.inputs, units=config.nn_h1,
                                      activation=tf.nn.relu, use_bias=True,
                                      name='l1')
            self.l2 = tf.layers.dense(inputs=self.l1, units=config.nn_h2,
                                      activation=tf.nn.relu, use_bias=True,
                                      name='l2')
            self.l3 = tf.layers.dense(inputs=self.l2, units=n_players,
                                      activation=None, use_bias=False,
                                      name='l3')

            if self.r_multiplier is None:
                self.action_layer = self.l3
            else:
                self.action_layer = tf.sigmoid(self.l3)

        with tf.variable_scope('Vp'):
            # Vp is trivial to calculate in this special case
            if self.r_multiplier is not None:
                self.vp = self.r_multiplier * self.action_layer
            else:
                self.vp = self.action_layer

        with tf.variable_scope('V_total'):
            if self.value_fn_variant == 'estimated':
                self.v = self.r_id
        with tf.variable_scope('cost_function'):
            if self.value_fn_variant == 'estimated':
                self.g_log_pi = tf.placeholder(
                    tf.float32, [n_players, env.l_obs, env.l_action],
                    "player_gradients")
            cost_list = []
            for underlying_agent in underlying_agents:
                # policy gradient theorem
                idx = underlying_agent.agent_idx
                if self.value_fn_variant == 'estimated':
                    self.g_Vp = (tf.reshape(self.g_log_pi[idx], [-1]) *
                                 self.vp[0, idx])
                    self.g_V = (tf.reshape(self.g_log_pi[idx], [-1]) *
                                self.v)

                cost_list.append(-underlying_agent.learning_rate *
                                 tf.tensordot(self.g_Vp, self.g_V, 1))

            if self.with_redistribution:
                extra_loss = config.cost_param * tf.norm(self.vp - tf.reduce_mean(self.vp))
            else:
                extra_loss = config.cost_param * tf.norm(self.vp)
            self.loss = tf.reduce_sum(tf.stack(cost_list)) + extra_loss

        with tf.variable_scope('trainPlanningAgent'):
            var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Policy_p')
            grads = tf.gradients(self.loss, var_list)
            grads = ([tf.clip_by_norm(grad, self.grad_clip) for grad in
                      grads] if self.grad_clip else grads)
            grads_and_vars = list(zip(grads, var_list))
            opt = tf.train.AdamOptimizer(config.lr)
            self.train_op = opt.apply_gradients(grads_and_vars)

    def learn(self, s, a_players, r_id, done, epsilon=0):
        s_expand = s[np.newaxis, :]
        a_players = np.asarray(a_players)
        feed_dict = {self.s: s_expand,
                     self.a_players: a_players[np.newaxis,:],
                     self.r_id: r_id}
        if self.value_fn_variant == 'estimated':
            g_log_pi_list = []
            for underlying_agent in self.underlying_agents:
                idx = underlying_agent.agent_idx
                # [l_obs, l_action]
                g_log_pi_list.append(underlying_agent.calc_g_log_pi(
                    s_expand, a_players[idx], epsilon))
            # [n_players, l_obs, l_action]
            g_log_pi_arr = np.asarray(g_log_pi_list)
            feed_dict[self.g_log_pi] = g_log_pi_arr

        self.sess.run(self.train_op, feed_dict)

    def get_log(self):
        return self.log

    def choose_action(self, s, a_players):

        s = s[np.newaxis, :]
        a_players = np.asarray(a_players)        
        a_plan = self.sess.run(self.action_layer, {self.s: s, self.a_players: a_players[np.newaxis,:]})[0,:]
        if self.r_multiplier is not None:
            a_plan = self.r_multiplier * a_plan

        return a_plan
