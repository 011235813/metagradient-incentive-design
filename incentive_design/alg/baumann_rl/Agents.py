import numpy as np
import tensorflow as tf
# import logging
# logging.basicConfig(filename='Agents.log',level=logging.DEBUG)

from incentive_design.utils import util

from enum import Enum, auto
class Critic_Variant(Enum):
    INDEPENDENT = auto()
    CENTRALIZED = auto()
    CENTRALIZED_APPROX = auto()

class Agent(object):
    def __init__(self, sess, env, learning_rate=0.001, gamma=0.95,
                 agent_idx=0):
        self.sess = sess
        self.env = env
        self.n_actions = env.l_action
        self.n_features = env.l_obs
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.agent_idx = agent_idx
        # self.log = [] # logs action probabilities

    def choose_action(self, s, epsilon=0):
        action_probs = self.calc_action_probs(s, epsilon)
        # select action w.r.t the actions prob
        if np.any(np.isnan(action_probs)):
            print('NaN in Agents.py choose_action()')
            print('action_probs', action_probs)
            print('s', s, 'epsilon', epsilon)
        action = np.random.choice(range(action_probs.shape[1]),
                                  p=action_probs.ravel())
        # self.log.append(action_probs[0,1])
        return action

    def learn_at_episode_end(self):
        pass

    def close(self):
        self.sess.close()
        tf.reset_default_graph()

    def reset(self):
        self.sess.run(tf.global_variables_initializer())


class Actor_Critic_Agent(Agent):
    def __init__(self, sess, env, learning_rate=0.001, n_units_actor=20, 
                 n_units_critic=20, gamma=0.95, agent_idx=0, 
                 critic_variant=Critic_Variant.INDEPENDENT, *args):
        super().__init__(sess, env, learning_rate, gamma, agent_idx)
        self.actor = Actor(env, n_units_actor, learning_rate, agent_idx)
        self.critic = Critic(env, n_units_critic, learning_rate, gamma, agent_idx, 
                            critic_variant)

    def learn(self, s, a, r, s_, done, epsilon=0, *args):

        td = self.critic.learn(self.sess, s, r, s_, done, *args)
        self.actor.learn(self.sess, s, a, td, epsilon)

    def __str__(self):
        return 'Actor_Critic_Agent_'+str(self.agent_idx)

    def calc_action_probs(self, s, epsilon=0):
        return self.actor.calc_action_probs(self.sess, s, epsilon)

    def pass_agent_list(self, agent_list):
        self.critic.pass_agent_list(agent_list)

    def get_action_prob_variable(self):
        return self.actor.actions_prob

    def get_state_variable(self):
        return self.actor.s

    def get_policy_parameters(self):
        # return [self.actor.w_l1,self.actor.b_l1,self.actor.w_pi1,self.actor.b_pi1]
        return [self.actor.theta]

    def calc_g_log_pi(self, s, a, epsilon=0):
        return self.actor.calc_g_log_pi(self.sess, s, a, epsilon)


class Actor(object):
    def __init__(self, env, n_units = 20, learning_rate=0.001, agent_idx = 0):
        self.s = tf.placeholder(tf.float32, [1, env.l_obs], "state")
        self.a = tf.placeholder(tf.int32, None, "act")
        self.td_error = tf.placeholder(tf.float32, None, "td_error")  # TD_error
        self.epsilon = tf.placeholder(tf.float32, None, 'epsilon')

        with tf.variable_scope('Actor_'+ str(agent_idx)):

            # [l_obs, l_action]
            self.theta = tf.Variable(tf.random_normal([env.l_obs, env.l_action], stddev=0.5))
            # [1, l_action]
            self.actions_prob = tf.nn.softmax(tf.matmul(self.s, self.theta))
            self.actions_prob = ((1 - self.epsilon)*self.actions_prob +
                                 self.epsilon/env.l_action)

            # self.w_l1 = tf.Variable(tf.random_normal([env.l_obs,n_units],stddev=0.1))
            # self.b_l1 = tf.Variable(tf.random_normal([n_units],stddev=0.1))
            #
            # self.l1 = tf.nn.relu(tf.matmul(self.s, self.w_l1) + self.b_l1)
            #
            # self.w_pi1 = tf.Variable(tf.random_normal([n_units,env.l_action],stddev=0.1))
            # self.b_pi1 = tf.Variable(tf.random_normal([env.l_action],stddev=0.1))
            #
            # self.actions_prob = tf.nn.softmax(tf.matmul(self.l1, self.w_pi1) + self.b_pi1)

        with tf.variable_scope('exp_v'):
            # scalar
            self.log_prob = tf.log(self.actions_prob[0, self.a] + 1e-15)
            # [l_obs, l_action]
            self.g_log_pi = tf.squeeze(tf.gradients(self.log_prob, self.theta))
            self.exp_v = tf.reduce_mean(self.log_prob * self.td_error)

        with tf.variable_scope('trainActor'):
            self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(-self.exp_v)

    def learn(self, sess, s, a, td, epsilon=0):
        s = s[np.newaxis, :]
        feed_dict = {self.s: s, self.a: a, self.td_error: td,
                     self.epsilon: epsilon}
        _, exp_v = sess.run([self.train_op, self.exp_v], feed_dict)
        return exp_v

    def calc_action_probs(self, sess, s, epsilon=0):
        s = s[np.newaxis, :]
        # get probabilities for all actions
        probs = sess.run(self.actions_prob,
                         {self.s: s, self.epsilon: epsilon})

        return probs

    def calc_g_log_pi(self, sess, s, a, epsilon=0):
        return sess.run(self.g_log_pi, feed_dict={
            self.s: s, self.a: a, self.epsilon: epsilon})


class PG(Actor):
    
    def __init__(self, sess, l_obs, l_action, config, agent_idx=0):

        self.sess = sess
        self.l_obs = l_obs
        self.l_action = l_action
        self.gamma = config.gamma
        self.entropy_coeff = config.entropy_coeff
        self.learning_rate = config.lr
        self.agent_idx = agent_idx

        self.s = tf.placeholder(tf.float32, [None, self.l_obs], "state")
        self.a_1hot = tf.placeholder(tf.float32, [None, self.l_action], "a_1hot")
        self.a = tf.placeholder(tf.int32, None, "act")
        self.returns = tf.placeholder(tf.float32, None, "returns")
        self.epsilon = tf.placeholder(tf.float32, None, 'epsilon')

        with tf.variable_scope('Actor_'+ str(agent_idx)):
            # [l_obs, l_action]
            self.theta = tf.Variable(tf.random_normal([self.l_obs, self.l_action], stddev=0.5))
            # [batch, l_action]
            self.actions_prob = tf.nn.softmax(tf.matmul(self.s, self.theta))
            # Exploration lower bound
            self.actions_prob = ((1 - self.epsilon)*self.actions_prob +
                                 self.epsilon/self.l_action)

        with tf.variable_scope('exp_v'):
            # [batch], used for agent's own training
            self.log_prob = tf.log(tf.reduce_sum(
                tf.multiply(self.actions_prob, self.a_1hot), axis=1) + 1e-15)
            
            # Used by the planner with batch size 1
            log_prob_single = tf.log(self.actions_prob[0, self.a] + 1e-15)
            # [l_obs, l_action]
            self.g_log_pi = tf.squeeze(tf.gradients(log_prob_single, self.theta))

            entropy = -tf.reduce_sum(tf.multiply(
                self.actions_prob, tf.log(self.actions_prob + 1e-15)))
            policy_loss = -tf.reduce_sum(self.log_prob * self.returns)

            self.loss = policy_loss - self.entropy_coeff * entropy

        with tf.variable_scope('trainActor'):
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def choose_action(self, s, epsilon=0):
        action_probs = self.calc_action_probs(self.sess, s, epsilon)
        # select action w.r.t the actions prob
        action = np.random.choice(range(action_probs.shape[1]),
                                  p=action_probs.ravel())
        return action

    def learn(self, sess, buf, epsilon=0):

        a_1hot = util.convert_batch_action_int_to_1hot(buf.action, self.l_action)
        returns = util.compute_returns(buf.r_env, self.gamma)
        feed_dict = {self.s: buf.obs,
                     self.a_1hot: a_1hot,
                     self.returns: returns,
                     self.epsilon: epsilon}
        sess.run(self.train_op, feed_dict)

    def calc_g_log_pi(self, s, a, epsilon=0):

        return self.sess.run(self.g_log_pi, feed_dict={
            self.s: s, self.a: a, self.epsilon: epsilon})


class Critic(object):
    def __init__(self, env, n_units, learning_rate, gamma, agent_idx, 
                critic_variant = Critic_Variant.INDEPENDENT):
        self.critic_variant = critic_variant
        self.env = env

        self.s = tf.placeholder(tf.float32, [1, env.l_obs], "state")
        self.v_ = tf.placeholder(tf.float32, [1, 1], "v_next")
        self.r = tf.placeholder(tf.float32, None, 'r')
        self.done = tf.placeholder(tf.float32, None, 'done')

        if self.critic_variant is Critic_Variant.CENTRALIZED:
            self.act_probs = tf.placeholder(tf.float32, shape=[1, env.l_action * env.n_agents], name="act_probs")
            self.nn_inputs = tf.concat([self.s,self.act_probs],axis=1)
        else: 
            self.nn_inputs = self.s

        with tf.variable_scope('Critic'):
            l1 = tf.layers.dense(
                inputs=self.nn_inputs,
                units=n_units,  # number of hidden units
                activation=tf.nn.relu,  # None
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1'+str(agent_idx)
            )

            self.v = tf.layers.dense(
                inputs=l1,
                units=1,  # output units
                activation=None,
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='V'+str(agent_idx)
            )

        with tf.variable_scope('squared_TD_error'):
            self.td_error = self.r + (1-self.done) * gamma * self.v_ - self.v
            self.loss = tf.square(self.td_error)
        with tf.variable_scope('trainCritic'):
            self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

    def pass_agent_list(self, agent_list):
        self.agent_list = agent_list

    def learn(self, sess, s, r, s_, done, *args):
        s,s_ = s.astype(np.float32), s_.astype(np.float32)

        if self.critic_variant is Critic_Variant.CENTRALIZED:
            if args: 
                obslist = args[0]
                obs_list = args[1]
                act_probs = np.hstack([agent.calc_action_probs(obslist[idx]) for idx, agent in enumerate(self.agent_list)])
                act_probs_ = np.hstack([agent.calc_action_probs(obs_list[idx]) for idx, agent in enumerate(self.agent_list)])
            else: 
                act_probs = np.hstack([agent.calc_action_probs(s) for idx, agent in enumerate(self.agent_list)])
                act_probs_ = np.hstack([agent.calc_action_probs(s_) for idx, agent in enumerate(self.agent_list)])
            nn_inputs = np.hstack([s[np.newaxis, :], act_probs])
            nn_inputs_ = np.hstack([s_[np.newaxis, :], act_probs_])
        else:
            nn_inputs, nn_inputs_ = s[np.newaxis, :], s_[np.newaxis, :]

        v_ = sess.run(self.v, {self.nn_inputs: nn_inputs_})
        td_error, _ = sess.run([self.td_error, self.train_op],
                               {self.nn_inputs: nn_inputs,
                                self.v_: v_, self.r: r, self.done: done})

        return td_error
