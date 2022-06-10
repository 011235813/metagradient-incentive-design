import numpy as np
import tensorflow as tf


def conv(t_input, scope, n_filters=6, k=(3, 3),
         s=(1, 1), data_format='NHWC'):

    if data_format == 'NHWC':
        channel_axis = 3
        strides = [1, s[0], s[1], 1]
        b_shape = [1, 1, 1, n_filters]
    elif data_format == 'NCHW':
        channel_axis = 1
        strides = [1, 1, s[0], s[1]]
        b_shape = [1, n_filters, 1, 1]
    
    # Flatten the tensor
    num_input_dim = len(t_input.get_shape())
    batch_dim = tf.shape(t_input)[:num_input_dim - 3]
    remaining_dim = t_input.get_shape().as_list()[num_input_dim - 3:]
    
    if num_input_dim > 4:
        t_input = tf.reshape(t_input, [-1] + remaining_dim)

    n_in = t_input.get_shape()[channel_axis].value
    w_shape = [k[0], k[1], n_in, n_filters]
    with tf.variable_scope(scope):
        w = tf.get_variable('w', w_shape)
        b = tf.get_variable('b', b_shape)
        out = b + tf.nn.conv2d(t_input, w, strides=strides, padding='SAME',
    
                               data_format=data_format)
    
    if num_input_dim > 4:
        out = tf.reshape(out, [batch_dim[0], batch_dim[1]] + out.get_shape().as_list()[1:])
        
    return out


def convnet(t_input, f=[6], k=[[3,3]], s=[[1,1]]):
    """Convolutional layer(s) with flattening.

    Args:
        t_input: TF placeholder
        f: list of filter numbers
        k: list of kernel size
        s: list of stride
    """
    assert len(f) == len(k) == len(s)
    h = t_input
    for idx in range(len(f)):
        h = tf.nn.relu(conv(h, 'conv_%d'%(idx+1), f[idx], k[idx], s[idx]))
    if len(h.get_shape().as_list()) == 5:
        # Output is a tensor with shape (batch_size, timesteps, output_shape)
        size = np.prod(h.get_shape().as_list()[2:])
        h_shape = tf.shape(h)
        conv_flat = tf.reshape(h, [h_shape[0], h_shape[1], size])
    else:
        size = np.prod(h.get_shape().as_list()[1:])
        conv_flat = tf.reshape(h, [-1, size])

    return conv_flat


def actor_mlp(obs, n_actions, config, return_logits=False):
    h1 = tf.layers.dense(inputs=obs, units=config.n_h1,
                         activation=tf.nn.relu,
                         use_bias=True, name='actor_h1')
    h2 = tf.layers.dense(inputs=h1, units=config.n_h2,
                         activation=tf.nn.relu,
                         use_bias=True, name='actor_h2')
    out = tf.layers.dense(inputs=h2, units=n_actions, activation=None,
                          use_bias=True, name='actor_out')
    probs = tf.nn.softmax(out, name='actor_softmax')
    if return_logits:
        return out, probs
    else:
        return probs


def actor_single_layer(obs, l_obs, n_actions):
    """Used for agents in ER for comparison to AMD."""
    theta = tf.Variable(tf.random_normal([l_obs, n_actions], stddev=0.5),
                        name='theta')
    probs = tf.nn.softmax(tf.matmul(obs, theta))

    return probs


def actor_image_vec(obs_image, obs_vec, dim_action, config, lstm_cell=None,
                    lstm_state=None):
    """policy network with image and flat observation spaces.

    Args:
        obs_image: image part of observation
        obs_vec: flat part of observation
        dim_action: list of ints (could be size 1)
        config: ConfigDict object

    If dim_action is a list of length |A|, then the policy is factored
    whereby pi(a|s) = \prod_{i=1}^|A| pi(a_i|s)
    and there are |A| separate softmax outputs

    Returns: list of Tensors (if dim_action) is a list, or single Tensor
             lstm state if lstm_cell != None
    """
    conv_out = convnet(obs_image, config.n_filters, config.kernel,
                       config.stride)
    
    if obs_vec != None:
        obs_shape = obs_vec.shape[1]  # [batch, flattened]
        if len(conv_out.shape) == 3:  # [n_agents, t, flattened]
            obs_shape = obs_vec.shape[2]

        conv_flat_dense = tf.layers.dense(
            inputs=conv_out, units=obs_shape,
            activation=tf.nn.relu, use_bias=True, name='conv_flat_dense')

        if len(conv_flat_dense.shape) == 3:
            h = tf.concat([conv_flat_dense, obs_vec], axis=2)
        else:
            h = tf.concat([conv_flat_dense, obs_vec], axis=1)
    else:
        h = conv_out
    
    for idx, n in enumerate(config.n_fc):
        h = tf.layers.dense(inputs=h, units=n, activation=tf.nn.relu,
                            use_bias=True, name='fc_%d'%(idx+1))
    
    
    if lstm_cell:
        """
        sequence_length = tf.fill([tf.shape(h)[0]], config.max_timesteps)
        h = tf.unstack(h, num=config.max_timesteps, axis=1)
        h, state = tf.nn.static_rnn(cell=lstm_cell, inputs=h,
                                                dtype=tf.float32,
                                                initial_state=lstm_state,
                                                sequence_length=sequence_length)
        h = tf.stack(h, axis=1)
        """
        if config.use_lstm_actor:
            h, state = tf.nn.dynamic_rnn(cell=lstm_cell, inputs=h,
                                                    dtype=tf.float32,
                                                    initial_state=lstm_state)
        else:
            zeros = tf.zeros_like(lstm_state.c)
            state = tf.nn.rnn_cell.LSTMStateTuple(zeros, zeros)
        
    
    if isinstance(dim_action, list) or isinstance(dim_action, np.ndarray):
        probs = []
        for idx, dim in enumerate(dim_action):
            out = tf.layers.dense(inputs=h, units=dim, activation=None,
                                  use_bias=True, name='out_%d'%idx)
            probs.append(tf.nn.softmax(out, name='probs_%d'%idx))
    else:
        out = tf.layers.dense(inputs=h, units=dim_action,
                              activation=None, use_bias=True, name='out')
        probs = tf.nn.softmax(out, name='probs')
    if lstm_cell:
        return probs, state
    else:
        return probs


def incentive_mlp(obs, a_others, config, n_outputs=1,
               output_nonlinearity=tf.nn.sigmoid):
    """Computes reward that this agent gives to all agents.

    Args:
        obs: TF placeholder
        a_others: TF placeholder for observation of other agents' actions
        config: configDict
        n_outputs: number of output nodes
        output_nonlinearity: None or a TF function

    Returns: TF tensor
    """
    concated = tf.concat([obs, a_others], axis=1)
    h1 = tf.layers.dense(inputs=concated, units=config.n_h1,
                         activation=tf.nn.relu,
                         use_bias=True, name='incentive_h1')
    h2 = tf.layers.dense(inputs=h1, units=config.n_h2,
                         activation=tf.nn.relu,
                         use_bias=True, name='incentive_h2')
    reward_out = tf.layers.dense(inputs=h2, units=n_outputs,
                                 activation=output_nonlinearity,
                                 use_bias=False, name='incentive')
    return reward_out


def actor_gaussian(obs, a_others, config, n_recipients=1):
    """Computes mean and stddev of Gaussian policy.

    Used by the incentive designer in dual-RL.

    Args:
        obs: TF placeholder
        a_others: TF placeholder for observation of other agents' actions
        config: configDict
        n_recipients: number of output nodes

    Returns: TF tensor
    """
    concated = tf.concat([obs, a_others], axis=1)
    h1 = tf.layers.dense(inputs=concated, units=config.n_h1,
                         activation=tf.nn.relu,
                         use_bias=True, name='h1')
    h2 = tf.layers.dense(inputs=h1, units=config.n_h2,
                         activation=tf.nn.relu,
                         use_bias=True, name='h2')
    mean = tf.layers.dense(inputs=h2, units=n_recipients,
                           activation=None, use_bias=True, name='mean')
    stddev = tf.layers.dense(inputs=h2, units=n_recipients,
                             activation=tf.nn.relu, use_bias=False, name='stddev')

    return mean, stddev


def vnet_image_vec(obs_image, obs_vec, config, lstm_cell=None, lstm_state=None):
    """Value function critic network with image and flat observations.

    Args:
        obs_image: image part of observation
        obs_vec: agents observe all agents' smoothed rewards
        config: ConfigDict object
    """
    conv_out = convnet(obs_image, config.n_filters, config.kernel,
                       config.stride)
    
    if obs_vec != None:
        obs_shape = obs_vec.shape[1]
        if len(conv_out.shape) == 3:
            obs_shape = obs_vec.shape[2]

        conv_flat_dense = tf.layers.dense(
            inputs=conv_out, units=obs_shape,
            activation=tf.nn.relu, use_bias=True, name='conv_flat_dense')

        if len(conv_flat_dense.shape) == 3:
            h = tf.concat([conv_flat_dense, obs_vec], axis=2)
        else:
            h = tf.concat([conv_flat_dense, obs_vec], axis=1)
    else:
        h = conv_out
        
    for idx, n in enumerate(config.n_fc):
        h = tf.layers.dense(inputs=h, units=n, activation=tf.nn.relu,
                            use_bias=True, name='fc_%d'%(idx+1))
    if lstm_cell:
        if config.use_lstm_critic:
            h, state = tf.nn.dynamic_rnn(
                cell=lstm_cell, inputs=h,
                dtype=tf.float32, initial_state=lstm_state)
        else:
            zeros = tf.zeros_like(lstm_state.c)
            state = tf.nn.rnn_cell.LSTMStateTuple(zeros, zeros)

    out = tf.layers.dense(inputs=h, units=1, activation=None,
                          use_bias=True, name='out')
        
    return out, state


def tax_image_vec(obs_image, obs_vec, n_outputs, config, noise,
                  lstm_cell=None, lstm_state=None):
    """tax network with image and flat observation spaces.

    Args:
        obs_image: image part of observation
        obs_vec: flat part of observation
        n_outputs: int number of output nodes
        config: ConfigDict object

    Returns: TF tensor
    """
    # NOTE: LSTM case is not yet tested, just carried over from
    # actor_image_vec

    conv_out = convnet(obs_image, config.n_filters, config.kernel,
                       config.stride)
    
    obs_shape = obs_vec.shape[1]
    if len(conv_out.shape) == 3: # Not tested
        obs_shape = obs_vec.shape[2]
    
    conv_flat_dense = tf.layers.dense(
        inputs=conv_out, units=obs_shape,
        activation=tf.nn.relu, use_bias=True, name='conv_flat_dense')
    
    if len(conv_flat_dense.shape) == 3: # Not tested
        h = tf.concat([conv_flat_dense, obs_vec], axis=2)
    else:
        h = tf.concat([conv_flat_dense, obs_vec], axis=1)
    
    for idx, n in enumerate(config.n_fc):
        h = tf.layers.dense(inputs=h, units=n, activation=tf.nn.relu,
                            use_bias=True, name='fc_%d'%(idx+1))
    
    if lstm_cell: # Not tested
        if config.use_lstm_actor:
            h, state = tf.nn.dynamic_rnn(cell=lstm_cell, inputs=h,
                                                    dtype=tf.float32,
                                                    initial_state=lstm_state)
        else:
            zeros = tf.zeros_like(lstm_state.c)
            state = tf.nn.rnn_cell.LSTMStateTuple(zeros, zeros)
        
    out = tf.layers.dense(inputs=h, units=n_outputs,
                          activation=None,
                          use_bias=False, name='out')
                     
    out = tf.nn.sigmoid(out + noise)

    if lstm_cell:
        return out[0], state
    else:
        return out


# ------------ Code below is used for SSD env only ---------------- #

def conv_ssd(t_input, scope, n_filters=6, k=(3,3), s=(1,1),
             data_format='NHWC'):

    if data_format == 'NHWC':
        channel_axis = 3
        strides = [1, s[0], s[1], 1]
        b_shape = [1, 1, 1, n_filters]
    elif data_format == 'NCHW':
        channel_axis = 1
        strides = [1, 1, s[0], s[1]]
        b_shape = [1, n_filters, 1, 1]
    n_in = t_input.get_shape()[channel_axis].value
    w_shape = [k[0], k[1], n_in, n_filters]
    with tf.variable_scope(scope):
        w = tf.get_variable('w', w_shape)
        b = tf.get_variable('b', b_shape)
        out = b + tf.nn.conv2d(t_input, w, strides=strides, padding='SAME',
                               data_format=data_format)

    return out


def convnet_ssd(t_input, f=6, k=(3, 3), s=(1, 1)):
    h = tf.nn.relu(conv_ssd(t_input, 'c1', f, k, s))
    size = np.prod(h.get_shape().as_list()[1:])
    conv_flat = tf.reshape(h, [-1, size])

    return conv_flat


def actor_ssd(obs, n_actions, config, return_logits=False):
    conv_out = convnet_ssd(obs, config.n_filters, config.kernel,
                           config.stride)
    h1 = tf.layers.dense(inputs=conv_out, units=config.n_h1,
                         activation=tf.nn.relu, use_bias=True, name='actor_h1')
    h2 = tf.layers.dense(inputs=h1, units=config.n_h2,
                         activation=tf.nn.relu, use_bias=True, name='actor_h2')
    out = tf.layers.dense(inputs=h2, units=n_actions, activation=None,
                          use_bias=True, name='actor_out')
    probs = tf.nn.softmax(out, name='actor_probs')
    if return_logits:
        return out, probs
    else:
        return probs


def vnet_ssd(obs, config):
    conv_out = convnet_ssd(obs, config.n_filters, config.kernel,
                           config.stride)
    h1 = tf.layers.dense(inputs=conv_out, units=config.n_h1,
                         activation=tf.nn.relu, use_bias=True, name='v_h1')
    h2 = tf.layers.dense(inputs=h1, units=config.n_h2,
                         activation=tf.nn.relu, use_bias=True, name='v_h2')
    out = tf.layers.dense(inputs=h2, units=1, activation=None,
                          use_bias=True, name='v_out')

    return out


def incentive_ssd(obs, a_others, config, n_outputs=1,
                  output_nonlinearity=tf.nn.sigmoid):
    """Computes reward that this agent gives to (all) agents.

    Uses a convolutional net to process image obs.

    Args:
        obs: TF placeholder
        a_others: TF placeholder for observation of other agents' actions
        config: configDict
        n_outputs: number of output nodes
        output_nonlinearity: None or a TF function

    Returns: TF tensor
    """
    conv_out = convnet_ssd(obs, config.n_filters, config.kernel,
                           config.stride)
    conv_reduced = tf.layers.dense(inputs=conv_out, units=config.n_h1,
                                   activation=tf.nn.relu, use_bias=True,
                                   name='reward_conv_reduced')
    concated = tf.concat([conv_reduced, a_others], axis=1)
    h2 = tf.layers.dense(inputs=concated, units=config.n_h2,
                         activation=tf.nn.relu, use_bias=True, name='reward_h2')
    reward_out = tf.layers.dense(inputs=h2, units=n_outputs,
                                 activation=output_nonlinearity,
                                 use_bias=False, name='reward')
    return reward_out
