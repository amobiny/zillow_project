import tensorflow as tf


def weight_variable(name, shape):
    """
    Create a weight variable with appropriate initialization
    :param name: weight name
    :param shape: weight shape
    :return: initialized weight variable
    """
    initer = tf.contrib.layers.xavier_initializer(uniform=False)
    return tf.get_variable('W_' + name, dtype=tf.float32,
                           shape=shape, initializer=initer)


def bias_variable(name, shape):
    """
    Create a bias variable with appropriate initialization
    :param name: bias variable name
    :param shape: bias variable shape
    :return: initial bias variable
    """
    initial = tf.constant(0., shape=shape, dtype=tf.float32)
    return tf.get_variable('b_' + name, dtype=tf.float32,
                           initializer=initial)


def fc_layer(x, num_units, layer_name, add_reg, lmbda):
    with tf.name_scope(layer_name):
        regularizer = None
        if add_reg:
            regularizer = tf.contrib.layers.l2_regularizer(scale=lmbda)
        net = tf.layers.dense(inputs=x, units=num_units, kernel_regularizer=regularizer)
        print('{}: {}'.format(layer_name, net.get_shape()))
        return net

def fc_layer(x, out_dim, layer_name, add_reg, lmbda):
    """
    Creates a fully-connected layer
    :param x: input from previous layer
    :param out_dim: number of hidden units in the fully-connected layer
    :param layer_name: layer name
    :return: The output array
    """
    in_dim = x.get_shape()[1]
    summary_list = []
    with tf.variable_scope(layer_name):
        weights = weight_variable(layer_name, shape=[in_dim, out_dim])
        summary_list.append(tf.summary.histogram('W_' + layer_name, weights))
        biases = bias_variable(layer_name, [out_dim])
        x = tf.matmul(x, weights) + biases
        summary_list.append(tf.summary.histogram('b_' + layer_name, biases))
        if add_reg:
            tf.add_to_collection('reg_weights', weights)
    return x, summary_list

def dropout(x, rate, training):
    """Create a dropout layer."""
    return tf.layers.dropout(inputs=x, rate=rate, training=training)

def relu(x):
    return tf.nn.relu(x)