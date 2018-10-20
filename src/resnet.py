import tensorflow as tf
import numpy as np

# Helper functions

# Get the x * w * c or total dims in the tensor minus the batch size
def get_dim(x):
    shape = x.get_shape().as_list()
    dim = 1
    for d in shape[1:]:
        dim *= d
    return dim

# simple weight var
def weight_var(shape, init=xavier_init, name='weight'):
    return tf.Variable(init(shape), dtype=tf.float32)

# simple bias var
def bias_var(shape, name='bias'):
    return tf.Variable(tf.zeros(shape=shape), dtype=tf.float32)

# Tf batch normalization
def batch_norm(x, name="batch_norm"):
    return tf.contrib.layers.batch_norm(x, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, scope=name)

# Convolution layer
def conv_layer(x, features, ks=3, s=1, b_shape=None, padding='SAME', 
               activation=tf.nn.relu, train=True, reuse=False, name='conv2d'):
    with tf.variable_scope(name):
        conv = tf.contrib.layers.conv2d(inputs=x, num_outputs=features, kernel_size=ks,
                                        stride=s, padding=padding, activation_fn=activation, 
                                        trainable=train, reuse=reuse)
        return conv

def residule_block(x, dim, ks=3, s=1, name='res'):
    p = int((ks - 1) / 2)
    y = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
    y = ops.conv_layer(y, dim, ks=ks, s=s, activation=None, padding='VALID', name=name+'_c1')
    y = norm(y, name=name+'_bn1')
    y = tf.pad(tf.nn.relu(y), [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
    y = ops.conv_layer(y, dim, ks=ks, s=s, activation=None, padding='VALID', name=name+'_c2')
    y = norm(y, name=name+'_bn2')
    return y + x
