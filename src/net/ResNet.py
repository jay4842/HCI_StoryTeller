import tensorflow as tf
import numpy as np
import src.net.model_helpers as mh
import src.net.vgg_helpers as vgg
# https://github.com/tensorflow/models/tree/master/official/resnet
# main model class I need: 
# - https://github.com/tensorflow/models/blob/master/official/resnet/resnet_model.py 

# - Using the tensorflow implementation for this guy.
_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5

# Network def here
# ResNet 18
def ResNet_18(args, inputs, image_size, type_, pooling='wave', train=True):
    with tf.variable_scope('ResNet'):
        if(type_ == 'MNIST'):
            # we need to reshape our inputs
            inputs = tf.reshape(inputs, [-1,28,28,1])
            inputs = tf.image.grayscale_to_rgb(inputs)
            inputs = tf.image.resize_images(inputs, (image_size[1], image_size[0]))
        #
        in_shape = inputs.get_shape().as_list()
        print(in_shape)
        # 
        # the initial [7 x 7 x 64] convolution
        inputs = conv2d_fixed_padding(inputs=inputs, filters=64, kernel_size=7, strides=2)
        print(inputs.get_shape().as_list())
        # here a pool would be used
        with tf.variable_scope('initial_pool'):
            shape = inputs.get_shape().as_list()
            inputs = get_pool(inputs, pooling='max') # ResNet uses max pooling here [we could change the pooling here]
            if(pooling=='wave'):
                inputs = tf.reshape(inputs, [-1, shape[1]//2, shape[2]//2, shape[-1]])

        print(inputs.get_shape().as_list())
        # now the four blocks
        block_1 = block_layer(inputs, filters=64, num_blocks=2, strides=1, training=train, name='conv1_x')
        print(block_1.get_shape().as_list())

        block_2 = block_layer(block_1, filters=128, num_blocks=2, strides=2, training=train, name='conv2_x')
        print(block_2.get_shape().as_list())

        block_3 = block_layer(block_2, filters=256, num_blocks=2, strides=2, training=train, name='conv3_x')
        print(block_3.get_shape().as_list())

        block_4 = block_layer(block_3, filters=512, num_blocks=2, strides=1, training=train, name='conv4_x')
        print(block_4.get_shape().as_list())
        # apply another pool here

        with tf.variable_scope('pool_out'):
            shape = block_4.get_shape().as_list()
            #axes = [1, 2]
            #pool_out = tf.reduce_mean(block_4, axes)
            pool_out = get_pool(block_4, pooling=pooling) # for now this is the one we will mess with
            if(pooling=='wave'):
                pool_out = tf.reshape(pool_out, [-1, shape[1]//2, shape[2]//2, shape[-1]])
            print(pool_out.get_shape().as_list())
            return pool_out

# ResNet 34
def ResNet_34(args, inputs, image_size, type_, pooling='wave', train=True):
    with tf.variable_scope('ResNet'):
        if(type_ == 'MNIST'):
            # we need to reshape our inputs
            inputs = tf.reshape(inputs, [-1,28,28,1])
            inputs = tf.image.grayscale_to_rgb(inputs)
            inputs = tf.image.resize_nearest_neighbor(inputs, (image_size[1], image_size[0]))
        #
        in_shape = inputs.get_shape().as_list()
        print(in_shape)
        # 
        # the initial [7 x 7 x 64] convolution
        inputs = conv2d_fixed_padding(inputs=inputs, filters=64, kernel_size=7, strides=2)
        print(inputs.get_shape().as_list())
        # here a pool would be used
        with tf.variable_scope('initial_pool'):
            shape = inputs.get_shape().as_list()
            inputs = get_pool(inputs, pooling='max') # ResNet uses max pooling here [we could change the pooling here]
            #if(pooling=='wave'):
            #	inputs = tf.reshape(inputs, [-1, shape[1]//2, shape[2]//2, shape[-1]])

        print(inputs.get_shape().as_list())
        # now the four blocks
        block_1 = block_layer(inputs, filters=64, num_blocks=3, strides=1, training=train, name='conv1_x')
        print(block_1.get_shape().as_list())

        block_2 = block_layer(block_1, filters=128, num_blocks=4, strides=2, training=train, name='conv2_x')
        print(block_2.get_shape().as_list())

        block_3 = block_layer(block_2, filters=256, num_blocks=6, strides=2, training=train, name='conv3_x')
        print(block_3.get_shape().as_list())

        block_4 = block_layer(block_3, filters=512, num_blocks=3, strides=2, training=train, name='conv4_x')
        print(block_4.get_shape().as_list())
        # apply another pool here

        with tf.variable_scope('pool_out'):
            shape = block_4.get_shape().as_list()
            axes = [1, 2]
            pool_out = tf.reduce_mean(block_4, axes)
            #pool_out = get_pool(block_4, pooling='avg') # for now this is the one we will mess with
            #if(pooling=='wave'):
                #pool_out = tf.reshape(pool_out, [-1, shape[1]//2, shape[2]//2, shape[-1]])
            print(pool_out.get_shape().as_list())
            return pool_out
# an inference function to get logits and predictions.
# - uses a tf.dense layer as a fully connected layer
def inference(args, inputs, classes, train=True):
    # flatten the input, get our classes
    num_classes = len(classes)
    dim = mh.get_dim(inputs)

    inputs = tf.reshape(inputs, [-1, dim])
    # feed through a tf.dense_layer
    inputs = tf.layers.dense(inputs=inputs, units=num_classes)
    logits = tf.identity(inputs, 'logits')
    probs = tf.nn.softmax(logits, name='probs')
    return logits, probs
# helper functions here!
# Need to finish fleshing this guy out.

# helper shorten my pooling
def get_pool(X, pooling='max'):
    if(pooling == 'avg'):
        return vgg.avg_pool(X,'pool',stride=2)
    if(pooling == 'max'):
        return vgg.max_pool(X,'pool',stride=2)

# From resNet tensorflow
# - they use tf.layers batch norm instead of tf.nn. 
def batch_norm(inputs, training, data_format='channels_last'):
    """Performs a batch normalization using a standard set of parameters."""
    # We set fused=True for a significant performance boost. See
    # https://www.tensorflow.org/performance/performance_guide#common_fused_ops
    return tf.layers.batch_normalization( inputs=inputs, axis=3, momentum=_BATCH_NORM_DECAY,
     epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=training, fused=True)

# TODO
# - Fixed padding
# From the tensorflow ResNet implementation.
def fixed_padding(inputs, kernel_size): 
    with tf.variable_scope('fixed_padding'):
        # channels are last for me
        pad_total = kernel_size - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg

        padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
        return padded_inputs
# end of that guy

# TODO
# - conv_2d fixed padding
def conv2d_fixed_padding(inputs, filters, kernel_size, strides, name='conv_2d_fixed_padding', data_format='channels_last'): # I shouldn't have to change this
    with tf.variable_scope(name):
        if strides > 1:
            inputs = fixed_padding(inputs, kernel_size)

        return tf.layers.conv2d( inputs=inputs, filters=filters, kernel_size=kernel_size, 
            strides=strides, padding=('SAME' if strides == 1 else 'VALID'), use_bias=False,
            kernel_initializer=tf.variance_scaling_initializer(),data_format=data_format)

# TODO
# - Make projection shortcut a function
def projection_shortcut(inputs, filters_out, strides):
    with tf.variable_scope('projection_shortcut'):
        return conv2d_fixed_padding(inputs=inputs, filters=filters_out, kernel_size=1, strides=strides)

# Net building blocks here
# NOTE: 
# - the building block residual_block is a version 1 block that is simplified.
def res_block_v1(inputs, filters, filters_out, training, projection_shortcut, strides, name, data_format='channels_last'):
    with tf.variable_scope(name):
        '''
        - setup shortcut
        - conv_2d_fixed_padding
        - batch_norm
        - activation
        - conv_2d_fixed_padding
        - batch_norm
        - add shortcut
        - activation
        '''
        shortcut = inputs
        if projection_shortcut is not None:
            shortcut = projection_shortcut(inputs, filters_out, strides)
            shortcut = batch_norm(inputs=shortcut, training=training)

        # now the two convolutions
        # conv_layer 1
        inputs = conv2d_fixed_padding(inputs=inputs, filters=filters, kernel_size=3, strides=strides, name='conv_2d_1')
        inputs = batch_norm(inputs=inputs, training=training)
        inputs = tf.nn.relu(inputs)

        # conv_layer 2
        inputs = conv2d_fixed_padding(inputs=inputs, filters=filters, kernel_size=3, strides=1, name='conv_2d_2')
        inputs = batch_norm(inputs=inputs, training=training)
        inputs += shortcut
        inputs = tf.nn.relu(inputs)

        return inputs
# a bit different than the previous implementation (cleaner than the older one)

def res_block_v2(inputs, filters, filters_out, training, projection_shortcut, strides, name, data_format='channels_last'):
    
    with tf.variable_scope(name):
        shortcut = inputs
        inputs = batch_norm(inputs, training, data_format)
        inputs = tf.nn.relu(inputs)

        # projection shortcut?
        if(projection_shortcut is not None):
            shortcut = projection_shortcut(shortcut)

        
# also from tensorflow ResNet
def block_layer(inputs, filters, num_blocks, strides, training, name, bottleneck=False,
                 block_function=res_block_v1, data_format='channels_last'):
    with tf.variable_scope(name):
        # This guy will create one layer of blocks
        # if there is a bottle neck the out features is 4x the input
        filters_out = filters * 4 if bottleneck else filters

        # projection and striding is only applied to the first layer
        inputs = block_function(inputs, filters, filters_out, training, projection_shortcut, strides, 'block_0', data_format)
        
        # Now add blocks using our block_function
        for _ in range(1, num_blocks):
            inputs = block_function(inputs, filters, filters_out, training, None, 1, 'block_{}'.format(_), data_format)

        return tf.identity(inputs, name)



