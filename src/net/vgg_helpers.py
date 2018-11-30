import tensorflow as tf
import numpy as np
import src.net.model_helpers as mh
# network types:

# This is one of the vgg implementations
# - it does include an infrence as well
def get_pool(X, pool='max'):
    if(pool == 'avg'):
        return avg_pool(X,'pool',stride=2)
    if(pool == 'max'):
        return max_pool(X,'pool',stride=2)

# Note for cifar-10 we cannot use five layers of vgg16, instead we will take out the second one?
# - again this is only for cifar-10
def vgg16(dict_,cfgs,inputs,train=True,mnist=False):
    with tf.variable_scope('process'):
        # unflatten the images
        in_shape = inputs.get_shape().as_list()
        print(in_shape)
        if(mnist):
            inputs = tf.reshape(inputs, [-1,28,28,1]) # for mnist only
            inputs = tf.image.grayscale_to_rgb(inputs)
        print(inputs.shape)
    with tf.variable_scope('vgg16'):
        conv1_1 = conv_layer_vgg(dict_,inputs, "conv1_1")
        conv1_2 = conv_layer_vgg(dict_,conv1_1, "conv1_2")
        pool1 = max_pool(conv1_2,'pool1')
        print(pool1.shape)

        conv2_1 = conv_layer_vgg(dict_,pool1, "conv2_1")
        conv2_2 = conv_layer_vgg(dict_,conv2_1, "conv2_2")
        pool2 = max_pool(conv2_2,'pool2')
        # done making conv block 2 side 2

        conv3_1 = conv_layer_vgg(dict_,pool2,"conv3_1") # configed for cifar-10
        conv3_2 = conv_layer_vgg(dict_,conv3_1,"conv3_2")
        conv3_3 = conv_layer_vgg(dict_,conv3_2,"conv3_3")
        features = conv3_3.get_shape().as_list()[-1]
        pool3 = max_pool(conv3_3,'pool3')
        # done making conv block 3 side 3
        print(pool3.shape)
        
        conv4_1 = conv_layer_vgg(dict_,pool3,"conv4_1")
        conv4_2 = conv_layer_vgg(dict_,conv4_1,"conv4_2")
        conv4_3 = conv_layer_vgg(dict_,conv4_2,"conv4_3")
        features = conv4_3.get_shape().as_list()[-1]
        pool4 = max_pool(conv4_3, 'pool4')
        # done making conv block 4 side 4
        print(pool4.shape)

        conv5_1 = conv_layer_vgg(dict_,pool4, "conv5_1")
        conv5_2 = conv_layer_vgg(dict_,conv5_1,"conv5_2")
        conv5_3 = conv_layer_vgg(dict_,conv5_2,"conv5_3")
        features = conv5_3.get_shape().as_list()[-1]
        pool5 = max_pool(conv5_3, 'pool4')
        pool5_shape = pool5.get_shape().as_list()[1:]
        print(pool5.shape)


        # This is the fully connected section
        with tf.variable_scope('fc_1'):		
            dim = mh.get_dim(pool5)
            print(dim)
            x = tf.reshape(pool5, [-1, dim])
            weights_1 = mh.weight_variable_tn([dim,256],'fc_weight')
            biases_1 = mh.biase_variable_tn([256], 'fc_bias')
            #print(x.shape)
            fc_1 = tf.nn.bias_add(tf.matmul(x, weights_1), biases_1)
            fc_1 = tf.nn.relu(fc_1)
            if(train):
                # do our dropout
                fc_1 = tf.nn.dropout(fc_1, cfgs['model']['dropout'])

        print(fc_1.shape)
        with tf.variable_scope('fc_2'):
            weights_2 = mh.weight_variable_tn([256,128],'fc_weight')
            biases_2 = mh.biase_variable_tn([128], 'fc_bias')
            fc_2 = tf.nn.bias_add(tf.matmul(fc_1, weights_2), biases_2)
            fc_2 = tf.nn.relu(fc_2)
            if(train):
                # do our dropout
                fc_2 = tf.nn.dropout(fc_2,cfgs['model']['dropout'])

        print(fc_2.shape)#'''
        with tf.variable_scope('fc_3'):
            weights_3 = mh.weight_variable_tn([128,cfgs['data']['num_classes']],'fc_weight')
            biases_3 = mh.biase_variable_tn([cfgs['data']['num_classes']], 'fc_bias')
            fc_3 = tf.nn.bias_add(tf.matmul(fc_2, weights_3), biases_3) # might need to make changes here, I think 
            softmax = tf.nn.softmax(fc_3, name='probs')             #  that the softmax might need to be the logits?
            print(fc_3.shape)
            return fc_3,softmax
# 
def vgg_infrence(cfgs, net_in, train=True, fc_features=[256,128]):
    # This is the fully connected section
    with tf.variable_scope('fc_1'):		
        dim = mh.get_dim(net_in)
        print(dim)
        x = tf.reshape(net_in, [-1, dim])
        weights_1 = mh.weight_variable_tn([dim,fc_features[0]],'fc_weight')
        biases_1 = mh.biase_variable_tn([fc_features[0]], 'fc_bias')
        #print(x.shape)
        fc_1 = tf.nn.bias_add(tf.matmul(x, weights_1), biases_1)
        fc_1 = tf.nn.relu(fc_1)
        if(train):
            # do our dropout
            fc_1 = tf.nn.dropout(fc_1, cfgs['model']['dropout'])

    print(fc_1.shape)
    with tf.variable_scope('fc_2'):
        weights_2 = mh.weight_variable_tn([fc_features[0],fc_features[1]],'fc_weight')
        biases_2 = mh.biase_variable_tn([fc_features[1]], 'fc_bias')
        fc_2 = tf.nn.bias_add(tf.matmul(fc_1, weights_2), biases_2)
        fc_2 = tf.nn.relu(fc_2)
        if(train):
            # do our dropout
            fc_2 = tf.nn.dropout(fc_2,cfgs['model']['dropout'])

    print(fc_2.shape)#'''
    with tf.variable_scope('fc_3'):
        weights_3 = mh.weight_variable_tn([fc_features[1],cfgs['data']['num_classes']],'fc_weight')
        biases_3 = mh.biase_variable_tn([cfgs['data']['num_classes']], 'fc_bias')
        fc_3 = tf.nn.bias_add(tf.matmul(fc_2, weights_3), biases_3) # might need to make changes here, I think 
        softmax = tf.nn.softmax(fc_3, name='probs')             #  that the softmax might need to be the logits?
        print(fc_3.shape)
        return fc_3,softmax


# Mainly helpers below

def conv_layer_vgg(dict_,bottom, name):
    #Adding a conv layer + weight parameters from a dict
        
    with tf.variable_scope(name):
        filt = get_conv_filter(dict_,name)

        conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

        conv_biases = get_bias(dict_,name)

        bias = tf.nn.bias_add(conv, conv_biases)

        relu = tf.nn.relu(bias)
        return relu

# a basic conv layer with some extra customization
def conv_layer(x, filter_x,filter_y,output_features,stride=1,name='conv_layer',padding='SAME',activation=tf.nn.relu):
    in_channel = x.get_shape().as_list()[-1]
    kernel = (filter_x, filter_y)
    conv = tf.layers.conv2d(x,output_features,kernel,strides=(stride,stride),kernel_initializer=tf.contrib.layers.xavier_initializer(),activation=activation,padding=padding)
    return conv

# our deconv layer
def deconv_layer(x,upscale,k_shape=4, name='deconv_layer',pad='SAME'):
    x_shape = tf.shape(x)
    in_shape = x.get_shape().as_list()

    w_shape = [k_shape, k_shape, in_shape[-1], in_shape[-1]]
    strides = [1, upscale, upscale, 1]
    
    W = mh.weight_variable(w_shape)
    #print(W.shape)

    out_shape = [x_shape[0], x_shape[1] * upscale, x_shape[2] * upscale, in_shape[-1]]

    return tf.nn.conv2d_transpose(x, W, out_shape, strides=strides, padding=pad)

# This is where we can add our fully connected layer
def get_conv_filter(dict_, name):
    return tf.constant(dict_[name][0], name="filter")

def get_bias(dict_, name):
    return tf.constant(dict_[name][1], name="biases")

def weight_variable(shape, initial):

    init = initial(shape)
    return tf.Variable(init)

def bias_variable(shape, initial):

    init = initial(shape)
    return tf.Variable(init)

def pad(x,size):
    " This is for padding images"
    paddings = [[0,0], [size,size],[size,size], [0, 0]]
    return tf.pad(x, paddings)

# shorten the max pool
def max_pool(bottom, name,stride=2): # make sure These filter sizes are right, I dont know if they should be 2x2
    return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, stride, stride, 1], padding='SAME', name=name)
    
def avg_pool(bottom, name,stride=2):
    return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, stride, stride, 1], padding='SAME', name=name)


# for cropping one layer to match another
def crop(input_1, input_2):
    shape_1 = tf.shape(input_1)
    shape_2 = tf.shape(input_2)
    # offsets for the top left corner of the crop
    offsets = [0, (shape_1[1] - shape_2[1]) // 2, (shape_1[2] - shape_2[2] // 2), 0]
    size = [-1, shape[2], shape[2], -1]
    output = tf.slice(input_1, offsets, size)
    return output
