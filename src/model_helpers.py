import numpy as np
import tensorflow as tf
import time
import os
import sys
import platform
from PIL import Image
from glob import glob
import cv2

# multi threading
from datetime import timedelta
import time
import multiprocessing
from multiprocessing import Process

from src.util import sigmoid_cross_entropy_balanced
from src.image_helpers import show

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Moving major helpers to this guy
def sub_side_layer(inputs, name, upscale=2): # we usually will upscale by 2 becasue we downscale by 2
	"""
        https://github.com/s9xie/hed/blob/9e74dd710773d8d8a469ad905c76f4a7fa08f945/examples/hed/train_val.prototxt#L122
        1x1 conv followed with Deconvoltion layer to upscale the size of input image sans color
	"""
	#print(inputs.shape.as_list())
	with tf.variable_scope(name):

		in_shape = inputs.shape.as_list()
		w_shape = [1, 1, in_shape[-1], 3]
		#print('w_shape {}\n'.format(w_shape))
		classifier = conv_layer(inputs, w_shape, b_shape=1,
                                        w_init=tf.constant_initializer(),
                                        b_init=tf.constant_initializer(),
                                        name=name + '_reduction')

			
		classifier = max_pool(classifier,name+'_pool')
			
		# this applys the crop but not sigmoid
			
		classifier = deconv_layer(classifier, upscale=upscale,
                                        name='{}_deconv_{}'.format(name, upscale),
                                        w_init=tf.truncated_normal_initializer(stddev=0.1),
                                        out_channel=3)
			
		# This will make the values range from 0 to 1
		#classifier = tf.sigmoid(classifier)


		return classifier

def get_dim(x):
	shape = x.get_shape().as_list()
	dim = 1
	for d in shape[1:]:
		dim *= d
	return dim
	
def weight_variable_tn(shape, name):
	init = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(init, name=name)

def biase_variable_tn(shape, name):
	init = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(init, name=name)

def weight_variable(shape, initial):
	init = initial(shape)
	return tf.Variable(init)

def bias_variable(shape, initial):
	init = initial(shape)
	return tf.Variable(init)

def pad(x,size):
	" This is for padding images"
	with tf.variable_scope("PAD"):
		paddings = [[0,0], [size,size],[size,size], [0, 0]]
		return tf.pad(x, paddings)

# shorten the max pool
def max_pool(bottom,stride=2,pad='SAME', name='max_pool'):
	with tf.variable_scope(name):
		return tf.nn.max_pool(bottom, ksize=[1, stride, stride, 1], strides=[1, stride, stride, 1], padding=pad, name=name)

# This is a huge work in progress for another model structure idea
def unpool(value, name='unpool'):
	'''N-dimensional version of the unpooling operation from
	https://www.robots.ox.ac.uk/~vgg/rg/papers/Dosovitskiy_Learning_to_Generate_2015_CVPR_paper.pdf
		
	:param value: A Tensor of shape [b, d0, d1, ..., dn, ch]
	:return: A Tensor of shape [b, 2*d0, 2*d1, ..., 2*dn, ch]
	'''
	print('unpooling')
	with tf.variable_scope(name) as scope:
		print('made_list')
		sh = value.get_shape().as_list()
		dim = len(sh[1:-1])
		out = (tf.reshape(value, [-1] + sh[-dim:]))
		print('starting loop')
		for i in range(dim, 0, -1):
			print('dim[i] = {}'.format(out[i]))
			print('iteration #{}'.format(i))
			# This is where we mess up
			out = tf.concat(i, [out, tf.zeros_like(out)])
			print('out = {}'.format(out))

		out_size = [-1] + [s * 2 for s in sh[1:-1]] + [sh[-1]]
		out = tf.reshape(out, out_size, name=scope)
			
			
	return out

# for cropping one layer to match another
def crop(input_1, input_2):
	with tf.variable_scope("CROP"):
		shape_1 = tf.shape(input_1)
		shape_2 = tf.shape(input_2)
		# offsets for the top left corner of the crop`
		offsets = [0, (shape_1[1] - shape_2[1]) // 2, (shape_1[2] - shape_2[2] // 2), 0]
		size = [-1, shape[2], shape[2], -1]
		output = tf.slice(input_1, offsets, size)
		return output

# due to how the vgg is loaded we have to leave it alone
# But we can change how our side layers are predicted
# In the origin HED they apply sigmoid after deconvolution
def side_layer(inputs, name, upscale):
	"""
        https://github.com/s9xie/hed/blob/9e74dd710773d8d8a469ad905c76f4a7fa08f945/examples/hed/train_val.prototxt#L122
        1x1 conv followed with Deconvoltion layer to upscale the size of input image sans color
	"""
	#print(inputs.shape.as_list())
	with tf.variable_scope(name):

		in_shape = inputs.shape.as_list()
		w_shape = [1, 1, in_shape[-1], 1]
		#print('w_shape {}\n'.format(w_shape))
		classifier = conv_layer(inputs, w_shape, b_shape=1,
                                       w_init=tf.constant_initializer(),
                                       b_init=tf.constant_initializer(),
                                       name=name + '_reduction')

		# this applys the crop but not sigmoid
		classifier = deconv_layer(classifier, upscale=upscale,
                                        name='{}_deconv_{}'.format(name, upscale),
                                        w_init=tf.truncated_normal_initializer(stddev=0.1))

			
		# This will make the values range from 0 to 1
		#classifier = tf.sigmoid(classifier)

		#print(classifier.shape)
		#input('->')
		return classifier


def conv_2d(x, filter_shape, stride, name):
	with tf.variable_scope(name):
		out_channel = filter_shape[3]

		init = tf.truncated_normal(shape, stddev=0.1)
		filter_ = tf.Variable(init, name='conv_2d_filter_weight')

		conv = tf.nn.conv2d(x, filter=filter_,strides=[1,stride,stride,1], padding='SAME')

		return conv
		

def conv_layer(x, W_shape, b_shape=None, name=None,
					padding='SAME', use_bias=True, w_init=None, b_init=None):

	with tf.variable_scope(name):
		W = weight_variable(W_shape, w_init)
		#tf.summary.histogram('weights_{}'.format(name), W)

		if use_bias:
			b = bias_variable([b_shape], b_init)
			#tf.summary.histogram('biases_{}'.format(name), b)

		conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding=padding)
		return conv + b if use_bias else conv

def deconv_layer(x, upscale, name, padding='SAME', w_init=None,out_channel=1):
	with tf.variable_scope(name):
		x_shape = tf.shape(x)
		in_shape = x.shape.as_list()

		w_shape = [upscale * 2, upscale * 2, in_shape[-1], out_channel]
		strides = [1, upscale, upscale, 1]

		W = weight_variable(w_shape, w_init)
		#tf.summary.histogram('weights_{}'.format(name), W)

		out_shape = tf.stack([x_shape[0], x_shape[1], x_shape[2], w_shape[3]]) * tf.constant(strides, tf.int32)
		deconv = tf.nn.conv2d_transpose(x, W, out_shape, strides=strides, padding=padding)

		return deconv


# A LOT OF HELPERS INSPIRED BY DUGAN OPS
def lrelu(x):
	with tf.variable_scope('lrelu') as scope:
		if x.dtype is not tf.complex64:
			return tf.nn.leaky_relu(x)
		else:
			return x

def relu(x):
	with tf.variable_scope('relu') as scope:
		if x.dtype is not tf.complex64:
			return tf.nn.relu(x)
		else:
			return x

def delist(net):
	if type(net) is list: # if the value is a list delist it
		net = tf.concat(net,-1,name = 'cat')
	return net

# another way  to define a conv_2d
def conv2d(net, filters, kernel = 3, stride = 1, dilation_rate = 1, activation = relu, 
			padding = 'SAME', trainable = True, name = None, reuse = None):
	# define the output
	net = tf.layers.conv2d(delist(net),filters,kernel,stride,padding,dilation_rate = dilation_rate, 
			activation = activation,trainable = trainable, name = name, reuse = reuse)
	return net

def batch_norm_dugan(net,training,trainable,activation = relu):
	with tf.variable_scope('Batch_Norm'):
		#FLAGS.bn_scope = FLAGS.bn_scope + 1
		net = tf.layers.batch_normalization(delist(net),training = training, trainable = trainable)
		if activation is not None:
			net = activation(net)

		return net

def bn_conv2d(net, training, filters, kernel = 3, stride = 1, dilation_rate = 1, activation = lrelu, use_bias = False, padding = 'SAME', trainable = True, name = 'bn_convd2', reuse = None):
	with tf.variable_scope(name) as scope:
		net = conv2d(delist(net), filters, kernel, stride, dilation_rate, activation, padding, trainable, name, reuse)
		net = batch_norm(net,training,trainable,activation)
		return net

# an average pool
def avg_pool(net, kernel = 3, stride = 1, padding = 'SAME', name = 'avg_pool'):
	with tf.variable_scope(name):
		return tf.layers.average_pooling2d(net,kernel,stride,padding=padding,name=name)

# another max pool
def max_pool_d(net, kernel = 3, stride = 3, padding = 'SAME', name = 'max_pool'):
	with tf.variable_scope(name):
		return tf.layers.max_pooling2d(net,kernel,stride,padding=padding,name=name)

# dense block

def dense_block(net,training, filters = 2, kernel = 3, kmap = 5, stride = 1,
                activation = relu, padding = 'SAME', trainable = True,
                name = 'Dense_Block', prestride_return = True,use_max_pool = True, batch_norm = True):
	with tf.variable_scope(name) as scope:

		net = delist(net)

		for n in range(kmap):
			out = conv2d(net,filters=filters,kernel=kernel,stride=1,activation=activation,padding=padding,trainable=trainable,name = '_map_%d'%n)
			net = tf.concat([net,out],-1,name = '%d_concat'%n)

		if batch_norm:
			net = batch_norm(net,training,trainable)

		if stride is not 1:
			prestride = net
			if use_max_pool:
				net = max_pool_d(net,stride,stride)
			else:
				net = avg_pool(net,stride,stride)
			if prestride_return:
				return prestride, net
			else:
				return net

		else:
			return net
# debse reduction
def dense_reduction(net,training, filters = 2, kernel = 3, kmap = 5, stride = 1,
                activation = lrelu, trainable = True,dropout = True, name = 'Dense_Block'):
	with tf.variable_scope(name) as scope:
		net = delist(net)
		for n in range(kmap):
			out = conv2d(net, filters=filters, kernel=kernel, stride=1,
                        activation=None, trainable=trainable, name = '_map_%d'%n)
			net = tf.concat([net,out],-1,name = '%d_concat'%n)
		net = batch_norm(net,training,trainable,activation)
		if(dropout): net = tf.nn.dropout(net,.85)
		if stride is not 1:
			net = max_pool_d(net,kernel=stride,stride=stride)
		return net
# end of dense boy

def deconv(net, features = 3, kernel = 3, stride = 2, activation = None,padding = 'SAME', trainable = True, name = None):
	return tf.layers.conv2d_transpose(net,features,kernel,stride,activation=activation,padding=padding,trainable=trainable,name=name,use_bias = False)

def new_deconvxy(net, stride = 2, padding = 'SAME', trainable = True, dropout = True,kernel = 3, name = 'Deconv_xy'):
	with tf.variable_scope(name) as scope:
		net = delist(net)
		kernel = kernel
		features = net.shape[-1].value // stride

		netx = deconv(net , features  , kernel = kernel, stride = (stride,1), name = "x",  trainable = trainable)
		nety = deconv(net , features  , kernel = kernel, stride = (1,stride), name = "y",  trainable = trainable)

		features = net.shape[-1].value // stride

		netxy = deconv(net , features  , kernel = kernel, stride = stride    , name = 'bolth', trainable = trainable)
		if(dropout): netxy = tf.nn.dropout(netxy, .85)
		netx = deconv(netx, features  , kernel = kernel, stride = (1,stride), name = "xy"   , trainable = trainable)
		if(dropout): netx = tf.nn.dropout(netx, .85)
		nety = deconv(nety, features  , kernel = kernel, stride = (stride,1), name = "yx"   , trainable = trainable)
		if(dropout): nety = tf.nn.dropout(nety, .85)

		net  = tf.concat((netx,nety,netxy),-1)
		return net

def atrous_block(net,training,filters = 8,kernel = 3,dilation = 1,kmap = 5,stride = 1,activation = relu,trainable = True,padding='SAME',name = 'Atrous_Block'):
	newnet = []
	with tf.variable_scope(name) as scope:
		for x in range(dilation,kmap * dilation,dilation):
			# Reuse and not trainable if beyond the first layer.
			re = True  if x > dilation else None
			tr = False if x > dilation else trainable

			with tf.variable_scope("ATROUS",reuse = tf.AUTO_REUSE) as scope:
				# Total Kernel visual size: Kernel + ((Kernel - 1) * (Dilation - 1))
				# At kernel = 9 with dilation = 2; 9 + 8 * 1, 17 px
				layer = conv2d(net,filters = filters, kernel = kernel, dilation_rate = x,reuse = re,trainable = tr,padding=padding)
				newnet.append(layer)

		net = delist(newnet)
		net = bn_conv2d(net,training,filters = filters,kernel = stride,stride = stride,trainable = trainable,name = 'GradientDisrupt',activation = activation)
		return net

#FROM  https://github.com/machrisaa/tensorflow-vgg --> made some changes to it
def fc_layer(bottom, in_size, out_size,w_shape,b_shape, name):
	with tf.variable_scope(name):
		weights = weight_variable_tn(w_shape,'fc_weight')
		biases = biase_variable_tn(b_shape, 'fc_bias')
			
		x = tf.reshape(bottom, [-1, in_size])
		fc = tf.nn.bias_add(tf.matmul(x, weights), biases)
			
		return fc

def batch_norm(x, name="batch_norm"):
    return tf.contrib.layers.batch_norm(x, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, scope=name)
