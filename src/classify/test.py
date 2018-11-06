import tensorflow as tf
import numpy as np
import os
from glob import glob
from termcolor import colored, cprint
import time

# custom
import src.net.vgg_helpers as vgg
import src.net.ResNet as resnet
import src.util as util

def test(args, image_size=[64,64], channels=3):
    accs  = []
    os.makedirs(args.save_dir, exist_ok=True)
    with tf.variable_scope('input'):
        x = tf.placeholder(tf.float32,[None, image_size[1], image_size[0], channels],name='image_in') #
        labels = tf.placeholder(tf.float32, [None, args.num_classes] ,name='label_in')

    # setup our data
    images_in = glob(args.data_path + 'test/*.png')
    cifar_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    cifar_spliter = '+'
    # now setup the network
    with tf.variable_scope('network'):
        net_in = resnet.ResNet_18(args, x, image_size, type_='cifar-10', pooling='avg')
        logits, _ = resnet.inference(args, net_in, cifar_classes)
    
    with tf.variable_scope('train'):
        with tf.variable_scope('loss'):
            #loss = tf.nn.softmax_cross_entropy_with_logits(labels=tf.to_int64(labels), logits=[logits])
            # Just our metrics now
            arg_logit = tf.argmax(logits, -1)
            arg_label = tf.argmax(tf.to_int64(labels),-1)
            # accuracy and rmse
            correct = tf.equal(arg_logit, arg_label)
            accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
            _op, rmse = tf.metrics.mean_squared_error(arg_label, arg_logit)
    
    sess = util.get_session()
    
    for path in images_in:
        print('\r{}'.format(path), end='')
        # load image