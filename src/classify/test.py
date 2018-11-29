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

def run_test(args, image_size=[64,64], channels=3):
    tf.reset_default_graph()
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
        net_in = resnet.ResNet_34(args, x, image_size, type_='cifar-10', pooling='avg')
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
    # restore our boy
    check_point = tf.train.latest_checkpoint(args.save_dir)
    new_saver = tf.train.Saver()
    new_saver.restore(sess, check_point)
    accs = []
    rmses = []
    # now lets start loading images and giving it to the network to make a prediction
    for path in images_in:
        print('\r{}'.format(path), end='')
        image, label = util.get_test_image(path, label_split='+', classes=cifar_classes)
        # load image
        acc_, rmse_ = sess.run([accuracy, rmse],
                      feed_dict={x:batch_images, labels:batch_labels})
        accs.append(acc_)
        rmses.append(rmse_)

    # calculate averages
    avg_acc = round(sum(accs) / len(accs), 3)
    avg_rmse = round(sum(rmses) / len(rmses), 3)
    with open('test_results.txt', 'w') as file:
        line = 'ACC: {} RMSE: {}'.format(avg_acc, avg_rmse)
        print(line)
        file.write(line + '\n')
        file.close()
    # # #

# This is for running a single image test
# - good for deployment
def run_single(args, image_path, image_size=[64,64], channels=3):
    cifar_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    cifar_spliter = '+'

    tf.reset_default_graph() # If we have a graph 
    with tf.variable_scope('input'):
        x = tf.placeholder(tf.float32,[None, image_size[1], image_size[0], channels],name='image_in') #
        labels = tf.placeholder(tf.float32, [None, args.num_classes] ,name='label_in')
    
    with tf.variable_scope('network'):
        net_in = resnet.ResNet_34(args, x, image_size, type_='cifar-10', pooling='avg')
        logits, _ = resnet.inference(args, net_in, cifar_classes)
    
        #loss = tf.nn.softmax_cross_entropy_with_logits(labels=tf.to_int64(labels), logits=[logits])
        # Just our metrics now
        arg_logit = tf.argmax(logits, -1)
        arg_label = tf.argmax(tf.to_int64(labels),-1)
    
    sess = util.get_session()
    # restore our boy
    with sess:
        check_point = tf.train.latest_checkpoint(args.save_dir)
        new_saver = tf.train.Saver()
        new_saver.restore(sess, check_point)

        image, label = util.get_test_image(image_path, w=image_size[0], h=image_size[1], label_split='+', classes=cifar_classes)
        # load image
        predicted_label = sess.run([arg_logit], feed_dict={x:[image]})
        # now process it a little
        predicted_label = cifar_classes[predicted_label[0][0]]
        print(predicted_label)
        sess.close()
    return predicted_label
