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

def train(args, image_size=[64,64], channels=3):
    accs  = []
    os.makedirs(args.save_dir, exist_ok=True)
    with tf.variable_scope('input'):
        x = tf.placeholder(tf.float32,[None, image_size[1], image_size[0], channels],name='image_in') #
        labels = tf.placeholder(tf.float32, [None, args.num_classes] ,name='label_in')

    # setup our data
    images_in = glob(args.data_path + 'train/*.png')
    train_ids = list(range(0,len(images_in)))
    np.random.shuffle(train_ids)
    epoch_size = (len(images_in) // args.batch_size)

    cifar_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    cifar_spliter = '+'
    # now setup the network
    with tf.variable_scope('network'):
        net_in = resnet.ResNet_18(args, x, image_size, type_='cifar-10', pooling='avg')
        logits, _ = resnet.inference(args, net_in, cifar_classes)
    
    with tf.variable_scope('train'):
        with tf.variable_scope('loss'):
            loss = tf.nn.softmax_cross_entropy_with_logits(labels=tf.to_int64(labels), logits=[logits])
            arg_logit = tf.argmax(logits, -1)
            arg_label = tf.argmax(tf.to_int64(labels),-1)
            correct = tf.equal(arg_logit, arg_label)
            accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
            _op, rmse = tf.metrics.mean_squared_error(arg_label, arg_logit)

            reduce_loss = tf.reduce_mean(loss)

        with tf.variable_scope('optimize'):
            # first get our optimizer
            learning_rate = tf.placeholder('float', [])
            opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
            train_op = opt.minimize(reduce_loss)
    
    sess = util.get_session()
    save_interval =  int(epoch_size/5)

    with sess:
        start_time = time.time()
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        base_lr = args.learning_rate
        
        # now start the epochs
        for e in range(args.epochs):
            with open(args.save_dir+'train_out_{}.txt'.format(e), 'w+') as out_file:
                for idx in range(epoch_size):
                    batch_images, batch_labels = util.get_batch(images_in, train_ids, 
                                                batch_size=args.batch_size, label_split=cifar_spliter, 
                                                classes=cifar_classes, standardize=True, 
                                                w=image_size[0], h=image_size[1])
                    # now feed in the images
                    _, loss_, acc_ = sess.run([train_op, reduce_loss, accuracy], 
                                feed_dict={x:batch_images, labels:batch_labels, learning_rate: base_lr})

                    text = '\rEpoch [{}/{}] | [{}/{}] TRAINING loss : {:.4f} TRAINING accuracy : {:.4f}'.format(e, args.epochs-1, idx, epoch_size, loss_, acc_)
                    print(text, end='')

                    if idx % save_interval == 0:
                        batch_images, batch_labels = util.get_batch(images_in, train_ids, 
                                                batch_size=args.batch_size, label_split=cifar_spliter, 
                                                classes=cifar_classes, standardize=True, 
                                                w=image_size[0], h=image_size[1])
                        # now feed in the images
                        _, loss_, acc_, rmse_ = sess.run([train_op, reduce_loss, accuracy, rmse], 
                                feed_dict={x:batch_images, labels:batch_labels, learning_rate: base_lr})
                        
                        saver = tf.train.Saver() # save
                        # save our model state, override previous save as well
                        # first remove old val
                        old_val = glob(args.save_dir+'val/val*') # get all val files
                        for path in old_val: # go through and remove them
                            os.remove(path)
                        saver.save(sess, args.save_dir + 'val/val_{}_{}.ckpt'.format(e, idx),global_step=idx) 

                        text = 'Epoch [{}/{}] | [{}/{}] VAL loss : {:.4f} VAL accuracy : {:.4f} VAL RMSE : {:.3f}'.format(e,args.epochs-1, idx, epoch_size, loss_, acc_, rmse_)
                        out_file.write('{}\n'.format(text))
                        cprint('\r{}'.format(text), 'yellow', attrs=['bold'])    
            # end of epoch
            elapsed_time = time.time() - start_time
            time_out = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
            out_file.write('elapsed_time: {}\t raw: {}\n'.format(time_out, elapsed_time))
            out_file.close()
            saver = tf.train.Saver()
            saver.save(sess, args.save_dir + 'epoch_{}_model.ckpt'.format(e),global_step=epoch_size + (e*epoch_size)) # save our model state
        # end
        print('\nDone!')
        

