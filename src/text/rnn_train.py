import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib import rnn
import numpy as np
import os
import time
import math
import my_txtutils as txt
tf.set_random_seed(15)

# TODO: Add restoring the model

# This wil be the training flow of the rnn
# - inspired from the shakespeer rnn
def train_rnn(args):
    SEQLEN = 30
    BATCHSIZE = 200
    ALPHASIZE = txt.ALPHASIZE
    INTERNALSIZE = 512
    NLAYERS = 3
    learning_rate = 0.0002 # small learning rate
    dropout_keep = .9 # only some dropout they use .8 but .9 is my preference

    text_files = args.data_path + '*.txt' # get all of the text files from data_path
    codetext, valitext, bookranges = txt.read_data_files(text_files, validation=True)
    # set epoch size based on batchsize and sequence len
    epoch_size = len(codetext) // (BATCHSIZE * SEQLEN)
    # model placeholders
    lr = tf.placeholder(tf.float32, name='learning_rate')
    p_keep = tf.placeholder(tf.float32, name='p_keep')
    batch_size = tf.placeholder(tf.int32, name='batch_size')
    # input placeholders
    X   = tf.placeholder(tf.uint8, [None, None], name='X') # 
    Xo  = tf.one_hot(X, ALPHASIZE, 1.0, 0.0)
    # expected outputs = same sequence shifted by 1 
    Y_  = tf.placeholder(tf.uint8, [None, None], name='Y_')
    Yo_ = tf.one_hot(Y, ALPHASIZE, 1.0, 0.0)
    # input state
    Hin = tf.placeholder(tf.float32, [None, INTERNALSIZE*NLAYERS], name='Hin')
    # Using a NLAYERS=3 of cells, unrolled SEQLEN=30 times
    # dynamic_rnn infers SEQLEN from the size of the inputs Xo
    cells = [rnn.GRUCell(INTERNALSIZE) for _ in range(NLAYERS)]
    # weird dropout, well simple dropout
    drop_cells = [rnn.DropoutWrapper(cell, input_keep_prob=p_keep) for cell in cells]
    multi_cell = rnn.MultiRNNCell(drop_cells, state_is_tuple=False)
    multi_cell = rnn.DropoutWrapper(multi_cell, output_keep_prob=p_keep)
    # ^ The last layer is for the softmax dropout
    Yr, H = tf.nn.dynamic_rnn(multi_cell, Xo, dtype=tf.float32, initial_state=Hin)
    # H is that last state
    H = tf.identity(H, name='H') # give it a tf name
    # Softmax layer implementation:
    # Flatten the first two dimension of the output [ BATCHSIZE, SEQLEN, ALPHASIZE ] => [ BATCHSIZE x SEQLEN, ALPHASIZE ]
    # then apply softmax readout layer. This way, the weights and biases are shared across unrolled time steps.
    # From the readout point of view, a value coming from a sequence time step or a minibatch item is the same thing.
    Yflat = tf.reshape(Yr, [-1, INTERNALSIZE])
    Ylogits = layers.linear(Yflat, ALPHASIZE)
    Yflat_ = tf.reshape(Yo_, [-1, ALPHASIZE]) 
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Yflat_)
    loss = tf.reshape(loss, [batchsize, -1])
    Yo = tf.nn.softmax(Ylogits, name='Yo')
    Y = tf.argmax(Yo, 1) 
    Y = tf.reshape(Y, [batchsize, -1], name="Y")
    train_step = tf.train.AdamOptimizer(lr).minimize(loss)
    # stats for display
    seqloss = tf.reduce_mean(loss, 1)
    batchloss = tf.reduce_mean(seqloss)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(Y_, tf.cast(Y, tf.uint8)), tf.float32))
    loss_summary = tf.summary.scalar("batch_loss", batchloss)
    acc_summary = tf.summary.scalar("batch_accuracy", accuracy)
    summaries = tf.summary.merge([loss_summary, acc_summary])
    # For saving models
    os.makedirs('checkpoints/', exist_ok=True)
    # Only the last checkpoint will be saved
    saver = tf.train.Saver(max_to_keep=1000)
    # For displaying progress
    # - changing this to my own implementation
    # - Theyres is too much output
    istate = np.zeros([BATCHSIZE, INTERNALSIZE*NLAYERS])
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    step = 0
    # Training loop
    
