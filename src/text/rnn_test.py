import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib import rnn
import numpy as np
import os
import time
import math
from glob import glob

import src.text.my_txtutils as txt
# deploy code
# - Load up the author you want to use
# - 
def test_rnn(args, display=True):
    tf.reset_default_graph()
    SEQLEN = 50
    ALPHASIZE = txt.ALPHASIZE
    INTERNALSIZE = 512
    NLAYERS = 5
    # model 
    pkeep = tf.placeholder(tf.float32, name='pkeep')
    batch_size = tf.placeholder(tf.int32, name='batchsize')
    X   = tf.placeholder(tf.uint8, [None, None], name='X') # 
    Xo  = tf.one_hot(X, ALPHASIZE, 1.0, 0.0)
    Hin = tf.placeholder(tf.float32, [None, INTERNALSIZE*NLAYERS], name='Hin')
    # Using a NLAYERS=3 of cells, unrolled SEQLEN=30 times
    # dynamic_rnn infers SEQLEN from the size of the inputs Xo
    cells = [rnn.GRUCell(INTERNALSIZE) for _ in range(NLAYERS)]
    # weird dropout, well simple dropout
    drop_cells = [rnn.DropoutWrapper(cell, input_keep_prob=pkeep) for cell in cells]
    multi_cell = rnn.MultiRNNCell(drop_cells, state_is_tuple=False)
    multi_cell = rnn.DropoutWrapper(multi_cell, output_keep_prob=pkeep)
    # ^ The last layer is for the softmax dropout
    Yr, H = tf.nn.dynamic_rnn(multi_cell, Xo, dtype=tf.float32, initial_state=Hin)
    # H is that last state
    H = tf.identity(H, name='H') # give it a tf name
    Yflat = tf.reshape(Yr, [-1, INTERNALSIZE])
    Ylogits = layers.linear(Yflat, ALPHASIZE)
    Yo = tf.nn.softmax(Ylogits, name='Yo')
    # 
    restore_dir = 'rnn_saves/'
    # more authors will be added later on
    restore = restore_dir + args.author + '/5_layers_100/*'
    files = glob(restore)
    files.sort()
    meta_file = ''  # get our meta file from all the other files
    for x in files:
        meta_file = x if '.meta' in x else ''

    if(display): print(meta_file)

    output = ""
    # now lets restore the graph
    ncnt = 0
    with tf.Session() as sess:
        #new_saver = tf.train.import_meta_graph(meta_file)
        check_point = tf.train.latest_checkpoint(restore[:len(restore)-1])
        new_saver = tf.train.Saver()
        new_saver.restore(sess, check_point)
        x = txt.convert_from_alphabet(ord("L"))
        x = np.array([[x]])  # shape [BATCHSIZE, SEQLEN] with BATCHSIZE=1 and SEQLEN=1

        # initial values
        os.makedirs(args.test_dir + 'rnn/', exist_ok=True)
        test_file = open(args.test_dir + 'rnn/{}.txt'.format(args.author), 'w')
        y = x
        h = np.zeros([1, INTERNALSIZE * NLAYERS], dtype=np.float32)  # [ BATCHSIZE, INTERNALSIZE * NLAYERS]
        for i in range(args.test_size):
            yo, h = sess.run(['Yo:0', 'H:0'], feed_dict={'X:0': y, 'pkeep:0': 1.0, 'Hin:0': h, 'batchsize:0': 1})

            # If sampling is be done from the topn most likely characters, the generated text
            # is more credible and more "english". If topn is not set, it defaults to the full
            # distribution (ALPHASIZE)

            # Recommended: topn = 10 for intermediate checkpoints, topn=2 or 3 for fully trained checkpoints

            c = txt.sample_from_probabilities(yo, topn=2)
            y = np.array([[c]])  # shape [BATCHSIZE, SEQLEN] with BATCHSIZE=1 and SEQLEN=1
            c = chr(txt.convert_to_alphabet(c))
            if(display): print(c, end="")
            output += c
            test_file.write(c)
            if c == '\n':
                ncnt = 0
            else:
                ncnt += 1
            if ncnt == 100:
                if(display): print("")
                test_file.write('\n')
                ncnt = 0
        test_file.close()
    if(display): print('\n')

    return output

# This guy needs to be a class, so that it's easoer to call
class Tester:
    def __init__(self, args, sess):
        self.SEQLEN = 50
        self.ALPHASIZE = txt.ALPHASIZE
        self.INTERNALSIZE = 512
        self.NLAYERS = 5
        # model 
        self.pkeep = tf.placeholder(tf.float32, name='pkeep')
        self.batch_size = tf.placeholder(tf.int32, name='batchsize')
        self.X   = tf.placeholder(tf.uint8, [None, None], name='X') # 
        self.Xo  = tf.one_hot(self.X, self.ALPHASIZE, 1.0, 0.0)
        self.Hin = tf.placeholder(tf.float32, [None, self.INTERNALSIZE*self.NLAYERS], name='Hin')
        # Using a NLAYERS=3 of cells, unrolled SEQLEN=30 times
        # dynamic_rnn infers SEQLEN from the size of the inputs Xo
        self.cells = [rnn.GRUCell(self.INTERNALSIZE) for _ in range(self.NLAYERS)]
        # weird dropout, well simple dropout
        self.drop_cells = [rnn.DropoutWrapper(cell, input_keep_prob=self.pkeep) for cell in self.cells]
        self.multi_cell = rnn.MultiRNNCell(self.drop_cells, state_is_tuple=False)
        self.multi_cell = rnn.DropoutWrapper(self.multi_cell, output_keep_prob=self.pkeep)
        # ^ The last layer is for the softmax dropout
        self.Yr, self.H = tf.nn.dynamic_rnn(self.multi_cell, self.Xo, dtype=tf.float32, initial_state=self.Hin)
        # H is that last state
        self.H = tf.identity(self.H, name='H') # give it a tf name
        self.Yflat = tf.reshape(self.Yr, [-1, self.INTERNALSIZE])
        self.Ylogits = layers.linear(self.Yflat, self.ALPHASIZE)
        self.Yo = tf.nn.softmax(self.Ylogits, name='Yo')
        self.restore_dir = 'rnn_saves/'
        # more authors will be added later on
        self.set_author(args, sess)
    
    # For setting the author/setting the model
    def set_author(self, args, sess):
        self.restore = self.restore_dir + args.author + '/5_layers_100/*'
        self.output = ""
        # now lets restore the graph
        tvars = tf.trainable_variables()
        tvars = [var for var in tvars if not 'vgg' in var.name]

        self.check_point = tf.train.latest_checkpoint(self.restore[:len(self.restore)-1])
        new_saver = tf.train.Saver(tvars)
        new_saver.restore(sess, self.check_point)

    def run(self, args, sess, display=False):
        self.output = ''
        ncnt = 0
        
        #new_saver = tf.train.import_meta_graph(meta_file)
        x = txt.convert_from_alphabet(ord("L"))
        x = np.array([[x]])  # shape [BATCHSIZE, SEQLEN] with BATCHSIZE=1 and SEQLEN=1

         # initial values
        os.makedirs(args.test_dir + 'rnn/', exist_ok=True)
        test_file = open(args.test_dir + 'rnn/{}.txt'.format(args.author), 'w')
        y = x
        h = np.zeros([1, self.INTERNALSIZE * self.NLAYERS], dtype=np.float32)  # [ BATCHSIZE, INTERNALSIZE * NLAYERS]
        for i in range(args.test_size):
            yo, h = sess.run(['Yo:0', 'H:0'], feed_dict={'X:0': y, 'pkeep:0': 1.0, 'Hin:0': h, 'batchsize:0': 1})

            # If sampling is be done from the topn most likely characters, the generated text
            # is more credible and more "english". If topn is not set, it defaults to the full
            # distribution (ALPHASIZE)

            # Recommended: topn = 10 for intermediate checkpoints, topn=2 or 3 for fully trained checkpoints

            c = txt.sample_from_probabilities(yo, topn=2)
            y = np.array([[c]])  # shape [BATCHSIZE, SEQLEN] with BATCHSIZE=1 and SEQLEN=1
            c = chr(txt.convert_to_alphabet(c))
            if(display): print(c, end="")
            self.output += c
            test_file.write(c)
            if c == '\n':
                ncnt = 0
            else:
                ncnt += 1
            if ncnt == 100:
                if(display): print("")
                test_file.write('\n')
                ncnt = 0
        test_file.close()
        if(display): print('\n')

        return self.output