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
def test_rnn(args):
    SEQLEN = 30
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

    print(meta_file)

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
            print(c, end="")
            test_file.write(c)
            if c == '\n':
                ncnt = 0
            else:
                ncnt += 1
            if ncnt == 100:
                print("")
                test_file.write('\n')
                ncnt = 0
        test_file.close()
    print('\n')