import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib import rnn
import numpy as np
import os
import time
import math
import src.text.my_txtutils as txt
tf.set_random_seed(15)

# TODO: Add restoring the model
# TODO: Add displaying progress

# This wil be the training flow of the rnn
# - inspired from the shakespeer rnn
def train_rnn(args):
    SEQLEN = 30
    BATCHSIZE = args.batch_size
    ALPHASIZE = txt.ALPHASIZE
    INTERNALSIZE = 512
    NLAYERS = 5
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
    Yo_ = tf.one_hot(Y_, ALPHASIZE, 1.0, 0.0)
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
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=Ylogits, labels=Yflat_)
    loss = tf.reshape(loss, [batch_size, -1])
    Yo = tf.nn.softmax(Ylogits, name='Yo')
    Y = tf.argmax(Yo, 1) 
    Y = tf.reshape(Y, [batch_size, -1], name="Y")
    train_step = tf.train.AdamOptimizer(lr).minimize(loss)
    # stats for display
    seqloss = tf.reduce_mean(loss, 1)
    batchloss = tf.reduce_mean(seqloss)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(Y_, tf.cast(Y, tf.uint8)), tf.float32))
    loss_summary = tf.summary.scalar("batch_loss", batchloss)
    acc_summary = tf.summary.scalar("batch_accuracy", accuracy)
    summaries = tf.summary.merge([loss_summary, acc_summary])
    # Init Tensorboard stuff. This will save Tensorboard information into a different
    # folder at each run named 'log/<timestamp>/'. Two sets of data are saved so that
    # you can compare training and validation curves visually in Tensorboard.
    timestamp = str(math.trunc(time.time()))
    #summary_writer = tf.summary.FileWriter("log/" + timestamp + "-training")
    #validation_writer = tf.summary.FileWriter("log/" + timestamp + "-validation")
    # For saving models
    os.makedirs(args.save_dir, exist_ok=True)
    # Only the last checkpoint will be saved
    saver = tf.train.Saver(max_to_keep=1000)
    gen_file = open(args.save_dir + 'generated.txt', 'w')
    # For displaying progress
    # - changing this to my own implementation
    # - Theyres is too much output
    # for display: init the progress bar
    # TODO: Change this guy eventually
    DISPLAY_FREQ = 50
    _50_BATCHES = DISPLAY_FREQ * BATCHSIZE * SEQLEN
    progress = txt.Progress(DISPLAY_FREQ, size=111+2, msg="Training on next "+str(DISPLAY_FREQ)+" batches")
    #
    istate = np.zeros([BATCHSIZE, INTERNALSIZE*NLAYERS])
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    step = 0
    # Training loop
    for x, y_, epoch in txt.rnn_minibatch_sequencer(codetext, BATCHSIZE, SEQLEN, nb_epochs=args.epochs):
        # train on one minibatch
        feed_dict = {X: x, Y_: y_, Hin: istate, lr: learning_rate, p_keep: dropout_keep, batch_size: BATCHSIZE}
        _, y, ostate = sess.run([train_step, Y, H], feed_dict=feed_dict)
        # validation step
        if step % _50_BATCHES == 0:
            feed_dict = {X: x, Y_: y_, Hin: istate, p_keep: 1.0, batch_size: BATCHSIZE}  # no dropout for validation
            y, l, bl, acc, smm = sess.run([Y, seqloss, batchloss, accuracy, summaries], feed_dict=feed_dict)
            txt.print_learning_learned_comparison(x, y, l, bookranges, bl, acc, epoch_size, step, epoch)
            #summary_writer.add_summary(smm, step)
        
        # run a validation step every 50 batches
        # The validation text should be a single sequence but that's too slow (1s per 1024 chars!),
        # so we cut it up and batch the pieces (slightly inaccurate)
        # tested: validating with 5K sequences instead of 1K is only slightly more accurate, but a lot slower.
        if step % _50_BATCHES == 0 and len(valitext) > 0:
            VALI_SEQLEN = 1*1024  # Sequence length for validation. State will be wrong at the start of each sequence.
            bsize = len(valitext) // VALI_SEQLEN
            txt.print_validation_header(len(codetext), bookranges)
            vali_x, vali_y, _ = next(txt.rnn_minibatch_sequencer(valitext, bsize, VALI_SEQLEN, 1))  # all data in 1 batch
            vali_nullstate = np.zeros([bsize, INTERNALSIZE*NLAYERS])
            feed_dict = {X: vali_x, Y_: vali_y, Hin: vali_nullstate, p_keep: 1.0,  # no dropout for validation
                        batch_size: bsize}
            ls, acc, smm = sess.run([batchloss, accuracy, summaries], feed_dict=feed_dict)
            txt.print_validation_stats(ls, acc)
            # save validation data for Tensorboard
            #validation_writer.add_summary(smm, step)
            saver.save(sess, '{}/rnn_val_save'.format(args.save_dir+'val/'), global_step=step)
        # display a short text generated with the current weights and biases (every 150 batches)
        if step // 3 % _50_BATCHES == 0:
            txt.print_text_generation_header()
            ry = np.array([[txt.convert_from_alphabet(ord("K"))]])
            rh = np.zeros([1, INTERNALSIZE * NLAYERS])
            gen_file.write('----------------- STEP {} -----------------\n'.format(step))
            for k in range(1000):
                ryo, rh = sess.run([Yo, H], feed_dict={X: ry, p_keep: 1.0, Hin: rh, batch_size: 1})
                rc = txt.sample_from_probabilities(ryo, topn=10 if epoch <= 1 else 2)
                letter = (chr(txt.convert_to_alphabet(rc)))
                gen_file.write(letter)
                print(letter, end='')
                ry = np.array([[rc]])
            txt.print_text_generation_footer()
            gen_file.write('\n')
        # display progress bar
        progress.step(reset=step % _50_BATCHES == 0)
        # loop state around
        istate = ostate
        step += BATCHSIZE * SEQLEN
    
    gen_file.close()
    saved_file = saver.save(sess, '{}rnn_train_'.format(args.save_dir) + timestamp, global_step=step)
    print("Saved file: " + saved_file)