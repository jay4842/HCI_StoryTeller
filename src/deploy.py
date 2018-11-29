import tensorflow as tf
import numpy as np
from glob import glob
import os
import time
import signal 
import nltk

import src.classify.test as cnn # classifier
import src.text.rnn_test as rnn # text generator
import src.text.coreNLP as coreNLP # post process
# more later
# This guy will combine the classifier and the rnn
# - Flow
#   - Get image, and give it to the classifier to get a class output
#   - Load author we are using and generate a text output
#   - Give both outputs to the post processing function to get final output
#   - After that is done read story aloud using something
#   - Maybe give the rnn input with voice

def deploy(args):
    os.system('clear')
    print('working...')
    nltk.download('all') # first need to try this guy out

     # Now move on to the actual bulk stuff
    # label stuff here
    os.system('clear')
    test_images_in = glob(args.data_path + 'test/*.png')
    np.random.shuffle(test_images_in)
    cifar_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    cifar_spliter = '+'
    test_image_path = test_images_in[np.random.randint(0,high=len(test_images_in))]
    print(test_image_path)
    # run test_single
    class_label = cnn.run_single(args, test_image_path)

    # now generate the rnn output
    story_output = rnn.test_rnn(args, display=False)
    print(story_output)

    # Get out named entities if any found
    # - tagged will also get our positions
    tokens = nltk.word_tokenize(story_output)
    tagged = nltk.pos_tag(tokens)
    entities = nltk.chunk.ne_chunk(tagged)

    print('\n{}'.format(entities))
