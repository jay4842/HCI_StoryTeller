import tensorflow as tf
import numpy as np
from glob import glob
import os
import time
import signal 
import nltk
import cv2
from tkinter import *
from src.gui.window import Window
from src.net.vgg import vgg16
from data.imagenet.imagenet_classes import class_names
import src.text.rnn_test as rnn # text generator

# vgg16 runner helper
def classify(image_path='data/imagenet/laska.png'):
    sess = tf.Session()
    imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
    vgg = vgg16(imgs, 'model_saves/vgg16/vgg16_weights.npz', sess)
    os.system('clear')
    img1 = cv2.imread(image_path)
    img1 = cv2.resize(img1, (224, 224))

    prob = sess.run(vgg.probs, feed_dict={vgg.imgs: [img1]})[0]
    preds = (np.argsort(prob)[::-1])[0:5] # print top 5
    for p in preds:
        print('{} {}'.format(class_names[p], prob[p]))

# more later
# This guy will combine the classifier and the rnn
# - Flow
#   - Get image, and give it to the classifier to get a class output
#   - Load author we are using and generate a text output
#   - Give both outputs to the post processing function to get final output
#   - After that is done read story aloud using something
#   - Maybe give the rnn input with voice
def deploy(args):
    root = Tk()
    os.system('clear')
    print('working...')
    app = Window(root)
    root.mainloop()
    #nltk.download('all') # first need to try this guy out

     # Now move on to the actual bulk stuff
    # label stuff here
    os.system('clear')
    test_images_in = glob(args.data_path + 'test/*.png')
    np.random.shuffle(test_images_in)
    # run test_single
    #class_label = cnn.run_single(args, test_image_path)
    #classify()
    #input('->')
    # now generate the rnn output
    story_output = rnn.test_rnn(args, display=False)
    print(story_output)

    # Get out named entities if any found
    # - tagged will also get our positions
    tokens = nltk.word_tokenize(story_output)
    #print('\n{}'.format(tokens)) # print tokens, turn the whole string into an array of words
    tagged = nltk.pos_tag(tokens)
    print('\n{}\n'.format(tagged)) # print the type of word in the token array
    # now here we go through tagged and locate the most occuring NNP tag
    for tag in tagged:
        if(tag[1] == 'NNP'):
            print(tag)
    entities = nltk.chunk.ne_chunk(tagged)

    # here we will replace the main entity found with our predicted class
    #print('\n{}'.format(entities))

    # Rules for post processing
    # - First determine the most occuring NNP tag for each token
    #   - If there are only two or three that are close to eachother group them together, they 
    #     are most likely part of the same entity/name.
    # - Then replace each instance with the classification of the image from the vgg network.
    # - Also we can randomly assign a name to the entity too.
    #   Ex: Alex the ferret

