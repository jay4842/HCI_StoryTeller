import tensorflow as tf
import numpy as np
from glob import glob
import os
import time
import signal 
import nltk
import cv2
from tkinter import *
from random import randint
import string

import src.util as util
from src.gui.window import Window
from src.net.vgg import vgg16
from data.imagenet.imagenet_classes import class_names
import src.text.rnn_test as rnn # text generator

def random_name(name_list):
    max_ = len(name_list)
    return name_list[randint(0,max_)]

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
    #for p in preds:
    #    print('{} {}'.format(class_names[p], prob[p]))
    
    top = preds[0]
    return class_names[top]
    

# more later
# This guy will combine the classifier and the rnn
# - Flow
#   - Get image, and give it to the classifier to get a class output
#   - Load author we are using and generate a text output
#   - Give both outputs to the post processing function to get final output
#   - After that is done read story aloud using something
#   - Maybe give the rnn input with voice
def deploy(args):
    #root = Tk()
    name_list = util.make_names_list()
    os.system('clear')
    print('working...')
    #app = Window(root)
    #root.mainloop()
    #nltk.download('all') # first need to try this guy out

     # Now move on to the actual bulk stuff
    # label stuff here
    os.system('clear')
    test_images_in = glob(args.data_path + 'test/*.png')
    np.random.shuffle(test_images_in)
    # run test_single
    #class_label = cnn.run_single(args, test_image_path)
    #class_type = classify()
    class_type = 'mink'
    print(class_type)
    #input('->')
    # now generate the rnn output
    story_output = rnn.test_rnn(args, display=False)
    #story_output = story_output.lower()
    #print(story_output)

    # Get out named entities if any found
    # - tagged will also get our positions
    tokens = nltk.word_tokenize(story_output)
    #print('\n{}'.format(tokens)) # print tokens, turn the whole string into an array of words
    tagged = nltk.pos_tag(tokens)
    #print('\n{}\n'.format(tagged)) # print the type of word in the token array
    # now here we go through tagged and locate the most occuring NNP tag
    characters = []
    for tag in range(len(tagged)):
        if(tagged[tag][1] == 'NNP'):
            #print(tag)
            #print(tagged[tag])
            characters.append([tagged[tag][0], tag])
    #entities = nltk.chunk.ne_chunk(tagged)
    #print(characters)
    # check to see if characters are right next to each other
    for idx in range(len(characters)-2):
        if(characters[idx][1] == characters[idx+1][1]-1):
            characters.remove(characters[idx])
            idx-=1
    changed_names = []
    for idx in range(len(characters)):
        if(len(changed_names) < 0):
            changed_names.append( [characters[idx][0], random_name(name_list)])
        else:
            # check if the name has already been changed
            fail = False
            for name_idx in range(len(changed_names)):
                if(characters[idx][0] == changed_names[name_idx][0]):
                    fail = True
                    break;
            if not(fail):
                changed_names.append( [characters[idx][0], random_name(name_list)] )
    #print(changed_names)
    for idx in range(len(changed_names)):
        count = 0
        for idx_j in range(len(characters)):
            if(changed_names[idx][0] == characters[idx_j][0]):
                if(count == 0):
                    tokens[characters[idx_j][1]] = '{} the {}'.format(changed_names[idx][1], class_type)
                    count += 1
                else:
                    tokens[characters[idx_j][1]] = changed_names[idx][1]
                    count += 1
    rebuild = "".join([" "+i if not i.startswith("'") and i not in string.punctuation else i for i in tokens]).strip()
    print('\n{}'.format(rebuild))
    # here we will replace the main entity found with our predicted class
    #print('\n{}'.format(entities))

    # Rules for post processing
    # - First determine the most occuring NNP tag for each token
    #   - If there are only two or three that are close to eachother group them together, they 
    #     are most likely part of the same entity/name.
    # - Then replace each instance with the classification of the image from the vgg network.
    # - Also we can randomly assign a name to the entity too.
    #   Ex: Alex the ferret

