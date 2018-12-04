from tkinter import filedialog
from tkinter import *

import tensorflow as tf
import numpy as np
import os
from random import randint
import cv2
import nltk
import string
from threading import Thread

import src.util as util
from src.net.vgg import vgg16
from data.imagenet.imagenet_classes import class_names
import src.text.rnn_test as rnn_test
import src.gui.speech_recog as speech

def random_name(name_list):
    max_ = len(name_list)
    return name_list[randint(0,max_)]

# vgg16 runner helper
def classify(vgg, sess, image_path='data/imagenet/laska.png'):
    os.system('clear')
    img1 = cv2.imread(image_path)
    img1 = cv2.resize(img1, (224, 224))

    prob = sess.run(vgg.probs, feed_dict={vgg.imgs: [img1]})[0]
    preds = (np.argsort(prob)[::-1])[0:5] # print top 5
    #for p in preds:
    #    print('{} {}'.format(class_names[p], prob[p]))
    
    top = preds[0]
    output = class_names[top].split(',')[0]
    return output
    
def post(string_input, name_list, class_type):
    # Rules for post processing
    # - First determine the most occuring NNP tag for each token
    #   - If there are only two or three that are close to eachother group them together, they 
    #     are most likely part of the same entity/name.
    # - Then replace each instance with the classification of the image from the vgg network.
    # - Also we can randomly assign a name to the entity too.
    #   Ex: Alex the ferret
     # Get out named entities if any found
    # - tagged will also get our positions
    print(string_input)
    tokens = nltk.word_tokenize(string_input)
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
    if(len(characters) <= 0):
        return None
    # check to see if characters are right next to each other
    if(len(characters) >= 2):
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
    # change the tokens
    # - also make sure that when we change the first name its set to <name> the <class name>
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
    rebuild = rebuild.replace('``', '\"').replace('\'\'', '\"')
    return rebuild

# open a cv2 window and take pictures using the spacebar
# - once done press ESC to exit and return the last captured frame
def capture_picture():
    # open the camera
    os.makedirs('data/tmp/',exist_ok=True)
    cam = cv2.VideoCapture(0)
    cv2.namedWindow('Capture')
    img_counter = 0
    last_frame_name = ''
    while True:
        ret, frame = cam.read()
        frame = np.fliplr(frame)
        cv2.imshow('Capture', frame)
        if not ret:
            break;
        k = cv2.waitKey(1)
        if(k%256 == 27):
            # esc pressed
            print('closing camera')
            break;
        elif(k%256 == 32):
            # space pressed
            img_name = 'data/tmp/frame_{}.png'.format(img_counter)
            last_frame_name = img_name
            cv2.imwrite(img_name, frame)
            print('wrote {}!'.format(img_name))
            img_counter+=1
    cam.release()
    cv2.destroyWindow('Capture')
    return last_frame_name

# This guy will just take a few pics in the background without 
# showing the window.
def speech_capture_image():
    # open the camera
    os.makedirs('data/tmp/',exist_ok=True)
    cam = cv2.VideoCapture(0)
    img_counter = 0
    last_frame_name = ''
    for i in range(0,5):
        ret, frame = cam.read()
        frame = np.fliplr(frame)
        if not ret:
            break;
        # space pressed
        img_name = 'data/tmp/frame_{}.png'.format(img_counter)
        last_frame_name = img_name
        cv2.imwrite(img_name, frame)
        print('wrote {}!'.format(img_name))
        img_counter+=1
    # release cam
    cam.release()
    return last_frame_name

# Window class
# - Holds GUI
# - executes deploy flow
class WindowGUI:
    def __init__(self, master, args):
        # tkinter stuff
        self.master = master
        self.running = True
        self.capturing = False
        master.title("HCI Project")
        self.authorStg = StringVar(master, value='doyle')

        #Title
        #self.label = Label(master, text="Menu").grid(row=0, column=1)

        #Author Button
        self.label = Label(master, text="Author").grid(row=1, column=0)
        self.auEnt = Entry(master, textvariable=self.authorStg).grid(row=1, column=1)
        self.getAu = Button(master, text="Get Author", comman=self.Author).grid(row=1, column=2)
        
        #Run Example Button
        self.greet_button = Button(master, text="Run Example", command=self.Run_Example)
        self.greet_button.grid(row=2, column=0)

        #Image input Button
        self.greet_button = Button(master, text="Image Input", command=self.Img_Input)
        self.greet_button.grid(row=2, column=1)

        #Quit Button
        self.close_button = Button(master, text="Quit", command=self.quit_)
        self.close_button.grid(row=2, column=2)
        
        # capture button
        self.greet_button = Button(master, text="Captuer Image", command=self.capture_image)
        self.greet_button.grid(row=3, column=1)

        '''#Test output Button
        self.dis_button = Button(master, text="Sample Display", command=self.displayText)
        self.dis_button.grid(row=3, column=1)'''
        # Tensorflow stuff
        self.args = args
        self.vgg_sess = tf.Session()
        self.rnn_sess = tf.Session()
        with tf.variable_scope('vgg'):
            self.imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
            self.vgg = vgg16(self.imgs, 'model_saves/vgg16/vgg16_weights.npz', self.vgg_sess)
        
        self.rnn = rnn_test.Tester(self.args, self.rnn_sess)
        self.name_list = util.make_names_list()
        self.story_output = ''
        self.filename = 'data/imagenet/laska.png'
        self.author = 'doyle'
        #self.rnn.set_author(self.author, self.rnn_sess)
        # speech stuff
        self.speech_handler = speech.Speech_Recogn()

        self.thread = Thread(target=self.capture_speech_image, args=())
        self.thread.start() # start listing
    # 
    def quit_(self):
        self.running = False
        self.thread.join()
        self.master.quit()

    def deploy(self):
        os.system('clear')
        print('working...')
        # Now move on to the actual bulk stuff
        # label stuff here
        class_type = classify(self.vgg, self.vgg_sess, image_path=self.filename)
        print(class_type)
        #input('->')
        # now generate the rnn output
        post_none = False
        story_output = ''
        for idx in range(0,5):
            story_output = self.rnn.run(self.args, self.rnn_sess, display=False)
            story_output = post(story_output, self.name_list, class_type)
            if(story_output is not None):
                return story_output
        return story_output

    #Handler function to greet
    def Run_Example(self):
        print("Run Example Selected")
        self.story_output = self.deploy()
        self.displayText()
    # TODO: add changing the author
    def Author(self):
        self.author = self.authorStg.get()
        self.rnn.set_author(self.author, self.rnn_sess)
        return self.authorStg

    def Img_Input(self):
        #print("Image Input Selected")
        if not(self.capturing):
            self.capturing = True
            self.filename = filedialog.askopenfilename(initialdir = "./",title = "Select file",filetypes = (("jpeg files","*.jpg"),("png files","*.png"),("all files","*.*")))
            print(self.filename)
            self.capturing = False
    #Display the story in its own window
    def displayText(self):
        self.disTop = Toplevel()    #create a new window
        self.disTop.title("Story")
        self.S = Scrollbar(self.disTop) #add a scroller to scroll through text
        self.T = Text(self.disTop, height=30, width=100)  #how many lines and characters per line to show in screen
        self.S.grid(row=0, column=1)
        self.T.grid(row=0)
        self.S.config(command=self.T.yview)
        self.T.config(yscrollcommand=self.S.set)
        self.T.insert(END, self.story_output)

    # capture button
    def capture_image(self):
        if not(self.capturing):
            self.capturing = True
            self.filename = capture_picture()
            self.capturing = False

    # listen to voice. If something is heard captuer it
    def capture_speech_image(self):
        while self.running:
            word_found = self.speech_handler.speech_runner()
            if(word_found is not None and word_found == 'capture'):
                if not(self.capturing):
                    self.capturing = True
                    self.filename = speech_capture_image()
                    self.capturing = False

if __name__ == "__main__":
    root = Tk()
    my_gui = WindowGUI(root)
    root.mainloop()