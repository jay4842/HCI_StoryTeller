from tkinter import *

from src.gui.window import WindowGUI# text generator
# Okay so to have two models with different session
# - we will use two graphs and two session

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
    my_gui = WindowGUI(root, args)
    root.mainloop()
