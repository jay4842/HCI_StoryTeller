from tkinter import filedialog
from tkinter import *

class WindowGUI:
    def __init__(self, master):
        # tkinter stuff
        self.master = master
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
        self.close_button = Button(master, text="Quit", command=master.quit)
        self.close_button.grid(row=2, column=2)

        '''#Test output Button
        self.dis_button = Button(master, text="Sample Display", command=self.displayText)
        self.dis_button.grid(row=3, column=1)'''

    #Handler function to greet
    def Run_Example(self):
        #print("Run Example Selected")

    def Author(self):
        print(self.authorStg.get())
        return self.authorStg

    def Img_Input(self):
        #print("Image Input Selected")
        self.filename = filedialog.askopenfilename(initialdir = "./",title = "Select file",filetypes = (("jpeg files","*.jpg"),("png files","*.png"),("all files","*.*")))
        print(self.filename)
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
        self.story = """HAMLET: To be, or not to be--that is the question:
        Whether 'tis nobler in the mind to suffer
        The slings and arrows of outrageous fortune
        Or to take arms against a sea of troubles
        And by opposing end them. To die, to sleep--
        No more--and by a sleep to say we end
        The heartache, and the thousand natural shocks
        That flesh is heir to. 'Tis a consummation
        Devoutly to be wished."""
        self.T.insert(END, self.story)

if __name__ == "__main__":
    root = Tk()
    my_gui = WindowGUI(root)
    root.mainloop()