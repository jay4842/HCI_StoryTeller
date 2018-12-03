from tkinter import *

class MyFirstGUI:
    def __init__(self, master):
        self.master = master
        master.title("HCI Project")
        #e = Entry(master)

        self.label = Label(master, text="Menu")
        self.label.grid(row=0)

        self.greet_button = Button(master, text="Run Example", command=self.Run_Example)
        self.greet_button.grid(row=1)

        self.greet_button = Button(master, text="Author", command=self.Author)
        self.greet_button.grid(row=2)

        self.greet_button = Button(master, text="Image Input", command=self.Img_Input)
        self.greet_button.grid(row=3)

        self.close_button = Button(master, text="Quit", command=master.quit)
        self.close_button.grid(row=4)

    #Handler function to greet
    def Run_Example(self):
        print("Run Example Selected")

    def Author(self):
        print("Author Selected")
        e.pack()

    def Img_Input(self):
        print("Image Input Selected")

root = Tk()
my_gui = MyFirstGUI(root)
root.mainloop()