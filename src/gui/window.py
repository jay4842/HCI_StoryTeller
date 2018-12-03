from tkinter import *

class WindowGUI:
    def __init__(self, master):
        # tkinter stuff
        self.master = master
        master.title("HCI Project")

        #Title
        self.label = Label(master, text="Menu")
        self.label.grid(row=0, column=1)
        
        #Run Example Button
        self.greet_button = Button(master, text="Run Example", command=self.Run_Example)
        self.greet_button.grid(row=1, column=0)

        #Author Button
        self.greet_button = Button(master, text="Author", command=self.Author)
        self.greet_button.grid(row=1, column=1)

        #Image input Button
        self.greet_button = Button(master, text="Image Input", command=self.Img_Input)
        self.greet_button.grid(row=1, column=2)

        #Quit Button
        self.close_button = Button(master, text="Quit", command=master.quit)
        self.close_button.grid(row=2, column=1)
        # tensorflow stuff
        
    #Handler function to greet
    def Run_Example(self):
        print("Run Example Selected")

    def Author(self):
        print("Author Selected")
        self.eTop = Toplevel()    #object to create a new window
        self.eTop.title("Author")
        self.e = Entry(self.eTop).grid(row=0, column=1)
        self.get = Button(self.eTop, text = "Get Text", command=self.getText)
        self.quit = Button(self.eTop, text = "Quit", command=self.eTop.destroy)

    def Img_Input(self):
        print("Image Input Selected")

    def getText(self):
        self.e.get()

if __name__ == "__main__":
    root = Tk()
    my_gui = WindowGUI(root)
    root.mainloop()