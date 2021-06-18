from tkinter import *
from tkinter.font import BOLD
from static import facedetection, traindata

window = Tk()
window.title("Face Recognition")

def train():
	pass

def detect():
	pass


Button(text = "TRAIN DATA", font = ("", 40, BOLD), command = train).pack()
Button(text = "RECOGNIZE", font = ("", 40, BOLD), command = detect).pack()

window.resizable(False, False)
window.mainloop()