from tkinter import *
from tkinter.font import BOLD
from static import facedetection, traindata
from functools import partial

window = Tk()
window.title("Face Recognition")


Button(text = "TRAIN DATA", font = ("", 40, BOLD), command = partial(traindata.train)).pack()
Button(text = "RECOGNIZE", font = ("", 40, BOLD), command = partial(facedetection.detect)).pack()

window.resizable(False, False)
window.mainloop()