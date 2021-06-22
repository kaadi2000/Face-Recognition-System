from tkinter import *
from tkinter import filedialog
from tkinter.font import ITALIC
import cv2, numpy, os
from PIL import Image
from functools import partial

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("assets/haarcascade_frontalface_default.xml")

def selectfolder(id):
    temp = Tk()
    temp.withdraw()
    path = filedialog.askdirectory()
    def getImagesAndLabels(path):
        imagePaths = os.listdir(path)  
        faceSamples=[]
        ids = []

        for imagePath in imagePaths:
            temp = path + "/" + imagePath
            PIL_img = Image.open(temp).convert('L')
            img_numpy = numpy.array(PIL_img,'uint8')

            faces = detector.detectMultiScale(img_numpy)

            for (x,y,w,h) in faces:
                faceSamples.append(img_numpy[y:y+h,x:x+w])
                ids.append(id)

        return faceSamples,ids

    
    faces,ids = getImagesAndLabels(path)
    recognizer.train(faces, numpy.array(ids))
    recognizer.write('trainer/trained.yml')


def camera(id):
    print(id)

def train():
    train = Tk()
    train.title("Train Dataset")

    id = StringVar()

    def folder():
        selectfolder(id.get())
    
    def manual():
        camera(int(str(id.get())))
    
    Label(train, text = "Enter ID", font = ("", 14)).grid(row =0, column = 0)
    Entry(train, textvariable = id, font = ("", 11, ITALIC)).grid(row = 0, column = 1)
    Button(train, text = "Train from Folder", command = folder, font = ("", 14)).grid(row = 1, column = 0)
    Button(train, text = "Train from Camera", command = manual, font = ("", 14)).grid(row = 1, column = 1)

    train.resizable(False, False)
    train.mainloop()
