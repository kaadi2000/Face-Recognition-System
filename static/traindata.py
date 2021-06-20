from tkinter import *
from tkinter import filedialog
import cv2
import numpy as np
from PIL import Image
import os

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("assets/haarcascade_frontalface_default.xml")

def selectfolder():
    window = Tk()
    window.withdraw()
    path = filedialog.askdirectory()
    def getImagesAndLabels(path):
        imagePaths = os.listdir(path)  
        faceSamples=[]
        ids = []

        for imagePath in imagePaths:
            temp = path + "/" + imagePath
            PIL_img = Image.open(temp).convert('L')
            img_numpy = np.array(PIL_img,'uint8')

            id = 2
            faces = detector.detectMultiScale(img_numpy)

            for (x,y,w,h) in faces:
                faceSamples.append(img_numpy[y:y+h,x:x+w])
                ids.append(id)

        return faceSamples,ids

    
    faces,ids = getImagesAndLabels(path)
    recognizer.train(faces, np.array(ids))
    recognizer.write('trainer/trained.yml')


def camera():
    pass

def train():
    pass