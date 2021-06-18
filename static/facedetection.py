import numpy as np
import cv2

def detect():
    faceCascade = cv2.CascadeClassifier('assets/haarcascade_frontalface_default.xml')

    capture = cv2.VideoCapture(0)
    capture.set(3,640) # set Width
    capture.set(4,480) # set Height

    while True:
        ret, img = capture.read()
        faces = faceCascade.detectMultiScale(
            img,
            scaleFactor=1.3,
            minNeighbors=5,     
            minSize=(20, 20)
        )

        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            roi_color = img[y:y+h, x:x+w]
            

        cv2.imshow('video',img)

        k = cv2.waitKey(30) & 0xff
        if k == 27: # press 'ESC' to quit
            break

    capture.release()
    cv2.destroyAllWindows()