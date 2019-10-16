# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 14:13:46 2019

@author: inkuml05
"""

import cv2 
import numpy as np

face_classifier = cv2.CascadeClassifier('C:/Users/inkuml05/Downloads/opencv-master/opencv-master/data/haarcascades/haarcascade_frontalface_default.xml')


##creating a method to extract the face features and return the data 
def face_extractor(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face = face_classifier.detectMultiScale(gray, 1.3, 5)
    if face is():
        return None
    for(x,y,w,h) in face:
        cropped_faces = img[y:y+h, x:x+w]
    return cropped_faces


cap = cv2.VideoCapture(0)
count = 0

## Run the camera and capture.
while True:
    ret, frame = cap.read()
    if face_extractor(frame) is not None:
        count+=1
       
        face = cv2.resize(face_extractor(frame),(200,200))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        
        file_name_path = 'D:/CV2_Project_Data/'+ str(count)+'.jpg'
        cv2.imwrite(file_name_path, face)
        
        cv2.putText(face, str(count), (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
        cv2.imshow('Face Cropper', face)
    else:
        print('Face not found, Please show your face in the camera')
        pass
    ## 13 is the ASCII value for Enter key, if the enter key is pressed by user stop the process
    if cv2.waitKey(1) == 13:
        break

cap.release()
cv2.destroyAllWindows()
print('Collecting samples complete, Realeasing the camera port')
print('Camera port released successfully....')