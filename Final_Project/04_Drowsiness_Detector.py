import cv2 as cv
import numpy as np
from tensorflow.keras.models import load_model
from pygame import mixer
import os

mixer.init()
sound=mixer.Sound('Siren.wav')
mixer.set_num_channels(1)

faceCascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
lefteyeCascade = cv.CascadeClassifier("haarcascade_lefteye_2splits.xml")
righteyeCascade = cv.CascadeClassifier("haarcascade_righteye_2splits.xml")
frameWidth = 640
frameHeight = 480
box_color = (0,255,0)
text_color = (255,255,255)
minArea = 1000
cap = cv.VideoCapture(0)
cap.set(3,frameWidth) #3 = width 640 pixels
cap.set(4,frameHeight) #4 = height 480 pixels
cap.set(10,100)#10 is brightness
model = load_model('full_batch_model.h5')
font = cv.FONT_HERSHEY_COMPLEX_SMALL


count = 0
score = 0
left_pred = 100
right_pred = 100

while True:
    success, img = cap.read()
    imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(imgGray,1.1,10)
    leftEye = lefteyeCascade.detectMultiScale(imgGray,1.2,20)
    rightEye = righteyeCascade.detectMultiScale(imgGray, 1.1, 4)

    cv.rectangle(img, (0,frameHeight-50) , (200,frameHeight) , (0,0,0) , thickness=cv.FILLED )

    for(x,y,w,h) in faces:
        area = w*h
        if area > minArea:
            img = cv.rectangle(img,(x,y), (x+w, y+h), box_color, 2)
            cv.putText(img,'Face', (x,y+h+20),
            font,1,text_color,2)
            
            
            
    for (x2,y2,w2,h2) in leftEye:
        lefteye_frame = img[y2:y2+h2, x2:x2+w2]
        count +=1
        left_eye = cv.cvtColor(lefteye_frame,cv.COLOR_BGR2GRAY)
        left_eye = cv.resize(left_eye,(66,66))
        left_eye = left_eye.astype('float32')
        left_eye = left_eye/255
        # cv.imshow("left",left_eye)
        left_eye = left_eye.reshape(66,66,-1)
        left_eye = np.expand_dims(left_eye,axis=0)
        left_pred = np.round(model.predict(left_eye))
        print(left_pred)
        eye_center = (x2 + w2//2, y2 + h2//2)
        radius = int(round((w2 + h2)*0.25))
        cv.ellipse(img,eye_center, (radius+5, radius),0,0,360,(0,0,255),2 )
       
    

                
    for (x3,y3,w3,h3) in rightEye:
        righteye_frame = img[y3:y3+h3, x3:x3+w3]
        count +=1
        right_eye = cv.cvtColor(righteye_frame,cv.COLOR_BGR2GRAY)
        right_eye = cv.resize(right_eye,(66,66))
        right_eye = right_eye.astype('float32')
        right_eye = right_eye/255
        # cv.imshow("right",right_eye)
        right_eye = right_eye.reshape(66,66,-1)
        right_eye = np.expand_dims(right_eye,axis=0)
        right_pred = np.round(model.predict(right_eye))
        print(right_pred)
        # cv.imshow("left",left_eye)
        eye_center = (x3 + w3//2, y3 + h3//2)
        radius = int(round((w3 + h3)*0.25))
        cv.ellipse(img,eye_center, (radius+5, radius),0,0,360,(255,0,0),2 )
        
# and right_pred==0
    if left_pred==0 and right_pred==0:
        score +=1 
        cv.putText(img,'Closed', (10,frameHeight-20),font,1,(255,255,255),1,cv.LINE_AA)
    else:
        score -=1
        cv.putText(img,'Open', (10,frameHeight-20),font,1,(255,255,255),1,cv.LINE_AA)
    if score<0:
        score = 0 
    cv.putText(img,'Score:'+str(score), (100,frameHeight-20),font,1,(255,255,255),1,cv.LINE_AA)
    if score == 15:
        try:
            sound.play()
            score=0
        except: # isplaying = False
            pass


    cv.imshow("Video",img)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
