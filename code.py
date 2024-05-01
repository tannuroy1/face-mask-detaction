


import cv2
import os

img=cv2.imread("E:\image.jpg.jpg")

img.shape

img[0]

img

import matplotlib.pyplot as plt

plt.imshow(img)

pip install opencv-python

pip install cvlib-python

get_ipython().run_cell_magic('cmd', '', 'pip install cmake')

pip install opencv-python

pip install opencv-contrib-python

pip install opencv-python

import cv2

pip install python-for-android

import numpy as np

while True:
    cv2.imshow('result',img)
    if cv2.waitKey(2)==27:
         break
cv2.destroyAllWindows()        

haar_data=cv2.CascadeClassifier('C:/Users/Dell/anaconda3/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')

haar_data.detectMultiScale(img)

while True:
    faces=haar_data.detectMultiScale(img)
    for x,y,w,h in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0))
    cv2.imshow('result',img)
    if cv2.waitKey(2)==27:
         break
cv2.destroyAllWindows()  

capture=cv2.VideoCapture(0)
while True:
    flag,img=capture.read()
    if flag:
        faces=haar_data.detectMultiScale(img)
        for x,y,w,h in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0))
    cv2.imshow('result',img)
    if cv2.waitKey(2)==27:
         break
            
capture.release()
cv2.destroyAllWindows()  

capture=cv2.VideoCapture(0)
data=[]
while True:
    flag,img=capture.read()
    if flag:
        faces=haar_data.detectMultiScale(img)
        for x,y,w,h in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),4)
            face=img[y:y+h,x:x+w, :]
            face=cv2.resize(face,(50,50))
            print(len(data))
            if len(data)<400:
                data.append(face)
    cv2.imshow('result',img)
    if cv2.waitKey(2)==27 or len(data)>=200:
         break
cv2.destroyAllWindows()  




capture=cv2.VideoCapture(0)
data=[]
while True:
    flag,img=capture.read()
    if flag:
        faces=haar_data.detectMultiScale(img)
        for x,y,w,h in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),4)
            face=img[y:y+h,x:x+w, :]
            face=cv2.resize(face,(50,50))
            print(len(data))
            if len(data)<400:
                data.append(face)
    cv2.imshow('result',img)
    if cv2.waitKey(2)==27 or len(data)>=200:
         break
cv2.destroyAllWindows()  

np.save('without_mask.npy',data)

np.save('with1_mask.npy',data)

import numpy as np

import matplotlib.pyplot as plt

plt.imshow(data[20])

import cv2
import cv2

with_mask=np.load('with1_mask.npy')
without_mask=np.load('without_mask.npy')

with_mask.shape

without_mask.shape

with_mask=with_mask.reshape(200,50*50*3)
without_mask=without_mask.reshape(200,50*50*3)

without_mask.shape

x=np.r_[with_mask,without_mask]

labels=np.zeros(x.shape[0])

names={0:'Mask',1:'NoMask'}

labels[200:]=1.0

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,labels,test_size=0.15)

x_train.shape

from sklearn.decomposition import PCA

pca=PCA(n_components=3)
x_train=pca.fit_transform(x_train)
x_train[0]

svm=SVC()
svm.fit(x_train,y_train)

x_test=pca.transform(x_test)
y_pred=svm.predict(x_test)

accuracy_score(y_test,y_pred)

haar_data=cv2.CascadeClassifier('C:/Users/Dell/anaconda3/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
capture=cv2.VideoCapture(0)
data=[]
font=cv2.FONT_HERSHEY_COMPLEX
while True:
    flag,img=capture.read()
    if flag:
        faces=haar_data.detectMultiScale(img)
        for x,y,w,h in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),4)
            face=img[y:y+h,x:x+w, :]
            face=cv2.resize(face,(50,50))
            face=face.reshape(1,-1)
            face= pca.transform(face)
            pred=svm.predict(face)
            n=names[int(pred)]
            cv2.putText(img,n,(x,y),font,1,(244,250,250),2)
            print(n)
    cv2.imshow('result',img)
    if cv2.waitKey(2)==27:
         break
cv2.destroyAllWindows()  




##project description

# Face Mask Detection using SVM and OpenCV

This project utilizes machine learning techniques to detect faces in real-time video streams and classifies them as wearing a mask or not. It uses the Support Vector Machine (SVM) classifier and OpenCV for face detection.

## Installation

- Install Python (3.6 or higher)
- Install the required libraries using pip:


## Usage

- Run the `train.py` script to train the SVM classifier on the provided dataset.
- Run the `detect.py` script to detect faces in real-time video streams and classify them as wearing a mask or not.

## Dataset

The dataset used for training the classifier contains images of people wearing and not wearing masks. It can be found at [link to dataset].

## Acknowledgements

- [OpenCV](https://opencv.org/) for face detection
- [scikit-learn](https://scikit-learn.org/) for the SVM classifier
- [matplotlib](https://matplotlib.org/) for data visualization











