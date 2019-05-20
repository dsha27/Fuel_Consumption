import cv2
import numpy as np
import webbrowser
import subprocess as sp
import os
from os import listdir
from os.path import isfile, join
print(cv2.__version__)

# Get the training data we previously made
data_path = '/root/Desktop/disha/'
# a=listdir('d:/faces')
# print(a)
# """
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]

# Create arrays for training data and labels
Training_Data, Labels = [], []

# Open training images in our datapath
# Create a numpy array for training data
for i, files in enumerate(onlyfiles):
    image_path = data_path + onlyfiles[i]
    images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    Training_Data.append(np.asarray(images, dtype=np.uint8))
    Labels.append(i)
# 
# Create a numpy array for both training data and labels
Labels = np.asarray(Labels, dtype=np.int32)

aditi_model=cv2.face_LBPHFaceRecognizer.create()

# Initialize facial recognizer
# model = cv2.face_LBPHFaceRecognizer.create()
# model=cv2.f
# NOTE: For OpenCV 3.0 use cv2.face.createLBPHFaceRecognizer()

# Let's train our model 
aditi_model.train(np.asarray(Training_Data), np.asarray(Labels))

print("Model trained sucessefully")










face_classifier = cv2.CascadeClassifier('/root/Desktop/haarcascade_frontalface_default.xml')

def face_detector(img, size=0.5):
    
    # Convert image to grayscale
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if faces is ():
        return img, []	
    
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
        roi = img[y:y+h, x:x+w]
        roi = cv2.resize(roi, (200, 200))
    return img, roi


# Open Webcam
#cap = cv2.VideoCapture(0)
import requests
url ="http://192.168.43.1:8080/shot.jpg"

    #ret, frame = cap.read()
while True: #for video streaming
    geturl=requests.get(url) #connect to the url
    photoweb=geturl.content #load content
    type(photoweb) # show datatype
    photobyte=bytearray(photoweb) #photo binary(bytes) into binary array
    imageId=np.array(photobyte) #bytearray converted into id numpy array
    frame = cv2.imdecode(imageId,-1) 
    reframe=cv2.resize(frame,(400,400))
    #cv2.imshow("hi",reframe)
    
    image, face = face_detector(frame)
    
    try:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        # Pass face to prediction model
        # "results" comprises of a tuple containing the label and the confidence value
        results = aditi_model.predict(face)
        if results[1] < 500:
            confidence = int( 100 * (1 - (results[1])/400) )
            display_string = str(confidence) + '% Confident it is User'
            
        cv2.putText(image, display_string, (100, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (255,120,150), 2)
        
        if confidence > 80:
            cv2.putText(image, "Hey Aditi", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
            cv2.imshow('Face Recognition', image )
            webbrowser.open("http://localhost:8888/notebooks/fuel/fuelConsumption.ipynb")
	
            #os.system('python36 /root/Desktop/PythonCodeBatch2/menu.py')
            cv2.destroyAllWindows()                      
            break
        else:
            cv2.putText(image, "I Dont Rec You", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
            cv2.imshow('Face Recognition', image )

    except:
        cv2.putText(image, "No Face Found", (220, 120) , cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
        cv2.putText(image, "Locked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
        cv2.imshow('Face Recognition', image )
        pass
        
    if cv2.waitKey(1) == 13: #13 is the Enter Key
        break
        
cv2.destroyAllWindows()
#cap.release()   
