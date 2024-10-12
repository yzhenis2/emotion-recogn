#Yelaman Zhenis

#Facial Emotion Recognition
#####################################
#Description:
#This program detects and classifies emotions of a .jpg file

#Libraries utilized
import numpy as np
from keras.models import load_model
import cv2

#load the model and give it classifications
model = load_model('model_final.keras')
emotions = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}


#read an image 
img = cv2.imread('photo_1_2024-04-29_13-06-01.jpg') 
  
#convert image to grayscale 
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
  
#load haar cascade classifier file 
haar_cascade = cv2.CascadeClassifier('Haarcascade_frontalface_default.xml') 
  
#face detection application on the grayscale image 
faces_rect = haar_cascade.detectMultiScale(gray_img, 1.3, 5) 


#Function to convert images into grayscale numpy 48x48 numpy array
def img_convert(image):
    resize = np.array(image)
    resize = resize.reshape(1, 48, 48, 1)
    resize = resize / 255.0
    return resize


  
# Iterating through rectangles of detected faces 
for (x, y, w, h) in faces_rect:
    #draw a rectangle around the face
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 1)
    #Resize detected face to 48x48 pixels and make pixel array
    image_gray = gray_img[y:y + h, x:x + w]
    image_gray = cv2.resize(image_gray, (48, 48))
    image_pixels = img_convert(image_gray)
    #Make classification prediction
    prediction = model.predict(image_pixels)
    # Get the predicted label for emotion
    prediction_label = emotions[prediction.argmax()]
    # Display the predicted emotion label near the detected face
    cv2.putText(img, emotions[int(np.argmax(prediction))], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)

  
cv2.imshow('Detected faces', img) 
  
cv2.waitKey(0) 

