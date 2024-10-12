#Yelaman Zhenis

#Facial Emotion Recognition
#####################################
#Description:
#This program creates a model for Facial Emotion Recognition 
#First itis provided with database that is split into Training and Testing
#Utilized database is split into 80%-20% for Training and Testing
#Utilized database is already provided in size of 48x48 pixel grayscale images of faces
#The database provides the following emotions: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise
#This script uses Convolution Neural Network(CNN) to create a model for emotion recognition
#The Databbase is taken from here: https://www.kaggle.com/datasets/msambare/fer2013?resource=download
#####################################
#The script requires path for database images in variables
#Due to complexity, computation time for all layer of CNN is prolonged


#Libraries utilized
import sys, os
import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.callbacks import ReduceLROnPlateau, TensorBoard, EarlyStopping, ModelCheckpoint
from keras.models import load_model
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from keras.models import model_from_json
from keras.utils import to_categorical
from keras.utils import load_img
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D


#Training and Testing directories
trainDir = 'train'
testDir = 'test'


#get from images containing datapath, label from the foldername 
def data(dir):
    image_paths = []
    labels = []
    for label in os.listdir(dir):
        for imagename in os.listdir(os.path.join(dir, label)):
            image_paths.append(os.path.join(dir, label, imagename))
            labels.append(label)
        print(label, "completed")
    return image_paths, labels


#make 2-D table of images and their emotioin
train = pd.DataFrame()
train['image'], train['label'] = data(trainDir)
print(train)

test = pd.DataFrame()
test['image'], test['label'] = data(testDir)
print(test)



#Function to convert images into grayscale numpy 48x48 numpy array
def img_convert(images):
    features = []
    #show progress bar and convert each image into grayscale numpy 48x48 numpy array
    for image in tqdm(images):
        img = load_img(image, color_mode='grayscale')
        img = np.array(img)
        features.append(img)
    features = np.array(features)
    features = features.reshape(len(features), 48, 48, 1)
    return features


#Apply function to training and testing dataset
train_converterd = img_convert(train['image'])
test_converted = img_convert(test['image'])


#Normalize training and testing dataset
X_train = train_converterd / 255.0
X_test = test_converted / 255.0

#make labels from emotions to recignize
#make labels 0 if not correct emotion and 1 is correct emotion
LE = LabelEncoder()
LE.fit(train['label'])

# output variables for output as emotion labels 
y_train = LE.transform(train['label'])
y_test = LE.transform(test['label'])

#mkae output as a class vector to 0/1 class matrix
y_train = to_categorical(y_train, num_classes=7)
y_test = to_categorical(y_test, num_classes=7)



# Set CNN parameters
model = Sequential()

# Parameters for convolutional layers, activation function ReLU
model.add(Conv2D(128, kernel_size=(3,3), activation='relu', input_shape=(48,48,1)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))



model.add(Conv2D(256, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))



model.add(Conv2D(512, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))



model.add(Conv2D(512, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

# Make 2-D into 1-D
model.add(Flatten())

# Parameters for fully connected layers
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.4))

model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))

# Make output layer to output 7 labels or emotions
model.add(Dense(7, activation='softmax'))

#Model summary to keep track of trainable parameters
model.summary()

# Compile the model, categorical crossentropy loss function is recommended for multiple class models
model.compile(optimizer = 'adam', 
              loss = ['categorical_crossentropy'], 
              metrics = ['accuracy'] )

#Train the model
CNN = model.fit(X_train, y=y_train,batch_size=128,  epochs=100, 
                    validation_data=(X_test, y_test),shuffle=True,
                    )


#Save the model
model.save("model_final_1.keras")


scores = model.evaluate(np.array(X_test), np.array(y_test), batch_size=128)
print("Loss: " + str(scores[0]))
print("Accuracy: " + str(scores[1]))
