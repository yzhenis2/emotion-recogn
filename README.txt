#Yelaman Zhenis
#CSCI 6397 SPECIAL TOPICS Multimedia 
#Facial Emotion Recognition
#####################################
#Description:
#The programs create a model for Facial Emotion Recognition 
#First itis provided with database that is split into Training and Testing
#Utilized database is split into 80%-20% for Training and Testing
#Utilized database is already provided in size of 48x48 pixel grayscale images of faces
#The database provides the following emotions: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise
#This script uses Convolution Neural Network(CNN) to create a model for emotion recognition
#The Databbase is taken from here: https://www.kaggle.com/datasets/msambare/fer2013?resource=download
#####################################


1. Start by making sure these libraries are installed

pandas
numpyopencv-contrib-python
scikit-learn
matplotlib
tensorflow
keras
jupyter
notebook
tqdm


you can install them by running the following 
"pip install [library name]"

2. Open the program "project_update3.py" in a preferred environment.
   In the program "project_update3.py" provide paths for testing and training directories.
	The variables are "trainDir" and "testDir".
   Each directories should contain emotion directories named after the emotion 
   and with corresponding emotions.

   Run the program.

	BE AWARE THAT THE COMPUTATION TIME DEPENDS ON YOUR SYSTEM AND DATASET 
	AND CAN TAKE MORE THAN TWO HOURS TO CREATE A MODEL
   
   "project_update3.py" should output a model named "model_final.keras" 

3. Open the program "face_detection.py" in a preferred environment.
   In the program "project_update3.py" provide path of an image with face 
   that you'd like to classify.
	The variable is "img"

   Run the program. The program should output original image with detected face and emotion above it.


##################################### OPTIONAL LIVE EMOTION RECOGNITION.#####################################

4. Open the program "face_detection_live.py" in a preferred environment.
   Run the program. The program should output live webcam frame with detected face and emotion above it.
