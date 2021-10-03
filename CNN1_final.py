import os 

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense ,Dropout
import tensorflow as tf
import keras
from keras.optimizers import SGD,Adam
from sklearn.metrics import classification_report, confusion_matrix



classifier = Sequential()
# Convolution & MaxPooling & Dropout
classifier.add(Conv2D(32, (7, 7),padding='same', input_shape = (128,128, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Dropout(0.5)) 
classifier.add(Conv2D(64, (5, 5), padding='same', activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Dropout(0.5)) 
classifier.add(Conv2D(64, (3, 3), padding='same', activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Dropout(0.5)) 
classifier.add(Conv2D(128, (3, 3), padding='same', activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Dropout(0.5)) 
classifier.add(Conv2D(128, (3, 3), padding='same', activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Dropout(0.5)) 
# Flattening
classifier.add(Flatten())
# Full connection
classifier.add(Dense(units = 1024, activation = 'relu'))  
classifier.add(Dense(units = 1024, activation = 'relu'))  
classifier.add(Dropout(0.5))         
# Softmax 
classifier.add(Dense(units =2, activation = 'softmax'))    # units = 2 for whether is knee or shoulder 
# Compiling the CNN
classifier.compile(optimizer = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, 
epsilon=1e-08), loss = 'categorical_crossentropy', metrics = ['accuracy'])   
# Fitting the CNN to the images 

# Load weights of training model
classifier.load_weights("graph_CNN1.h5") 

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
shear_range = 0.2,
zoom_range = 0.2,
horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)
t_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('C:/Users/ESA.DESKTOP-NTEFSD9/.spyder-py3/dataset_S172/training', 
target_size = (128,128),
batch_size = 32,
color_mode="rgb",
class_mode="categorical",
shuffle=True)

test_set = test_datagen.flow_from_directory('C:/Users/ESA.DESKTOP-NTEFSD9/.spyder-py3/dataset_S172/testing',
target_size = (128, 128),                
batch_size = 1,
color_mode="rgb",
class_mode="categorical",
shuffle=False)        


history=classifier.fit_generator(training_set,
steps_per_epoch = 1,  
epochs = 1,  
validation_data = test_set,
validation_steps =1) 


import numpy as np
import pandas as pd

# Making new predictions
from keras.preprocessing import image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import glob
 

file_path = 'D:/ultrasound_exam/20190508202218/WEN/MEDIA/'
f_names = glob.glob(file_path + '*.jpg')
 
img = []
# Load image to the list
for i in range(len(f_names)):
    images = image.load_img(f_names[i], target_size=(128, 128))
    x = image.img_to_array(images)
    x = np.expand_dims(x, axis=0)
    img.append(x)
    f_result = classifier.predict(x)
   
    ans=0     #knee
    max_p=f_result[0][0]
    if f_result[0][1]>max_p:
        ans=1 #shoulder
        max_p=f_result[0][1]
	
	# ans write into a txt file 
    f = open('C:/Users/ESA.DESKTOP-NTEFSD9/Desktop/result_0/result.txt', 'w')    
    f.write(str(ans))
    f.close()
    
   


