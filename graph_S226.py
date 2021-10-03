# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 16:55:29 2019

@author: ESA
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 21:56:38 2019

@author: ESA
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 18:10:19 2019

@author: ESA

graph_3h
"""

import os 
import random
import shutil
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense ,Dropout
import tensorflow as tf
from keras.optimizers import SGD
import keras
from keras.callbacks import EarlyStopping
print("Runing the graph_S226...\n")
with tf.device('/device:GPU:0'):
#with tf.device('/cpu:0'):   
    def random_copyfile(srcPath,dstPath,numfiles):
        name_list=list(os.path.join(srcPath,name) for name in os.listdir(srcPath))
        random_name_list=list(random.sample(name_list,numfiles))
        if not os.path.exists(dstPath):
            os.mkdir(dstPath)
        for oldname in random_name_list:
            shutil.move(oldname,oldname.replace(srcPath, dstPath))
            #shutil.copyfile(oldname,oldname.replace(srcPath, dstPath))

    # Initialising the CNN
    classifier = Sequential()
    # Step 1 - Convolution
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
    # Step 3 - Flattening
    classifier.add(Flatten())
    # Step 4 - Full connection
    classifier.add(Dense(units = 1024, activation = 'relu'))   ##128->1000  #input_dim 
    classifier.add(Dense(units = 1024, activation = 'relu'))   ##128->1000  #input_dim 
    classifier.add(Dropout(0.5))         ####
    #classifier.add(Dense(units = 128, activation = 'relu'))   ##128->1000
    #classifier.add(Dropout(0.5))         ####
    classifier.add(Dense(units =3, activation = 'softmax'))  #sigmoid
    # Compiling the CNN
    classifier.compile(optimizer = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08), loss = 'categorical_crossentropy', metrics = ['accuracy'])   #binary_crossentropy categorical_crossentropy  adam
    # Part 2 - Fitting the CNN to the images 
    
    #classifier.load_weights("graph_5.h5")
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)
    print(classifier.summary())
    

    from keras.preprocessing.image import ImageDataGenerator
    
    train_datagen = ImageDataGenerator(rescale = 1./255,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True)
    test_datagen = ImageDataGenerator(rescale = 1./255)
    
    training_set = train_datagen.flow_from_directory('C:/Users/ESA.DESKTOP-NTEFSD9/.spyder-py3/dataset_S224/training',
    target_size = (128,128),
    batch_size = 32,
    color_mode="rgb",
    class_mode="categorical",
    shuffle=True)#,class_mode = 'binary')
    
    test_set = test_datagen.flow_from_directory('C:/Users/ESA.DESKTOP-NTEFSD9/.spyder-py3/dataset_S224/testing',
    target_size = (128, 128),
    batch_size = 32,
    color_mode="rgb",
    class_mode="categorical",
    shuffle=True)         #,class_mode = 'binary')
    
    
    STEP_SIZE_TRAIN=training_set.n//training_set.batch_size
    STEP_SIZE_VALID=test_set.n//test_set.batch_size
    
    history=classifier.fit_generator(training_set,
    steps_per_epoch = 500,  #8000 3000 STEP_SIZE_TRAINt rain_num/64
    epochs = 25,  #25
    validation_data = test_set,
    validation_steps =50) #500
  
    classifier.save_weights("graph_S226.h5")    


sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))


import matplotlib.pyplot as plt
training_loss = history.history['acc']
test_loss = history.history['val_acc']

plt.plot(training_loss, 'r--')
plt.plot(test_loss, 'b-')
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
#plt.show()
plt.savefig('C:/Users/ESA.DESKTOP-NTEFSD9/Desktop/graph_S226/graph_S226_accuracy.png')
plt.close()

import matplotlib.pyplot as plt

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
#plt.show()
plt.savefig('C:/Users/ESA.DESKTOP-NTEFSD9/Desktop/graph_S226/graph_S226_loss.png')
plt.close()


f = open('C:/Users/ESA.DESKTOP-NTEFSD9/Desktop/graph_S226/history.txt', 'a', encoding = 'UTF-8')    # 也可使用指定路徑等方式，如： C:\A.txt
f.write('graph_S226 SGD\n')
f.write('acc,  ')
for value in history.history['acc']:
    f.write(str(value))
    f.write(',  ')
f.write('\n')
f.write('val_acc,  ')
for value in history.history['val_acc']:
    f.write(str(value))
    f.write(',  ')
f.write('\n')
f.write('loss,  ')
for value in history.history['loss']:
    f.write(str(value))
    f.write(',  ')
f.write('\n')
f.write('val_loss,  ')
for value in history.history['val_loss']:
    f.write(str(value))
    f.write(',  ')
f.write('\n')
f.write('\n')
f.write('\n')
f.close()

print("End the graph_S226.\n")