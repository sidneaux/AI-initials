# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 20:03:25 2018

@author: Sidneaux
neural test
"""

import keras
from keras.datasets import mnist
from keras.layers import Dense,Dropout
from keras.models import Sequential
from keras.optimizers import SGD
import matplotlib.pyplot as plt

#loading the mnist datasets
(train_x, train_y), (test_x, test_y) = mnist.load_data()

#Defining the model
model = Sequential()
model.add(Dense(units=128, activation="relu", input_shape=(784,)))
model.add(Dense(units=128,activation="relu"))
model.add(Dense(units=128,activation="relu"))
model.add(Dense(units=10,activation="softmax"))

#specifying the training components
model.compile(optimizer=SGD(0.01),loss="categorical_crossentropy",metrics=["accuracy"])

#Load the pretrained model 
model.load_weights("mnistmodel.h5")

#Normalize the test dataset 
test_x = test_x.astype('float32') / 255

#Extract a specific image 
img = test_x[167]

#Create a flattened copy of the image 
test_img = img.reshape((1,784))

#Predict the class 
img_class = model.predict_classes(test_img)

classname = img_class[0] 
print("Class: ",classname)

#Display the original non-flattened copy of the image 
plt.title("Prediction Result: %s"%(classname))
plt.imshow(img) 
plt.show()



