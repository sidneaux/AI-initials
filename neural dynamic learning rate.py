# -*- coding: utf-8 -*-
"""
Created on Sat Jul  7 19:00:52 2018

@author: ACER
"""

import keras
from keras.datasets import mnist
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD

#loading the mnist dataset
(train_x, train_y) , (test_x, test_y) = mnist.load_data()

#normalize the data 
train_x = train_x.astype('float32') / 255 
test_x = test_x.astype('float32') / 255

#Print the shape of the data array
print("Train Images: ", train_x.shape)
print("Train Labels: ", train_y.shape)
print("Test Images: ", test_x.shape)
print("Test Labels: ", test_y.shape)

#Flatten the images
train_x = train_x.reshape(60000,784)
test_x = test_x.reshape(10000,784)

#Encoding the labels to the vectors
train_y = keras.utils.to_categorical(train_y, 10)
test_y = keras.utils.to_categorical(test_y, 10)

#Define the model
model = Sequential()
model.add(Dense(units=128,activation="relu",input_shape=(784,)))
model.add(Dense(units=128,activation="relu"))
model.add(Dense(units=128,activation="relu"))
model.add(Dense(units=10,activation="softmax")

#Define the learning rate schedule function
def lr_Schedule(epoch):
    lr = 0.1
    if epoch > 15:
        lr = lr/100
    elif epoch > 10:
        lr = lr/10
    elif epoch > 5:
        lr = lr/10
    print("learning rate: ", lr)
    return lr
#Pass the scheduler function to the Learning Rate Scheduler class 
lr_scheduler = LearningRateScheduler(lr_schedule)

#Specify the training components
 model.compile(optimizer=SGD(lr_schedule(0)),loss="categorical_crossentropy",
               metrics=["accuracy"])
 #Fit the model
 model.fit(train_x,train_y,batch_size=32,epochs=20,shuffle=True,verbose=1,callbacks=[lr_scheduler])
 
#Evaluate the accuracy of the test dataset
accuracy = model.evaluate(x=test_x,y=test_y,batch_size=32)

print("Accuracy: ",accuracy[1])



