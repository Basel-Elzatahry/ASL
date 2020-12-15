import os
import cv2
import numpy as np
from time import time
import tensorflow as tf
import keras
from tensorflow.keras import utils
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, MaxPool2D
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pylab as py


""" Plots one image from every single class
    
    Parameter: 
        train_dir - the directory of the folder containing the train data set
    
    Returns: 
        none

"""

def visualize(train_dir):
    classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 
           'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 
           'W', 'X', 'Y', 'Z', 'space', 'nothing', 'del']
    plt.figure(figsize=(15, 15))
    for i in range (0,29):
        plt.subplot(8,8,i+1)
        plt.xticks([])
        plt.yticks([])
        path = train_dir + "/{0}/{0}650.jpg".format(classes[i])
        img = plt.imread(path)
        plt.imshow(img)
        plt.xlabel(classes[i])
        
        

""" Loads the data from the given directory and split it into test and train x and y
    
    Parameter: 
        train_dir - the directory of the folder containing the train data set
    
    Returns: 
        x_train - the images of the train data set
        x_test - the images of the test data set
        y_train - the labels of the train data set
        y_test - the labels of the test data set

"""


def loadData(train_dir):
    images = []
    labels = []
    size = 64,64
    index = -1
    for folder in os.listdir(train_dir):
        index +=1
        for image in os.listdir(train_dir + "/" + folder):
            temp_img = cv2.imread(train_dir + '/' + folder + '/' + image)
            temp_img = cv2.resize(temp_img, size)
            images.append(temp_img)
            labels.append(index)
    
    images = np.array(images)
    images = images.astype('float32')/255.0
    labels = utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size = 0.1)
    
    print('Loaded', len(x_train),'images for training,','Train data shape =', x_train.shape)
    print('Loaded', len(x_test),'images for testing','Test data shape =', x_test.shape)
    
    return x_train, x_test, y_train, y_test



""" Computes the accuracy of our results
    
    Parameter: 
        y_test - the targeted results
        predictions - the predicted results of our model
    Returns: 
        none

"""

def accuracy(y_test, predictions):
    training_accuracy = 0
    for i in range(len(predictions)):
        training_accuracy +=(y_test[i] == predictions[i]).all()
    training_accuracy /= len(predictions)
    print(training_accuracy*100,"%")
 

""" Plots the train loss and the validation loss
    
    Parameter: 
        history - the history of training the data
    Returns: 
        none

"""
    
def lossPlotter(history):
    loss=history.history['loss']
    val_loss=history.history['val_loss']
    py.plot(loss,label='training',color='red')
    py.plot(val_loss,label='validation',color='blue')
    py.legend()

    
""" Plots the train accuracy and the validation accuracy
    
    Parameter: 
        history - the history of training the data
    Returns: 
        none

"""
def accPlotter(history):
    loss=history.history['accuracy']
    val_loss=history.history['val_accuracy']
    py.plot(loss,label='training',color='red')
    py.plot(val_loss,label='validation',color='blue')
    py.legend()
    
    
def modelArch():
    model = Sequential()

    model.add(Conv2D(16 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu' , input_shape = (64,64,3)))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))

    model.add(Conv2D(32 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu' , input_shape = (64,64,3)))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
    model.add(Dropout(0.4))

    model.add(Conv2D(64 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))

    model.add(Conv2D(128 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
    model.add(Dropout(0.4))

    model.add(Conv2D(256 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))

    model.add(Conv2D(256 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(256 , activation = 'relu'))
    model.add(Dense(128 , activation = 'relu'))
    model.add(Dense(29 , activation = 'softmax'))

    return model