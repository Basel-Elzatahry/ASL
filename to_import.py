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
        path = train_dir + "/{0}/{0}27.jpg".format(classes[i])
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


def split_data(train_dir):
    images = []
    labels = []
    size = 64,64
    index = -1
    for folder in os.listdir(train_dir):
        index +=1
        for image in os.listdir(train_dir + "/" + folder):
            temp_img = cv2.imread(train_dir + '/' + folder + '/' + image)
            
            if type(temp_img) is np.ndarray:
                temp_img = cv2.resize(temp_img, size)
                images.append(temp_img)
                labels.append(index)
    images = np.array(images)
    images = images.astype('float32')/255.0
    labels = utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size = 0.1)
    np.save("x2_train.npy",x_train)
    np.save("x2_test.npy",x_test)
    np.save("y2_train.npy",y_train)
    np.save("y2_test.npy",y_test)
    
#     print('Loaded', len(x_train),'images for training,','Train data shape =', x_train.shape)
#     print('Loaded', len(x_test),'images for testing','Test data shape =', x_test.shape)
    
    return x_train, x_test, y_train, y_test

def load_data(train_dir):
    if train_dir == '../asl_alphabet_train/asl_alphabet_train':
        x_train = np.load("x_train.npy")
        x_test = np.load("x_test.npy")
        y_train = np.load("y_train.npy")
        y_test = np.load("y_test.npy")
        print('Loaded', len(x_train),'images for training,','Train data shape =', x_train.shape)
        print('Loaded', len(x_test),'images for testing','Test data shape =', x_test.shape)
    elif train_dir =="both": 
        x_train = np.load("x_both_train.npy")
        x_test = np.load("x_both_test.npy")
        y_train = np.load("y_both_train.npy")
        y_test = np.load("y_both_test.npy")
    else:
        x_train = np.load("x2_train.npy")
        x_test = np.load("x2_test.npy")
        y_train = np.load("y2_train.npy")
        y_test = np.load("y2_test.npy")
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
    
    return ("%s-pc",str(training_accuracy*100))
 

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
    model.add(Conv2D(16 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu' , input_shape = (64,64,3)))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))

    model.add(Conv2D(32 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu' , input_shape = (64,64,3)))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
    model.add(Conv2D(32 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu' , input_shape = (64,64,3)))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
    model.add(Dropout(0.4))

    model.add(Conv2D(64 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
    model.add(Conv2D(64 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))

    model.add(Conv2D(128 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
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


def modelArch_less():
    model = Sequential()
    model.add(Conv2D(16 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu' , input_shape = (64,64,3)))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
    model.add(Conv2D(16 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu' , input_shape = (64,64,3)))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))

    model.add(Conv2D(32 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu' , input_shape = (64,64,3)))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
    model.add(Conv2D(32 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu' , input_shape = (64,64,3)))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
    model.add(Dropout(0.4))

    model.add(Conv2D(64 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
    model.add(Conv2D(64 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))

    model.add(Dropout(0.4))
    model.add(Flatten())
#     model.add(Dense(256 , activation = 'relu'))
#     model.add(Dense(128 , activation = 'relu'))
    model.add(Dense(29 , activation = 'softmax'))

    return model

def modelArch_less2():
    model = Sequential()
    model.add(Conv2D(16 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu' , input_shape = (64,64,3)))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
    model.add(Conv2D(16 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu' , input_shape = (64,64,3)))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
    model.add(Dropout(0.4))

    model.add(Conv2D(32 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu' , input_shape = (64,64,3)))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
    model.add(Conv2D(32 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu' , input_shape = (64,64,3)))
    model.add(BatchNormalization())
    model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))


    model.add(Dropout(0.4))
    model.add(Flatten())
#     model.add(Dense(256 , activation = 'relu'))
#     model.add(Dense(128 , activation = 'relu'))
    model.add(Dense(29 , activation = 'softmax'))

    return model

def map_index_to_letter(index,train_dir):
    folder = os.listdir(train_dir)[index]
   
    return folder


def confusion_matrix(predictions,y_test,train_dir):
#     train_dir = '../asl_alphabet_train/asl_alphabet_train'
    from sklearn.metrics import confusion_matrix
    letter_pred = [map_index_to_letter(np.argmax(i),train_dir)for i in predictions]
    letter_true = [map_index_to_letter(np.argmax(i),train_dir)for i in y_test]
    cm = confusion_matrix(letter_true, letter_pred)
    
    plt.figure(figsize = (24, 20))
    ax = plt.subplot()
    plt.imshow(cm, interpolation = 'nearest', cmap = plt.cm.Reds)
    ticks = np.arange(len(os.listdir(train_dir)))
    plt.xticks(ticks,os.listdir(train_dir))
    plt.yticks(ticks, os.listdir(train_dir))
    plt.ylabel('True letter')
    plt.xlabel('Predicted letter')
    plt.colorbar()
    
    
def mislabelled_list(predictions,y_test):
    train_dir = '../asl_alphabet_train/asl_alphabet_train'
    label_true_to_false ={}
    for i in range(predictions.shape[0]):
        if (np.argmax(predictions[i])-np.argmax(y_test[i])!= 0):

            real = map_index_to_letter(np.argmax(y_test[i]),train_dir)
            pred = map_index_to_letter(np.argmax(predictions[i]),train_dir)

            if real not in label_true_to_false.keys():
                label_true_to_false[real] =[]
            label_true_to_false[real].append(pred)
      

    mislabelled= []
    for x in label_true_to_false.keys():
        mislabelled.append((x,len(label_true_to_false[x])))

    mislabelled = sorted(mislabelled,key=lambda mislabelled: mislabelled[1],reverse=True)

    for i in mislabelled:
        print(i[0],":",i[1])
    return mislabelled


            
            
    