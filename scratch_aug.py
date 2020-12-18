import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout , BatchNormalization,Input, MaxPooling2D,SeparableConv2D, ZeroPadding2D
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import vgg16
from keras.callbacks import EarlyStopping
from keras.applications.vgg16 import preprocess_input
import matplotlib.pyplot as plt
import numpy as np
from to_import import load_data



def train_l( batch_size,lr, epochs,optimizer):
    
   
    train_dir = '../asl_alphabet_train/asl_alphabet_train'
    x1_train, x1_test, y1_train, y1_test = load_data(train_dir)
    train_dir2 = '../archive-2'
    x2_train, x2_test, y2_train, y2_test = load_data(train_dir2)
    x_train, x_test, y_train, y_test = load_data("both")
    
    datagen = ImageDataGenerator(
    rotation_range=25, 
    width_shift_range=0.2,
    height_shift_range=0.2, 
    horizontal_flip=True,
    vertical_flip=False)


    datagen.fit(x_train)
    from to_import import modelArch
    model = modelArch()
    rms = keras.optimizers.Adam(lr=lr)
    model.compile(optimizer = rms , loss = 'binary_crossentropy' , metrics = ['accuracy'])
    model.summary()
    es = EarlyStopping(monitor ='val_loss', min_delta=0.001, patience=50)
    
    history =model.fit(x_train,y_train,batch_size = batch_size,epochs=epochs,callbacks=[es],validation_split=0.2, shuffle = True)
    
    model.save('SCRATCH-MODELS/SCR_BOTH_lr-%f_BS-%d_E-%d_OPT-%s.h5'%(lr,batch_size,epochs,optimizer))
    np.save("SCR_BOTH_history-%f_BS-%d_E-%d_OPT-%s.h5"%(lr,batch_size,epochs,optimizer),history.history)
    
    from to_import import accuracy, lossPlotter, accPlotter
    import pylab as py
    predictions = (model.predict(x_test) > 0.5).astype("int32")
    ac = accuracy(y_test, predictions)
    
    
 
    f = open("scratch_aug.txt", "a")
    f.write("----------------------------------------------------------------")
    f.write("--------TESTING ON BOTH DATASETS---------------")
    f.write("\n")
    f.write(str('SCRATCH-MODELS/SCR_BOTH_lr-%f_BS-%d_E-%d_OPT-%s.h5'%(lr,batch_size,epochs,optimizer)))
    f.write(str(ac))
    predictions = (model.predict(x1_test) > 0.5).astype("int32")
    ac = accuracy(y1_test, predictions)
    f.write("\n")
    f.write("--------TESTING ON DATASET 1---------------")
    predictions = (model.predict(x1_test) > 0.5).astype("int32")
    ac = accuracy(y1_test, predictions)
    f.write(str(ac))
    f.write("\n")
    f.write("--------TESTING ON DATASET 2---------------")
    predictions = (model.predict(x2_test) > 0.5).astype("int32")
    ac = accuracy(y2_test, predictions)
    f.write(str(ac))
    f.write("----------------------------------------------------------------")
    f.write("\n")
    f.close()



if __name__=="__main__":

    batch_size = [16,32,64,128,256]
#     batch_size = [32]
    lr = [1E-3,1E-4,1E-5]
#     lr = [1E-3] 
    epochs =[5,10,15]
#     epochs= [20,30,40,50]
#     opt = ["rms","adam"]
    opt ="adam"
   
    for e in epochs: 
            for l in lr:
                for b in batch_size:
                    train_l(b,l,int(e),opt)
    
     
                    
    
