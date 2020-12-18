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
    x_train, x_test, y_train, y_test = load_data(train_dir)
    
    datagen = ImageDataGenerator(
                    rotation_range=45, 
                    width_shift_range=0.6,
                    height_shift_range=0.6, 
                    horizontal_flip=True,
                    vertical_flip=False,
                    )

    

    vgg = vgg16.VGG16(include_top=False, weights='imagenet',input_shape = (64,64,3))

    out = vgg.layers[-1].output
    output = keras.layers.Flatten()(out)

    vgg_model = Model(vgg.input, output) 

    vgg.trainable = False
    for layer in vgg.layers:
        layer.trainable = False


    model = Sequential()
    model.add(vgg_model)
#     model.add(Dropout(0.4))

    model.add(Dense(29, activation='sigmoid',input_shape=vgg_model.output_shape))
    
    if optimizer =="rms":
        opt = keras.optimizers.RMSprop(lr=lr)
    else: 
        opt = keras.optimizers.Adam(lr=lr)
        
    model.compile(optimizer = opt , loss = 'binary_crossentropy' , metrics = ['accuracy'])
    model.summary()
    es = EarlyStopping(monitor ='val_loss', min_delta=0.001, patience=50)
    
    x2_train, x2_test, y2_train, y2_test= load_data("../archive2")
    datagen.fit(x2_train)
    history =model.fit(x2_train,y2_train,batch_size = batch_size,epochs=epochs,callbacks=[es],validation_split=0.2, shuffle = True)
    
    model.save('VGG-MODELS/VGG16_lr-%f_BS-%d_E-%d_OPT-%s.h5'%(lr,batch_size,epochs,optimizer))
    np.save("VGGhistory_1_-%f_BS-%d_E-%d_OPT-%s.h5"%(lr,batch_size,epochs,optimizer),history.history)
    
    from to_import import accuracy, lossPlotter, accPlotter
    import pylab as py
#     predictions = (model.predict(x_test) > 0.5).astype("int32")
#     ac = accuracy(y_test, predictions)
    

    predictions = (model.predict(x_test) > 0.5).astype("int32")
    ac = accuracy(y_test, predictions)
    
    
 
    f = open("out_vgg.txt", "a")
    f.write("----------------------------------------------------------------")
    f.write("--------TESTING ON DIFFERENT DATASET (1) ---------------")
    f.write("\n")
    f.write(str('VGG-MODELS/VGG16_lr-%f_BS-%d_E-%d_OPT-%s.h5'%(lr,batch_size,epochs,optimizer)))
    f.write(str(ac))
    f.write("\n")
    f.write("----------------------------------------------------------------")
    f.write("\n")
    f.close()



if __name__=="__main__":

    batch_size = [4,8,16,32,64,128,256]
#     batch_size = [32]
    lr = [1E-3,1E-4,1E-5]
#     lr = [1E-3] 
    epochs =[5,10]
#     epochs= [20,30,40,50]
#     opt = ["rms","adam"]
    opt ="adam"
   
    for e in epochs: 
        for o in opt:
            for l in lr:
                for b in batch_size:
                    train_l(b,l,int(e),o)
    
     
                    
    
