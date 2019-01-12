import os
from numpy import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from keras.layers.core import Flatten
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras import optimizers
from keras import backend as K
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from sklearn.cross_validation import train_test_split
import tensorflow as tf
run_opts = tf.RunOptions(report_tensor_allocations_upon_oom = True)
if K.backend()=='tensorflow':
    K.set_image_dim_ordering("tf")
	
path1 = "E:\vqa assignment\2015medcluster\Training Data\clusterdata"
path2 = "E:\vqa assignment\2015medcluster\Training Data\cluster_resized"
listing = os.listdir("clusterdata")
num_samples = size(listing)
print (num_samples)
from PIL import Image
img_rows = 224
img_cols = 224
img_width = 3
for file in listing:
	im = Image.open("clusterdata" + '\\' + file )
	img = im.resize((img_rows, img_cols))
	img.save("cluster_resized" + '\\' + file, "JPEG" )

	
imlist = os.listdir("cluster_resized")
im1 = array(Image.open("cluster_resized" + '\\' + imlist[0]))
m,n = im1.shape[0:2]

immatrix = array([array(Image.open('cluster_resized' + '\\' + im2)).flatten() for im2 in imlist],'f')
#immatrix = immatrix.astype(int)
#print (immatrix)
labels = np.ones((num_samples,),dtype = int)
labels = [2,2,2,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1,1,1,1,1,1,0,1,1,1,0,0,0,0,1,0,1,1,1]
num_classes = 4
data, label = shuffle(immatrix, labels, random_state = 2)
train_data= [data, label]
(X,Y) = (train_data[0], train_data[1])
#X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state = 42) 
X_train = X
Y_train = Y
X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 3 )
#X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 3)

X_train = X_train.astype('float32')/255
#X_test = X_test.astype('float32')/255

y_train = np_utils.to_categorical(Y_train, 4)
#y_test = np_utils.to_categorical(Y_test, 4)



input_shape_data = (224, 224, 3)

model = Sequential()
model.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu', input_shape=input_shape_data, data_format = 'channels_last', padding='same'))
model.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(128, kernel_size = (3, 3), activation = 'relu', padding='same'))
model.add(Conv2D(128, kernel_size = (3, 3), activation = 'relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(256, kernel_size = (3, 3), activation = 'relu', padding='same'))
model.add(Conv2D(256, kernel_size = (3, 3), activation = 'relu', padding='same'))
model.add(Conv2D(256, kernel_size = (3, 3), activation = 'relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(512, kernel_size = (3, 3), activation = 'relu', padding='same'))
model.add(Conv2D(512, kernel_size = (3, 3), activation = 'relu', padding='same'))
model.add(Conv2D(512, kernel_size = (3, 3), activation = 'relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(512, kernel_size = (3, 3), activation = 'relu', padding='same'))
model.add(Conv2D(512, kernel_size = (3, 3), activation = 'relu', padding='same'))
model.add(Conv2D(512, kernel_size = (3, 3), activation = 'relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd)
model_info = model.fit(X_train, y_train, batch_size=256, epochs=1)
plot_model_history(model_info)
print(model.summary())





