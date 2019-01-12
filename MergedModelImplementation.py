from keras import layers
from keras import models
from keras.layers import Merge
from keras.layers import Add
from keras.layers import *
from keras.models import Sequential
import h5py
import gensim
import keras
from keras.models import load_model
from keras.layers.merge import Concatenate
from keras.layers.embeddings import Embedding
from numpy import array
lstm_model = lstmmodel()
print(len(lstm_model.layers))
model1 = models.Sequential()
model1.add(lstm_model)
vgg_model = vggmodel()
print(len(vgg_model.layers))
#print(vgg_model.layers[314].output.shape)
model2 = models.Sequential()
model2.add(vgg_model)
# Specifying input Layer
iL = [layers.Input(shape=(256,256)), layers.Input(shape=(256,256,3))]
hL = [model1(iL[0]), model2(iL[1])]
#Concatenating two models
mergemodel = concatenate([model1(iL[0]),model2(iL[1])])
output = Dense(1000, activation='tanh')(mergemodel)
outputfin = Dropout(0.5)(output)
#Softmax Activation
finalized_output =Dense(1000, activation='softmax')(outputfin)
modelfinal = models.Model(input= iL, output =finalized_output)
#Compiling the model
modelfinal.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=['accuracy'])
#modelfinal.fit(iL, finalized_output, epochs=10, verbose=0)  # Fit the model

#print('Accuracy: %0.3f' % accuracy)
		