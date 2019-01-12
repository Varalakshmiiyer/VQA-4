from keras.models import model_from_json
import numpy as np
import h5py
import gensim
from keras.models import load_model

# load json and create model lstm
def lstmmodel(): 
  json_file = open('lstm_model.json', 'r')
  loaded_model_json = json_file.read()
  json_file.close()
  loaded_model = model_from_json(loaded_model_json)
# load weights into new model
  loaded_model.load_weights("lstm_model.h5")
  loaded_model.layers.pop()
  loaded_model.layers.pop()
  loaded_model.layers.pop()
  loaded_model.summary()
  print("Loaded LSTM Model")
  return loaded_model
#load json and create model cnn images
def vggmodel():
  json_file = open('modelimagescnn_new.json', 'r')
  loaded_model_json = json_file.read()
  json_file.close()
  loaded_model = model_from_json(loaded_model_json)
# load weights into new model
  loaded_model.load_weights("inceptionmodelwtsnew_imagenet.h5")
  loaded_model.layers.pop()
  loaded_model.summary()
  print("Loaded model from cnn images")
  return loaded_model

