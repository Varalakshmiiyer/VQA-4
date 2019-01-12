import os
from keras.models import model_from_json
import h5py
import re
import nltk
import gensim
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.layers.recurrent import SimpleRNN
from gensim.models import Word2Vec
from keras.models import load_model
import numpy as np
from keras.preprocessing import sequence
def install():
    for d in dependencies:
        pip.main(['install', d])
    
    # after nltk module was installed
    import nltk
    for data in nltk_data:
	    extensions = [("taggers", "averaged_perceptron_tagger"),
                  ("corpora", "wordnet"),
                  ("tokenizers", "punkt")]

    missing = check_packages_exist(extensions)
    for ext_tuple in missing:
        nltk.download(data) 
        nltk.download(punkt)
import gensim 
from nltk import sent_tokenize, word_tokenize
from nltk.tokenize import word_tokenize
from nltk.tokenize import PunktSentenceTokenizer
from nltk.tokenize import TreebankWordTokenizer
from nltk.tokenize.regexp   import (RegexpTokenizer, WhitespaceTokenizer,
                                    BlanklineTokenizer, WordPunctTokenizer,
                                    wordpunct_tokenize, regexp_tokenize,
                                    blankline_tokenize)
_word_tokenize = TreebankWordTokenizer().tokenize
fp = open("./outputgen.csv","r")
data = fp.readlines()
names = []
ques = []
labels = []
new_names = []
label_map = {
   'body': 0,
   'head': 1,
   'limbs': 2
}
#assume we have items in csv in the format:
# image_id  label quetion -> for each question for a given id
line_no =0
for each in data:
    if line_no == 0:
        line_no = line_no+1
        continue
    #print(each)
    use = each.split(",")
    #print(use)
    image_id = use[0]
    label = use[1].lower()
    question = use[2]
    names.append(image_id)
    labels.append(label_map.get(label))
    tokens = _word_tokenize(question)
    ques.append(tokens)
    line_no = line_no + 1
    #print(line_no)
labels_for_lstm = []
path = "./train"
#check images in the path specified 
for f in os.listdir(path):
    for item in os.listdir(os.path.join(path,f)):
        if re.sub(".jpg",'',item):
            temp = re.sub(".jpg",'',item)
        else:
            temp = re.sub(".jpeg",'',item)
        new_names.append(temp)
# try to match the image Id with the images
indices = { value : [ i for i, v in enumerate(names) if v == value ] for value in new_names }
#print (indices)
ques_demo = []
for each in indices.keys():
	value = indices[each]

	put_label = labels[new_names.index(each)]
	for every in value:
		#ques_demo.append(ques[every])
		labels_for_lstm.insert(every,put_label)
#generate word vectors
def generatingwordvectors(data):
	vectors = []
	print("generatingwordvectors")
	model = Word2Vec(data, min_count=1, size=256)
	for each in data:
		for every in each:
			vectors.append(model.wv[every])

	return vectors
vectors = generatingwordvectors(ques)
# partition questions vectors
def partioningvectors(data, vectors):
	length_1 = []
	for each in data:
		length_1.append(len(each))
	list = []
	j=0
	for each in length_1:
		temp = []
		temp.extend(vectors[j:j+ each])

		list.append(temp)
		j = j + each
	print ("partioningvectors")
	list = sequence.pad_sequences(list, maxlen=256,dtype='float')

	return list
partition = partioningvectors(ques, vectors)

# data shaping
def shapingvectors(data_1):
	templist = []
	data=np.array(data_1)
	shapped = data.reshape((data.shape[0], data.shape[1], data.shape[2]))
	return shapped
shapped = shapingvectors(partition)

def labelsreshape(data):
	shapped = data.reshape((data.shape[0], 1))
	return shapped
labels_train = np.array(labels_for_lstm)
labels_shapped = labelsreshape(labels_train)

# fit LSTM Model
model = Sequential()
model.add(LSTM(512, input_shape=(256,256), return_sequences = False))
model.add(Dropout(0.5))
#model.add(LSTM(512))
model.add(Dropout(0.5))
model.add(Dense(3, activation = 'softmax'))
model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
print(model.summary())
model.fit(shapped,labels_shapped)
#model.save_weights('lstm.h5')
# serialize model to JSON
model_json = model.to_json()
with open("lstm_model.json", "w") as json_file:
    json_file.write(model_json)
print (model.wv["sinus"])
# serialize weights to HDF5
model.save_weights("lstm_model.h5")
