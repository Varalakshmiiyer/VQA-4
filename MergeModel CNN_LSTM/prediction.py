import cv2
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
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
image_path = cv2.imread('./train/body/1477-7819-8-101-3.jpg')
img = cv2.resize((image_path),(256, 256))
#img = image.load_img(image_path, target_size=(256, 256))
x = image.img_to_array(img) * 1.0/255
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
output1 = model2.predict(x)
print('Predicted:', (output1)[0])
#print('Predicted:', (preds)[0])
t = Tokenizer()
question1 = 'what does chest ct reveal after pelvic evisceration'
# fit the tokenizer on the documents
t.fit_on_texts(question1)
# summarize what was learned
print(t.word_counts)
print(t.document_count)
print(t.word_index)
print(t.word_docs)
# integer encode documents
encoded_docs = t.texts_to_matrix(question1, mode='count')
print(encoded_docs)
#hard code the question in some variable question1 = 'your question'
output2=model1.predict(t.word_docs)
#create numpy array with concatenation of output1, and output2 -> assign to some variable output3
output3 = np.concatenate((output1, output2), axis = 0)
result = modelfinal.predict(output3)
print(result)