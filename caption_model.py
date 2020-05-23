

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.applications.xception import Xception
from keras.models import load_model
from pickle import load
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import argparse
from emo_utils import *

def extract_features(filename, model):
  try:
    image = Image.open(filename)

  except:
    print("ERROR: Couldn't open image! Make sure the image path and extension is correct")
  image = image.resize((299,299))
  image = np.array(image)
  # for images that has 4 channels, we convert them into 3 channels
  if image.shape[2] == 4: 
    image = image[..., :3]
  image = np.expand_dims(image, axis=0)
  image = image/127.5
  image = image - 1.0
  feature = model.predict(image)
  return feature


def word_for_id(integer, tokenizer):
 for word, index in tokenizer.word_index.items():
     if index == integer:
         return word
 return None

def generate_desc(model, tokenizer, photo, max_length):
    in_text = 'start'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        pred = model.predict([photo,sequence], verbose=0)
        pred = np.argmax(pred)
        word = word_for_id(pred, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'end':
            break
    return in_text

def sentences_to_indices(X, word_to_index, max_len):
    """
    Converts an array of sentences (strings) into an array of indices corresponding to words in the sentences.
    The output shape should be such that it can be given to `Embedding()` (described in Figure 4). 
    
    Arguments:
    X -- array of sentences (strings), of shape (m, 1)
    word_to_index -- a dictionary containing the each word mapped to its index
    max_len -- maximum number of words in a sentence. You can assume every sentence in X is no longer than this. 
    
    Returns:
    X_indices -- array of indices corresponding to words in the sentences from X, of shape (m, max_len)
    """
    
    m = X.shape[0]                                   # number of training examples
    
    # Initialize X_indices as a numpy matrix of zeros and the correct shape (≈ 1 line)
    X_indices = np.zeros((m,max_len))
    
    for i in range(m):                               # loop over training examples
        
        # Convert the ith training sentence in lower case and split is into words. You should get a list of words.
        sentence_words = X[i].lower()
        sentence_words=sentence_words.split()
        
        # Initialize j to 0
        j = 0
        
        # Loop over the words of sentence_words
        for w in sentence_words:
            if(j>=max_len):
              break
            # Set the (i,j)th entry of X_indices to the index of the correct word.
            X_indices[i,j] = word_to_index[w]
            # Increment j to j + 1
            j = j+1

            
    
    return X_indices


def caption_generator(img_path):
  max_length = 32
  tokenizer = load(open("tokenizer.p","rb"))
  model = load_model('./models/model_9.h5')
  xception_model = Xception(include_top=False, pooling="avg")

  photo = extract_features(img_path, xception_model)
  img = Image.open(img_path)

  description = generate_desc(model, tokenizer, photo, max_length) #save to text file
  
  #print("\n\n")
  f = open("./static/io/caption.txt", "r+")  
  
  # absolute file positioning 
  f.seek(0)  
  # to erase all data  
  f.truncate()  
  f.close()  
  f = open("./static/io/caption.txt", "w")
  f.write(description)
  f.close()  
  #print(description)
  #plt.imshow(img)    




def emotion_generator():
  description=description[5:-3]  #read from text file
  dict=['love','sport','happy/joy','sad/disappointed','dinner/food/eating']
  category_path='/content/drive/My Drive/Major Project/Caption Classification'
  predict_category=load_model('predict_category.h5')
  word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('glove.6B.50d.txt')
  maxLen = 10  
  # Change the sentence below to see your prediction. Make sure all the words are in the Glove embeddings.  
  x_test = np.array([description])
  X_test_indices = sentences_to_indices(x_test, word_to_index, maxLen)
  dict[np.argmax(predict_category.predict(X_test_indices))]


caption_generator('./static/d.jpg')    