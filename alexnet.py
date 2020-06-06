import numpy as np
import pandas as pd
from collections import Counter
from random import shuffle
import keras.utils
from numpy import array
from numpy import argmax
from keras.backend import one_hot


train_data = np.load('extended_balanced_2.npy', allow_pickle=True)
encoded_data = []
df = pd.DataFrame(train_data)
#print(df.head())
#print(Counter(df[1].apply(str)))
for data in train_data:
    keys = train_data[0][1]
    keys = array(keys)

#print(keys)
    keys = keras.utils.to_categorical(keys)
    #print(keys)

#inverted = argmax(encoded)
#keras_encode = one_hot(keys,5)
#print(keys)
#print(encoded)
#print(inverted)
#print(keras_encode)
#print(encoded_data)
print(train_data)