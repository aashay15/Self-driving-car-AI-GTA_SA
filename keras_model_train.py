from keras_model import keras_multi
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import adam


model = Sequential()
model.add(Dense(32, input_dim=6, activation='relu', name = 'input') )
#model.add(Dropout(0.5))
model.add(Dense(64, activation = 'relu'))
#model.add(Dropout(0.5))
model.add(Dense(128, activation = 'relu'))
#model.add(Dropout(0.5))
#model.add(Dense(256, activation = 'relu'))

#model.add(Dense(512, activation = 'relu'))

#model.add(Dense(256, activation = 'relu'))

#model.add(Dense(128, activation = 'relu'))

model.add(Dense(64, activation = 'relu'))

model.add(Dense(32, activation = 'relu'))



model.add(Dense(5, activation = 'softmax'))
ADAM = keras.optimizers.Adam(learning_rate= 1e-6, beta_1=0.9, beta_2=0.999, amsgrad=False)
SGD = keras.optimizers.SGD(learning_rate=0.00001, momentum=0.9, nesterov=False)
model.compile(loss='categorical_crossentropy',
                        optimizer = ADAM,
                        metrics = ['accuracy'])

#model = keras_multi()

MODEL_NAME = 'KERAS_MLP'
data = np.load('extended_balanced_2.npy', allow_pickle=True)

train = data[:7000]
test = data[7000:]
X = np.array([i[0] for i in train])
Y = np.array([j[1] for j in train])

test_x = np.array([i[0] for i in test])
test_y = np.array([j[1] for j in test])

model.fit(X,Y,
          epochs = 60,

          batch_size = 5,
          validation_data = (test_x,test_y))
keras.callbacks.callbacks.ModelCheckpoint("/Users/aashaysharma/Desktop/Self-driving-car-AI-GTA_SA/model_checkpoints", monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=10)


model.save('keras_model_test_6.model')