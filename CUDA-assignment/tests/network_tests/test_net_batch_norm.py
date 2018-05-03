from src.py import loader
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.normalization import BatchNormalization
from keras.activations import softmax
from keras.losses import categorical_crossentropy
from keras.initializers import RandomUniform
from keras.optimizers import SGD
from sklearn.preprocessing import normalize
import numpy as np
import random

if __name__ == "__main__":
    
    data = loader.load_data('./Training/')

    random.shuffle(data)
    
    data_X = [x for x, _ in data]
    data_Y = [y for _, y in data]

    data_X = np.array(data_X)
    data_Y = np.array([[1 if x == i else 0 for i in range(62)] for x in data_Y])

    data_X = normalize(data_X, axis=0)

    model = Sequential()
    model.add(Dense(8192,input_dim=4096,))
    model.add(BatchNormalization())
    model.add(Dense(6144, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(3072, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(1024, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(62))
    model.add(Activation(softmax))
    
    model.compile(optimizer='rmsprop', loss=categorical_crossentropy, metrics=['accuracy'])
    model.summary()

    model.fit(data_X, data_Y, epochs=10, verbose=1, batch_size=32)
