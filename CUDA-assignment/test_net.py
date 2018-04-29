from src.py import loader
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.activations import softmax
from keras.losses import categorical_crossentropy

if __name__ == "__main__":
    data_X, data_Y = loader.load_data('./Training/')
    

    data_X = np.array(data_X)
    data_Y = np.array([[1 if x == i else 0 for i in range(62)] for x in data_Y])

    print(data_X.shape)
    # model = Sequential()
    # model.add(Dense(8192,input_dim=4096))
    # model.add(Dense(6144, activation='relu'))
    # model.add(Dense(3072, activation='relu'))
    # model.add(Dense(1024, activation='relu'))
    # model.add(Dense(62))
    # model.add(Activation(softmax))
    
    # model.compile(optimizer='rmsprop',loss=categorical_crossentropy, metrics=['accuracy'])
    # model.summary()

    # model.fit(data_X, data_Y, epochs=1, verbose=2)
