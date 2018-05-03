from keras.datasets import mnist
from keras.utils import np_utils
from sklearn.preprocessing import normalize
import ctypes
import numpy

def fit(data_x, data_y, **kwargs):

    # l = 10000
    
    # data_x = data_x[:l]
    # data_y = data_y[:l]

    # Transpose in memory
    data_x = data_x.T.copy()
    data_y = data_y.T.copy()

    data_x = data_x.astype(numpy.float32)
    data_y = data_y.astype(numpy.float32)

    c_data_x = data_x.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    c_data_y = data_y.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    c_len = ctypes.c_int(data_x.shape[1])
    c_epsilon = ctypes.c_float(0)
    c_lr = ctypes.c_float(0.1)
    c_epochs = ctypes.c_int(10)
    c_random = ctypes.c_int(1)

    dll = ctypes.CDLL('./test.so', ctypes.RTLD_GLOBAL)
    dll.fitMNIST(c_data_x, c_data_y, c_len, c_epsilon, c_lr, c_epochs, c_random)


if __name__ == '__main__':

    (x_train, y_train), _ = mnist.load_data()
    
    num_pixels = x_train.shape[1] * x_train.shape[2]
    x_train = x_train.reshape(x_train.shape[0], num_pixels).astype('float32')
    x_train = normalize(x_train, axis=0)

    y_train = np_utils.to_categorical(y_train)

    fit(x_train, y_train)