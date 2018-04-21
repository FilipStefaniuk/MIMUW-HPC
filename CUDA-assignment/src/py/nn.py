import ctypes

def fit(data_X, data_Y, **kwargs):

    size = 100#len(data_Y)
    data_YY = []
    data_XX = [row[:20] for row in data_X]

    for x in data_Y:
        tmp = [0]*62
        tmp[x] = 1
        data_YY.append(tmp)
    
    data_X = [x for row in data_XX for x in row]
    data_Y = [x for row in data_YY for x in row]

    c_data_X = (ctypes.c_float * len(data_X))(*data_X)
    c_data_Y = (ctypes.c_float * len(data_Y))(*data_Y)

    c_epsilon = ctypes.c_float(kwargs['epsilon'])
    c_learning_rate = ctypes.c_float(kwargs['learning_rate'])
    c_epochs = ctypes.c_int(kwargs['epochs'])
    c_ranodm = ctypes.c_int(True) if kwargs['random'] == 'true' else ctypes.c_int(False) 

    dll = ctypes.CDLL('./nn.so', ctypes.RTLD_GLOBAL)
    dll.fit(c_data_X, c_data_Y, size, c_epsilon, c_learning_rate, c_epochs, c_ranodm)