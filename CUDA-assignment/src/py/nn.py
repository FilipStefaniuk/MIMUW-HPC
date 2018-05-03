import ctypes


def normalize(data):

    maxes = [0]*len(data[0])
    mins = [250]*len(data[0])

    for row in data:
        for i, val in enumerate(row):
            maxes[i] = max([maxes[i], val])
            mins[i] = min([mins[i], val])

    return [[(val - mins[i])/(0.00001 + maxes[i] - mins[i]) for i, val in enumerate(row)] for row in data]


def fit(data_X, data_Y, **kwargs):

    size = len(data_Y)

    data_X  = normalize(data_X)

    data_Y = [[1 if x == i else 0 for i in range(62)] for x in data_Y]

    data_X = [row[i] for i in range(4096) for row in data_X]
    data_Y = [row[i] for i in range(62) for row in data_Y]

    c_data_X = (ctypes.c_float * len(data_X))(*data_X)
    c_data_Y = (ctypes.c_float * len(data_Y))(*data_Y)

    c_epsilon = ctypes.c_float(kwargs['epsilon'])
    c_learning_rate = ctypes.c_float(kwargs['learning_rate'])
    c_epochs = ctypes.c_int(kwargs['epochs'])
    c_ranodm = ctypes.c_int(True) if kwargs['random'] == 'true' else ctypes.c_int(False) 

    dll = ctypes.CDLL('./nn.so', ctypes.RTLD_GLOBAL)
    dll.fit(c_data_X, c_data_Y, size, c_epsilon, c_learning_rate, c_epochs, c_ranodm)