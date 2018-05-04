import ctypes
import random


# def normalize(data):

#     maxes = [0]*len(data[0])
#     mins = [250]*len(data[0])

#     for row in data:
#         for i, val in enumerate(row):
#             maxes[i] = max([maxes[i], val])
#             mins[i] = min([mins[i], val])

#     return [[(val - mins[i])/(0.00000001 + maxes[i] - mins[i]) for i, val in enumerate(row)] for row in data]


def fit(data, **kwargs):

    # Get Size of data
    size = len(data)

    # Shuffle data
    random.shuffle(data)

    # Get dataX and dataY
    data_X = [x for x, _ in data]
    data_Y = [y for _, y in data]

    # Normalize input
    # data_X  = normalize(data_X)

    # One hot encoding for output
    data_Y = [[1 if x == i else 0 for i in range(62)] for x in data_Y]

    # Transpose & flatten & normalize 
    data_X = [row[i] / 250 for i in range(4096) for row in data_X]
    data_Y = [row[i] for i in range(62) for row in data_Y]

    # To ctypes
    c_data_X = (ctypes.c_float * len(data_X))(*data_X)
    c_data_Y = (ctypes.c_float * len(data_Y))(*data_Y)
    c_epsilon = ctypes.c_float(kwargs['epsilon'])
    c_learning_rate = ctypes.c_float(kwargs['learning_rate'])
    c_epochs = ctypes.c_int(kwargs['epochs'])
    c_ranodm = ctypes.c_int(True) if kwargs['random'] == 'true' else ctypes.c_int(False) 

    # Call c function
    dll = ctypes.CDLL('./libmlp.so', ctypes.RTLD_GLOBAL)
    dll.fit.restype = ctypes.c_float
    
    return dll.fit(c_data_X, c_data_Y, size, c_epsilon, c_learning_rate, c_epochs, c_ranodm)
    