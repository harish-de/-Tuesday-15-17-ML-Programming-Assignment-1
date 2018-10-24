import numpy as np
import sys
import csv
def get_next():
    my_data = np.genfromtxt('data.csv', delimiter=',')
    X = my_data[:, -2].reshape(-1, 1)
    ones = np.ones([X.shape[0],1])
    X = np.concatenate([ones, X],1)
    y = my_data[:,-1].reshape(-1,1)
    print(X)
    print(y)
get_next()
