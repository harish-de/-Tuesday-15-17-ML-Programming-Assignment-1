import numpy as np
import matplotlib.pyplot as plt
import sys
import csv


def get_data():
    my_data = np.genfromtxt('data.csv', delimiter=',')
    X = my_data[:, 0:-1]
    ones = np.ones([X.shape[0], 1])
    X = np.concatenate((ones, X), 1)
    y = my_data[:, -1].reshape(-1, 1)
    return X, y


X, y = get_data()
print(X[5][6])
print(y[5])