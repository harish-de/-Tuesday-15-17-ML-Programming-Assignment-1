import numpy as np
import sys
import csv

'''this function takes in csv file and converts it into matrices X, y 
this function also adds a column of  ones to the X matrix.
Input:
        .csv file from the console
Output:
        X matrix with all the feature + X0 = 1
        y matrix with all the last column values from .csv file
'''


def get_data():
    my_data = np.genfromtxt(sys.argv[1], delimiter=',')
    X = my_data[:, 0:-1]
    ones = np.ones([X.shape[0], 1])
    X = np.concatenate((ones, X), 1)
    y = my_data[:, -1].reshape(-1, 1)
    return X, y


'''this function takes in X, y matrices and theta value and outputs cost of the function
Input: 
        X = a numpy_array
        y = a numpy_array
        theta = (1 * n) numpy_array
Output:
        cost = float'''


def compute_cost(X, y, theta):
    summation = ((X @ theta.T) - y) ** 2
    return np.sum(summation) / (2 * len(X))


# error calculation
'''this function takes in X, y matrices and theta value and outputs sum of errors of the function
Input: 
        X = a numpy_array
        y = a numpy_array
        theta = (1 * n) numpy_array
Output:
        error = float'''


def get_error(X, y, theta):
    return np.sum((X @ theta.T - y) ** 2)


# gradient descent
'''this function takes in X, y, theta matrices and alpha, threshold values and outputs the gradient at that particular 
time
In addition this function also writes the output vales to a .csv file that is available in the working folder
Input:
        X = a numpy_array
        y = a numpy_array
        theta = (1 * n) numpy_array
Output:
        iter = int
        final_theta = (1 * n) numpy_array
        now_error = error at that particular step'''


def gradient_descent(X, y, theta, alpha, threshold):
    iter = 0
    cost = []
    writer = csv.writer(open('results_' + str(sys.argv[1]) + "_alpha" + str(sys.argv[2])
                             + "_thres" + str(sys.argv[3]) + '.csv', "w", newline=""))
    while True:
        temp = theta
        weight = [iter]
        for j in theta[0]:
            weight.append(j)
        gradient = ((alpha) * (np.sum((X @ theta.T - y) * X, axis=0)))
        theta = theta - gradient
        cost += compute_cost(X, y, theta)
        next_error = get_error(X, y, theta)
        now_error = get_error(X, y, temp)
        iter += 1
        weight.append(now_error)
        writer.writerow(weight)
        if now_error - next_error <= threshold:
            final_theta = theta
            break
    weight = [iter]
    for j in theta[0]:
        weight.append(j)
    weight.append(now_error)
    writer.writerow(weight)
    return iter, final_theta, now_error

# main method starts from here


if __name__ == '__main__':
    X, y = get_data()
    alpha = float(sys.argv[2])  # input learning rate
    threshold = float(sys.argv[3])  # input threshold
    theta = np.zeros([1, X.shape[1]])
    iter, g, error = gradient_descent(X, y, theta, alpha, threshold)
    print (iter, g[0], error)
    final_cost = compute_cost(X, y, g)
