import numpy as np
import sys
import csv

if __name__ == '__main__':
    # function to get data from console
    def get_data():
        my_data = np.genfromtxt(sys.argv[1], delimiter=',')
        X = my_data[:, 0:-1]
        ones = np.ones([X.shape[0], 1])
        X = np.concatenate((ones, X), 1)
        y = my_data[:, -1].reshape(-1, 1)
        return X, y


    # cost function
    def compute_cost(X, y, theta):
        summation = ((X @ theta.T) - y) ** 2
        return np.sum(summation) / (2 * len(X))


    # error calculation
    def get_error(X, y, theta):
        return np.sum((X @ theta.T - y) ** 2)


    # gradient descent
    def gradient_descent(X, y, theta, alpha, threshold):
        iter = 0
        cost = []
        writer = csv.writer(open('results_' + str(sys.argv[1]) + "_alpha" + str(sys.argv[2])
                                 + "_thres" + str(sys.argv[3]) + '.csv',"w", newline=""))
        while True:
            temp = theta
            weight = [iter]
            for j in theta[0]:
                weight .append(j)
            gradient = ((alpha) * (np.sum((X @ theta.T - y) * X, axis=0)))
            theta = theta - gradient
            cost += compute_cost(X, y, theta)
            next_theta = get_error(X, y, theta)
            now_theta = get_error(X, y, temp)
            iter += 1
            weight.append(now_theta)
            writer.writerow(weight)
            if now_theta - next_theta <= threshold:
                final_theta = theta
                break
        weight = [iter]
        for j in theta[0]:
            weight.append(j)
        weight.append(now_theta)
        writer.writerow(weight)
        return iter, final_theta, now_theta


    X, y = get_data()
    alpha = float(sys.argv[2])  # input learning rate
    threshold = float(sys.argv[3])  # input threshold
    theta = np.zeros([1, X.shape[1]])
    iter, g, error = gradient_descent(X, y, theta, alpha, threshold)
    print((iter), g[0], error)
    final_cost = compute_cost(X, y, g)
