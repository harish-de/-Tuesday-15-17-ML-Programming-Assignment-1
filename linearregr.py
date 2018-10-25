import numpy as np
import sys
import csv

if __name__ == '__main__':
    # function to get data from console
    def get_data():
        # my_data = np.genfromtxt(sys.argv[1], delimiter=',')
        my_data = np.genfromtxt('yacht.csv', delimiter=',')
        X = my_data[:, 0:-1]
        ones = np.ones([X.shape[0], 1])
        X = np.concatenate((ones, X), 1)
        y = my_data[:, -1].reshape(-1, 1)
        return X, y


    # cost function
    def compute_cost(X, y, theta):
        summation = ((X @ theta.T) - y) ** 2
        return np.sum(summation) / (2 * len(X))


    def get_error(X,y,theta):
        return np.sum(X @ theta.T - y)


    # gradient descent
    def gradient_descent(X, y, theta, alpha, threshold):
        iter = 0
        cost = np.zeros(27400)
        #print(get_error(X, y, theta))
        writer = csv.writer(open("results.csv", 'w'), newline = "")
        while True:
            temp = theta
            theta = theta - (alpha / len(X)) * (np.sum((X @ theta.T - y) * X, axis=0))
            cost = compute_cost(X, y, theta)
            iter += 1
            output_string_a = str(iter)
            output_string_b = str(theta[0])
            output_string_c = str(np.sum((X @ theta.T - y)**2))
            writer.writerow([output_string_a + " "])
            writer.writerow([output_string_b.replace("[","").replace("]","") + ","])
            writer.writerow([output_string_c])
            if float(get_error(X, y, theta) - get_error(X, y, temp )) <= threshold:
                break
        print(output_string_a, output_string_b.replace("[","").replace("]",""), output_string_c)
        return iter, theta, cost


    X, y = get_data()
    alpha = 0.0001  # float(sys.argv[2]) learning rate
    threshold = 0.0001  # float(sys.argv[3])
    theta = np.zeros([1, X.shape[1]])
    iter, g, cost = gradient_descent(X, y, theta, alpha, threshold)
    #print(g)
    final_cost = compute_cost(X, y, g)
    #print(final_cost)
