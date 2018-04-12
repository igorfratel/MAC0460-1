import numpy as np
from util import randomize_in_place


def linear_regression_prediction(X, w):
    """
    Calculates the linear regression prediction.

    :param X: design matrix
    :type X: np.ndarray(shape=(N, d))
    :param w: weights
    :type w: np.array(shape=(d, 1))
    :return: prediction
    :rtype: np.array(shape=(N, 1))
    """

    return X.dot(w)


def standardize(X):
    """
    Returns standardized version of the ndarray 'X'.

    :param X: input array
    :type X: np.ndarray(shape=(N, d))
    :return: standardized array
    :rtype: np.ndarray(shape=(N, d))
    """

    # YOUR CODE HERE:
    X_out = (X - X.mean(axis=0, keepdims=True))/X.std(axis=0,keepdims=True)
    # END YOUR CODE

    return X_out


def compute_cost(X, y, w):
    """
    Calculates  mean square error cost.

    :param X: design matrix
    :type X: np.ndarray(shape=(N, d))
    :param y: regression targets
    :type y: np.ndarray(shape=(N, 1))
    :param w: weights
    :type w: np.array(shape=(d,))
    :return: cost
    :rtype: float
    """

    # YOUR CODE HERE:
    J = np.asscalar((X.dot(w) - y).T.dot(X.dot(w)-y)/X.shape[0])
    # END YOUR CODE

    return J


def compute_wgrad(X, y, w):
    """
    Calculates gradient of J(w) with respect to w.

    :param X: design matrix
    :type X: np.ndarray(shape=(N, d))
    :param y: regression targets
    :type y: np.ndarray(shape=(N, 1))
    :param w: weights
    :type w: np.array(shape=(d,))
    :return: gradient
    :rtype: np.array(shape=(d,))
    """

    # YOUR CODE HERE:
    grad = 2/X.shape[0]*(X.dot(w) - y).T.dot(X).T
    # END YOUR CODE

    return grad


def batch_gradient_descent(X, y, w, learning_rate, num_iters):
    """
     Performs batch gradient descent optimization.

    :param X: design matrix
    :type X: np.ndarray(shape=(N, d))
    :param y: regression targets
    :type y: np.ndarray(shape=(N, 1))
    :param w: weights
    :type w: np.array(shape=(d,))
    :param learning_rate: learning rate
    :type learning_rate: float
    :param num_iters: number of iterations
    :type num_iters: int
    :return: weights, weights history, cost history
    :rtype: np.array(shape=(d,)), list, list
    """

    weights_history = [w]
    cost_history = [compute_cost(X, y, w)]

    # YOUR CODE HERE:
    for i in range(0, num_iters):
        w -= learning_rate*compute_wgrad(X, y, w)
        weights_history.append(w.flatten())
        cost_history.append(compute_cost(X, y, w))
    # END YOUR CODE

    return w, weights_history, cost_history


def stochastic_gradient_descent(X, y, w, learning_rate, num_iters, batch_size):
    """
     Performs stochastic gradient descent optimization

    :param X: design matrix
    :type X: np.ndarray(shape=(N, d))
    :param y: regression targets
    :type y: np.ndarray(shape=(N, 1))
    :param w: weights
    :type w: np.array(shape=(d, 1))
    :param learning_rate: learning rate
    :type learning_rate: float
    :param num_iters: number of iterations
    :type num_iters: int
    :param batch_size: size of the minibatch
    :type batch_size: int
    :return: weights, weights history, cost history
    :rtype: np.array(shape=(d, 1)), list, list
    """
    # YOUR CODE HERE:
    weights_history = [w.flatten()]
    cost_history = [compute_cost(X, y, w)]
    s = np.arange(X.shape[0])
    for i in range(0, num_iters):
        np.random.shuffle(s)
        rand_y = y[s]
        rand_X = X[s]
        w -= learning_rate*compute_wgrad(rand_X[0:batch_size], rand_y[0:batch_size], w)
        weights_history.append(w.flatten())
        cost_history.append(compute_cost(X, y, w))
    # END YOUR CODE

    return w, weights_history, cost_history
