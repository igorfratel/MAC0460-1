#9793565

import numpy as np

def normal_equation_prediction(X, y):
    """
    Calculates the prediction using the normal equation method.
    You should add a new column with 1s.

    :param X: design matrix
    :type X: np.array
    :param y: regression targets
    :type y: np.array
    :return: prediction
    :rtype: np.array
    """
    # complete a função e remova a linha abaixo
    X_new = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
    w = np.linalg.inv(X_new.T.dot(X_new)).dot(X_new.T).dot(y)
    prediction = X_new.dot(w)
    return prediction
