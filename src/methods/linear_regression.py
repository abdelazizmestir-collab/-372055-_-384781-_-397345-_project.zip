import numpy as np
from src import utils

class LinearRegression(object):
    """
    Linear regression.
    """

    def __init__(self, weights = None):
        """
        Initialize the new object (see dummy_methods.py)
        and set its arguments.
        """
        self.weights = None

    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.

        Hint: You can use the closed-form solution for linear regression
        (with or without regularization). Remember to handle the bias term.

        Arguments:
            training_data (np.array): training data of shape (N,D)
            training_labels (np.array): regression target of shape (N,)
        Returns:
            pred_labels (np.array): target of shape (N,)
        """
        X_train = utils.append_bias_term(training_data)
        self.weights = self.get_w_analytical(X_train, training_labels)
        pred_labels = X_train @ self.weights 
        return pred_labels

    def predict(self, test_data):
        """
        Runs prediction on the test data.

        Arguments:
            test_data (np.array): test data of shape (N,D)
        Returns:
            pred_labels (np.array): labels of shape (N,)
        """
        X_test = utils.append_bias_term(test_data)
        pred_labels = X_test @ self.weights 
        return pred_labels
    
    ###########################################################
    
    #Helper functions
    def get_w_analytical(self, X_train, y_train):
        """
        Calculates the weights using the analytical closed form solution

        Arguments:
            X_train (np.array): test data of shape (N, D)
            y_train (np.array): labels of training set of shape (N,)
        Returns:
            w (np.array): weight vector that is solution of our minimization problem
        """

        w = np.linalg.pinv(X_train) @ y_train
        return w

    