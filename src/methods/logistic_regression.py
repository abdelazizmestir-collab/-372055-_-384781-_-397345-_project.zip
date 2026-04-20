import numpy as np
from ..utils import get_n_classes, label_to_onehot, onehot_to_label


class LogisticRegression(object):
    """
    Logistic regression classifier.
    """

    def __init__(self, lr, max_iters=500):
        self.lr = lr
        self.max_iters = max_iters

    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.

        Arguments:
            training_data (np.array): training data of shape (N, D)
            training_labels (np.array): labels of shape (N,)
        Returns:
            pred_labels (np.array): predicted labels of shape (N,)
        """
        N, D = training_data.shape
        C = get_n_classes(training_labels)
        one_hot = label_to_onehot(training_labels, C)  # (N, C)

        self.W = np.random.normal(0, 0.1, (D, C))  # (D, C)

        for _ in range(self.max_iters):
            grad = gradient_cross_entropy(training_data, one_hot, self.W)
            self.W -= self.lr * grad

        return onehot_to_label(softmax(training_data @ self.W))

    def predict(self, test_data):
        """
        Runs prediction on the test data.

        Arguments:
            test_data (np.array): test data of shape (N, D)
        Returns:
            pred_labels (np.array): labels of shape (N,)
        """
        probs = softmax(test_data @ self.W)  # (N, C)
        return onehot_to_label(probs)


# ── Auxiliary functions ────────────────────────────────────────────────────────

def softmax(scores):
    """
    Numerically stable softmax.

    Args:
        scores (array): Input of shape (N, C)
    Returns:
        probs (array): Probabilities of shape (N, C), rows sum to 1.
    """
    scores = scores - np.max(scores, axis=1, keepdims=True)  # stability
    exp_scores = np.exp(scores)
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)


def cross_entropy_loss(data, one_hot_labels, W):
    """
    Multi-class cross-entropy loss.

    Args:
        data (array): Dataset of shape (N, D)
        one_hot_labels (array): One-hot labels of shape (N, C)
        W (array): Weights of shape (D, C)
    Returns:
        loss (float): Mean cross-entropy loss
    """
    probs = softmax(data @ W)
    log_probs = -np.log(probs + 1e-15)
    return np.mean(np.sum(one_hot_labels * log_probs, axis=1))


def gradient_cross_entropy(data, one_hot_labels, W):
    """
    Gradient of the multi-class cross-entropy w.r.t. W.

    Args:
        data (array): Dataset of shape (N, D)
        one_hot_labels (array): One-hot labels of shape (N, C)
        W (array): Weights of shape (D, C)
    Returns:
        grad (array): Gradient of shape (D, C)
    """
    probs = softmax(data @ W)          # (N, C)
    return data.T @ (probs - one_hot_labels)  # (D, C)