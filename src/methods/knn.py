import numpy as np
from ..utils import label_to_onehot


class KNN(object):
    """
    kNN classifier object.
    """

    def __init__(self, k=1, task_kind="classification"):
        """
        Call set_arguments function of this class.
        """
        self.k = k
        self.task_kind = task_kind

    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.

        Hint: Since KNN does not really have parameters to train, you can try saving
        the training_data and training_labels as part of the class. This way, when you
        call the "predict" function with the test_data, you will have already stored
        the training_data and training_labels in the object.

        Arguments:
            training_data (np.array): training data of shape (N,D)
            training_labels (np.array): labels of shape (N,)
        Returns:
            pred_labels (np.array): labels of shape (N,)
        """
        self.training_data = training_data
        self.training_labels = training_labels
        self.n_samples = training_data.shape[0]
        pred_labels = self.predict(training_data)

       
        return pred_labels

    def predict(self, test_data):
        """
        Runs prediction on the test data.

        Arguments:
            test_data (np.array): test data of shape (N,D)
        Returns:
            test_labels (np.array): labels of shape (N,)
        """
        dists = self._pairwise_sq_distances(test_data)
        nn_idx = np.argpartition(dists, kth=self.k - 1, axis=1)[:, :self.k]
                # argpartition uses a 0-based "kth index" convention
                # argsort fully sorts each row — O(N log N) work
                # while argpartition only ensures that the k smallest
                # elements are in the first k positions, — O(N) work.
        nn_labels = self.training_labels[nn_idx]

        if self.task_kind == "classification":
            C = int(self.training_labels.max()) + 1  # nn_labels: (M, k). Flatten, one-hot (M*k, C), reshape to (M, k, C)
            onehots = label_to_onehot(nn_labels.reshape(-1), C=C).reshape(
                nn_labels.shape[0], self.k, C)   # Sum one-hots across the k neighbors -> vote counts per class.
            votes = onehots.sum(axis=1)  # (M, C)
            return np.argmax(votes, axis=1)

        else:  # regression
            test_labels = np.mean(nn_labels, axis=1)
            return test_labels

    def _pairwise_sq_distances(self, test_data):

        test_sqs = np.sum(test_data ** 2, axis=1, keepdims=True)
        train_sqs = np.sum(self.training_data ** 2, axis=1, keepdims=True)
        cross = test_data @ self.training_data.T

        dists = test_sqs + train_sqs.T - 2 * cross    ## ∥x−y∥**2 = ∥x∥**2 + ∥y∥**2 − 2x⊤y
        return np.maximum(dists, 0.0)