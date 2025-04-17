import numpy as np

class KNN(object):
    """
        kNN classifier object.
    """

    def __init__(self, k=1, task_kind = "classification"):
        """
            Call set_arguments function of this class.
        """
        self.k = k
        self.task_kind =task_kind

    def fit(self, training_data, training_labels):
        """
            Trains the model, returns predicted labels for training data.
            Hint: Since KNN does not really have parameters to train, you can try saving the training_data
            and training_labels as part of the class. This way, when you call the "predict" function
            with the test_data, you will have already stored the training_data and training_labels
            in the object.

            Arguments:
                training_data (np.array): training data of shape (N,D)
                training_labels (np.array): labels of shape (N,)
            Returns:
                pred_labels (np.array): labels of shape (N,)
        """

        self.training_data = training_data
        self.training_labels = training_labels

        return self.predict(training_data)

    def predict(self, test_data):
        """
            Runs prediction on the test data.

            Arguments:
                test_data (np.array): test data of shape (N,D)
            Returns:
                test_labels (np.array): labels of shape (N,)
        """
        test_labels = np.empty(test_data.shape[0], dtype=self.training_labels.dtype)
        
        for i in range(test_data.shape[0]):
            test_sample = test_data[i]

            # Compute the Euclidean distance between a single test sample and all the training samples
            # while avoiding rounding errors
            dists = np.sqrt(np.maximum(np.sum((self.training_data - test_sample)**2, axis=1), 0))

            # find the nearest neighbors
            nn_indices = np.argsort(dists)[:self.k]

            # find the labels of the nearest neighbors
            neighbor_labels = self.training_labels[nn_indices]
            
            # vote (classification) or average (regression)
            if self.task_kind == "classification":
                test_labels[i] = np.argmax(np.bincount(neighbor_labels))
            else:  # regression
                test_labels[i] = np.mean(neighbor_labels)
        
        return test_labels