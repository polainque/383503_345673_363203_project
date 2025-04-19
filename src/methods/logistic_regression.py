import numpy as np
from ..utils import get_n_classes, label_to_onehot


class LogisticRegression(object):
    """
    Logistic regression classifier.
    """

    def __init__(self, lr, max_iters=500):
        """
        Initialize the new object (see dummy_methods.py)
        and set its arguments.

        Arguments:
            lr (float): learning rate of the gradient descent
            max_iters (int): maximum number of iterations
        """
        self.lr = lr
        self.max_iters = max_iters
        self.weights = None
        self.bias = None

    def softmax(self, scores):
        """
        Compute the softmax function.
        
        Arguments:
            scores (array): Input scores of shape (N, C) where N is number of samples
                           and C is number of classes
            
        Returns:
            probs (array): Softmax probabilities of shape (N, C)
        """
        # Subtract max for numerical stability
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.

        Arguments:
            training_data (array): training data of shape (N,D)
            training_labels (array): regression target of shape (N,)
        Returns:
            pred_labels (array): target of shape (N,)
        """
        training_data = (training_data - np.mean(training_data, axis=0)) / np.std(training_data, axis=0)
        N, D = training_data.shape
        # Get number of classes and convert labels to one-hot encoding
        n_classes = get_n_classes(training_labels)
        y_onehot = label_to_onehot(training_labels, n_classes)
        
        # Initialize weights and bias
        self.weights = np.zeros((n_classes, D))
        self.bias = np.zeros(n_classes)
        
        # Gradient descent
        for _ in range(self.max_iters):
            # Forward pass
            scores = np.dot(training_data, self.weights.T) + self.bias  # (N, C)
            probs = self.softmax(scores)  # (N, C)
            
            # Compute loss if needed (cross-entropy loss)
            # loss = -np.sum(y_onehot * np.log(probs + 1e-15)) / N
            
            # Compute gradients - for softmax with cross-entropy
            # The gradient simplifies to: (probs - y_onehot)
            error = probs - y_onehot  # (N, C)
            
            # Gradient for weights: X^T * error / N
            grad_w = np.dot(error.T, training_data) / N  # (C, D)
            
            # Gradient for bias: sum(error) / N 
            grad_b = np.sum(error, axis=0) / N  # (C,)

            # Update weights and bias
            self.weights -= self.lr * grad_w
            self.bias -= self.lr * grad_b
        
        # Return predicted labels for training data
        return self.predict(training_data)

    def predict(self, test_data):
        """
        Runs prediction on the test data.

        Arguments:
            test_data (array): test data of shape (N,D)
        Returns:
            pred_labels (array): labels of shape (N,)
        """
        # Compute scores
        scores = np.dot(test_data, self.weights.T) + self.bias  # (N, C)
        
        # Get class with highest probability
        pred_labels = np.argmax(scores, axis=1)
        
        return pred_labels