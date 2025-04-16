import numpy as np

from ..utils import get_n_classes, label_to_onehot, onehot_to_label


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

    def sigmoid(self, z):
        """
        Compute the sigmoid function.
        
        Arguments:
            z (array): Input to the sigmoid function
            
        Returns:
            sigmoid_z (array): Output of sigmoid function
        """
        # Clip z to prevent overflow in exp
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.

        Arguments:
            training_data (array): training data of shape (N,D)
            training_labels (array): regression target of shape (N,)
        Returns:
            pred_labels (array): target of shape (N,)
        """
        N, D = training_data.shape
        # Get number of classes and convert labels to one-hot encoding
        n_classes = get_n_classes(training_labels)
        y_onehot = label_to_onehot(training_labels, n_classes)
        
        # Initialize weights and bias
        self.weights = np.zeros((n_classes, D))
        self.bias = np.zeros(n_classes)
        
        # Gradient descent
        for _ in range(self.max_iters):
            # Forward pass - compute scores for all classes
            scores = np.dot(training_data, self.weights.T) + self.bias
            
            # Compute probabilities using softmax
            # Subtract max for numerical stability
            exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
            probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
            
            # Compute gradients
            # For each class, compute gradient and update weights
            for c in range(n_classes):
                # Error for current class
                error = probs[:, c] - y_onehot[:, c]
                
                # Gradient for weights
                grad_w = np.dot(error, training_data) / N
                # Gradient for bias
                grad_b = np.sum(error) / N
                
                # Update weights and bias
                self.weights[c] -= self.lr * grad_w
                self.bias[c] -= self.lr * grad_b
        
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
        scores = np.dot(test_data, self.weights.T) + self.bias
        
        # Get the class with the highest score
        pred_onehot = np.zeros_like(scores)
        pred_onehot[np.arange(len(test_data)), np.argmax(scores, axis=1)] = 1
        
        # Convert one-hot encoded predictions back to labels
        n_classes = self.weights.shape[0]
        pred_labels = onehot_to_label(pred_onehot)
        
        return pred_labels
