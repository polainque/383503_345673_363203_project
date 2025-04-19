import numpy as np
import matplotlib.pyplot as plt


from ..utils import get_n_classes, label_to_onehot, onehot_to_label, accuracy_fn


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




def create_simple_visualizations(xtrain, ytrain, xtest, ytest, args):
    """
    Create simple visualizations for logistic regression performance with different hyperparameters.
    
    Arguments:
        xtrain, ytrain: Training data and labels
        xtest, ytest: Test data and labels
        args: Command line arguments
    """
    # Create a directory for saving plots if it doesn't exist
    import os
    if not os.path.exists('plots'):
        os.makedirs('plots')
    
    print("\nRunning hyperparameter search for visualization...")
    
    # 1. Learning rate vs. accuracy (with fixed iterations)
    learning_rates = [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
    max_iters = 500  # Fixed iterations
    
    train_accuracies = []
    test_accuracies = []
    
    for lr in learning_rates:
        print(f"Testing learning rate: {lr} with {max_iters} iterations")
        model = LogisticRegression(lr=lr, max_iters=max_iters)
        train_preds = model.fit(xtrain, ytrain)
        test_preds = model.predict(xtest)
        
        train_acc = accuracy_fn(train_preds, ytrain)
        test_acc = accuracy_fn(test_preds, ytest)
        
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
    
    # Plot learning rate vs. accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(np.log10(learning_rates), train_accuracies, 'o-', label='Training Accuracy')
    plt.plot(np.log10(learning_rates), test_accuracies, 's-', label='Test Accuracy')
    plt.title(f'Accuracy vs. Learning Rate (Max Iterations = {max_iters})')
    plt.xlabel('Learning Rate (log10)')
    plt.ylabel('Accuracy (%)')
    plt.xticks(np.log10(learning_rates), [f'{lr:.1e}' for lr in learning_rates], rotation=45)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/accuracy_vs_lr.png')
    plt.close()
    
    # 2. Iterations vs. accuracy (with best learning rate)
    best_lr_idx = np.argmax(test_accuracies)
    best_lr = learning_rates[best_lr_idx]
    print(f"Best learning rate found: {best_lr}")
    
    iterations_list = [10, 50, 100, 200, 300, 500, 750, 1000]
    train_accuracies_iter = []
    test_accuracies_iter = []
    
    for iters in iterations_list:
        print(f"Testing iterations: {iters} with learning rate {best_lr}")
        model = LogisticRegression(lr=best_lr, max_iters=iters)
        train_preds = model.fit(xtrain, ytrain)
        test_preds = model.predict(xtest)
        
        train_acc = accuracy_fn(train_preds, ytrain)
        test_acc = accuracy_fn(test_preds, ytest)
        
        train_accuracies_iter.append(train_acc)
        test_accuracies_iter.append(test_acc)
    
    # Plot iterations vs. accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(iterations_list, train_accuracies_iter, 'o-', label='Training Accuracy')
    plt.plot(iterations_list, test_accuracies_iter, 's-', label='Test Accuracy')
    plt.title(f'Accuracy vs. Iterations (Learning Rate = {best_lr:.1e})')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Accuracy (%)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/accuracy_vs_iterations.png')
    plt.close()
    
    # 3. Grid search visualization (if requested)
    if args.grid_search:
        iterations_grid = [100, 300, 500, 800]
        lr_grid = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
        
        results = np.zeros((len(iterations_grid), len(lr_grid)))
        
        for i, iters in enumerate(iterations_grid):
            for j, lr in enumerate(lr_grid):
                print(f"Grid search: iterations={iters}, lr={lr}")
                model = LogisticRegression(lr=lr, max_iters=iters)
                model.fit(xtrain, ytrain)
                test_preds = model.predict(xtest)
                test_acc = accuracy_fn(test_preds, ytest)
                results[i, j] = test_acc
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        plt.imshow(results, interpolation='nearest', cmap='viridis')
        plt.colorbar(label='Test Accuracy (%)')
        plt.title('Test Accuracy for different Learning Rates and Iterations')
        plt.xlabel('Learning Rate')
        plt.ylabel('Max Iterations')
        plt.xticks(np.arange(len(lr_grid)), [f'{lr:.1e}' for lr in lr_grid])
        plt.yticks(np.arange(len(iterations_grid)), iterations_grid)
        
        # Add text annotations
        for i in range(len(iterations_grid)):
            for j in range(len(lr_grid)):
                plt.text(j, i, f'{results[i, j]:.1f}', 
                        ha="center", va="center", 
                        color="white" if results[i, j] > np.mean(results) else "black")
        
        plt.tight_layout()
        plt.savefig('plots/grid_search.png')
        plt.close()
    
    # Find and return best hyperparameters
    best_lr = learning_rates[np.argmax(test_accuracies)]
    best_iters = iterations_list[np.argmax(test_accuracies_iter)]
    
    return best_lr, best_iters