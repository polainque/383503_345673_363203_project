import numpy as np
from src.utils import normalize_fn, append_bias_term, accuracy_fn, macrof1_fn, mse_fn

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
            
            if self.task_kind == "classification":
                neighbor_labels_int = neighbor_labels.astype(int)
                test_labels[i] = np.argmax(np.bincount(neighbor_labels_int))

        
        return test_labels
    
    @staticmethod
    def KFold_cross_validation_KNN(X, Y, K, k):
        """
            Perform K-fold cross-validation for k-NN.

            Arguments:
                X : (NxD) training data
                Y : (N,)   training labels
                K : number of folds
                k : the k-NN hyperparameter

            Returns:
                val_accs : mean validation accuracy for each k

                val_F1s  : mean validation for F1-score for each k
                
        """
        N = X.shape[0]
        idx = np.random.permutation(N)
        fold_size = N // K

        va_accs, va_f1s = [], []

        for fold in range(K):
            start = fold * fold_size
            end = (fold + 1) * fold_size if fold < K - 1 else N

            va_idx = idx[start:end]
            tr_idx = np.setdiff1d(idx, va_idx, assume_unique=True)

            X_tr, Y_tr = X[tr_idx], Y[tr_idx]
            X_va, Y_va = X[va_idx], Y[va_idx]

            model = KNN(k=k)
            model.fit(X_tr, Y_tr)

            pred_va = model.predict(X_va)
            va_accs.append(accuracy_fn(pred_va, Y_va))
            va_f1s.append(macrof1_fn(pred_va, Y_va))

        return (
            float(np.mean(va_accs)),
            float(np.mean(va_f1s))
        )


    @staticmethod 
    def run_cv_for_hyperparam(X, Y, K, k_range):
        """
        Run K-fold cross-validation for each k in k_range.

            Arguments:
                X  : (NxD) training data
                Y  : (N,)  training labels
                K  : number of folds
                k_range : list or array of k values to try

            Returns:
                va_acc : mean validation accuracy over the K folds
                va_f1  : mean validation F1 score over the K folds
        """
        test_accs, test_F1s = [], []

        for k in k_range:
            test_acc, test_f1 = KNN.KFold_cross_validation_KNN(X, Y, K, k)

            test_accs.append(test_acc)
            test_F1s.append(test_f1)

        return (
            np.array(test_accs),
            np.array(test_F1s)
        )