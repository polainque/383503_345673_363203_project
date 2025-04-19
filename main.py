import argparse

import numpy as np
import matplotlib.pyplot as plt
import time

from src.data import load_data
from src.methods.dummy_methods import DummyClassifier
from src.methods.logistic_regression import LogisticRegression
from src.methods.knn import KNN
from src.methods.kmeans import KMeans
from src.utils import normalize_fn, append_bias_term, accuracy_fn, macrof1_fn, mse_fn
import os

np.random.seed(100)


def main(args):
    """
    The main function of the script. Do not hesitate to play with it
    and add your own code, visualization, prints, etc!

    Arguments:
        args (Namespace): arguments that were parsed from the command line (see at the end
                          of this file). Their value can be accessed as "args.argument".
    """
    ## 1. First, we load our data

    # EXTRACTED FEATURES DATASET
    if args.data_type == "features":
        feature_data = np.load("features.npz", allow_pickle=True)
        xtrain, xtest = feature_data["xtrain"], feature_data["xtest"]
        ytrain, ytest = feature_data["ytrain"], feature_data["ytest"]

    # ORIGINAL IMAGE DATASET (MS2)
    elif args.data_type == "original":
        data_dir = os.path.join(args.data_path, "dog-small-64")
        xtrain, xtest, ytrain, ytest = load_data(data_dir)

    ## 2. Then we must prepare it. This is where you can create a validation set, normalize, add bias, etc.
    # Make a validation set (it can overwrite xtest, ytest)
    if not args.test:
        # Split training data to create a validation set
        validation_ratio = 0.2  # Using 20% of training data for validation
        num_val_samples = int(len(xtrain) * validation_ratio)
        
        # Shuffle indices
        indices = np.random.permutation(len(xtrain))
        train_indices = indices[num_val_samples:]
        val_indices = indices[:num_val_samples]
        
        # Create validation set
        xval = xtrain[val_indices]
        yval = ytrain[val_indices]
        
        # Update training set
        xtrain = xtrain[train_indices]
        ytrain = ytrain[train_indices]
        
        # Use validation set for evaluation
        xtest = xval
        ytest = yval
        
        print(f"Training set size: {len(xtrain)}")
        print(f"Validation set size: {len(xval)}")
    else:
        print(f"Training set size: {len(xtrain)}")
        print(f"Test set size: {len(xtest)}")
    
    # Normalize features (important for logistic regression)

    # First, calculate the means and standard deviations
    means = np.mean(xtrain, axis=0, keepdims=True)
    stds = np.std(xtrain, axis=0, keepdims=True)
    stds[stds == 0] = 1  # Avoid division by zero

    # Then normalize the training data
    xtrain_normalized = normalize_fn(xtrain, means, stds)

    # And later normalize the test data using the same means and stds
    xtest_normalized = normalize_fn(xtest, means, stds)

    ## 3. Initialize the method you want to use.

    # Use NN (FOR MS2!)
    if args.method == "nn":
        raise NotImplementedError("This will be useful for MS2.")

    # Follow the "DummyClassifier" example for your methods
    if args.method == "dummy_classifier":
        method_obj = DummyClassifier(arg1=1, arg2=2)
    elif args.method == "logistic_regression":
        method_obj = LogisticRegression(lr=args.lr, max_iters=args.max_iters)
    elif args.method == "knn":
        method_obj = KNN(args.K)
    elif args.method == "kmeans":
        method_obj = KMeans(K=args.K, max_iters=args.max_iters)
    else:
        raise ValueError(f"Unknown method: {args.method}")

    ## 4. Train and evaluate the method
    # Fit (:=train) the method on the training data for classification task
    print(f"\nTraining {args.method}...")
    preds_train = method_obj.fit(xtrain_normalized, ytrain)

    # Predict on unseen data
    print("Evaluating on test set...")
    preds = method_obj.predict(xtest_normalized)

    # Report results: performance on train and valid/test sets
    acc = accuracy_fn(preds_train, ytrain)
    macrof1 = macrof1_fn(preds_train, ytrain)
    print(f"\nTrain set: accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")

    acc = accuracy_fn(preds, ytest)
    macrof1 = macrof1_fn(preds, ytest)
    print(f"Test set:  accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")

    if args.method == "logistic_regression" and not args.test:
        print("\nGenerating simple visualizations...")
        #best_lr, best_iters = create_simple_visualizations(xtrain, ytrain, xtest, ytest, args)
    
        print("\nHyperparameter search for logistic regression:")
        best_acc = 0
        best_lr = args.lr
        best_max_iters = args.max_iters
        
        # We try different learning rates
        learning_rates = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
        max_iters_options = [100, 300, 500, 1000]
        
        for lr in learning_rates:
            for max_iters in max_iters_options:
                print(f"  Testing lr={lr}, max_iters={max_iters}...")
                
                # Initialize and train model
                lr_model = LogisticRegression(lr=lr, max_iters=max_iters)
                lr_model.fit(xtrain_normalized, ytrain)
                
                # Evaluate on validation set
                val_preds = lr_model.predict(xtest_normalized)
                val_acc = accuracy_fn(val_preds, ytest)
                
                print(f"    Validation accuracy: {val_acc:.3f}%")
                
                # Update best parameters if better
                if val_acc > best_acc:
                    best_acc = val_acc
                    best_lr = lr
                    best_max_iters = max_iters
        
        print(f"\nBest hyperparameters: lr={best_lr}, max_iters={best_max_iters}")
        print(f"Best validation accuracy: {best_acc:.3f}%")
    ### WRITE YOUR CODE HERE if you want to add other outputs, visualization, etc. mettez d'autres trucs pour vos methodes les kheys avec un  if args.method == "votre methode" and not args.test:
    if args.method == "kmeans" and not args.test:
        print("\nHyperparameter search for KMeans:")
        best_acc = 0
        best_k = args.K
        best_max_iters = args.max_iters

        # Try different K values and max iterations
        k_values = [2, 3, 4, 5, 8, 10]
        max_iters_options = [100, 300, 500, 1000]

        # Store results for plotting
        results = []
        training_times = []
        prediction_times = []

        for k in k_values:
            for max_iters in max_iters_options:
                print(f"  Testing K={k}, max_iters={max_iters}...")
                
                # Initialize and train model
                kmeans_model = KMeans(K=k, max_iters=max_iters)
                
                # Measure training time
                start_time = time.time()
                kmeans_model.fit(xtrain_normalized, ytrain)
                end_time = time.time()
                train_time = end_time - start_time
                training_times.append((k, max_iters, train_time))
                
                # Measure prediction time
                start_time = time.time()
                val_preds = kmeans_model.predict(xtest_normalized)
                end_time = time.time()
                pred_time = end_time - start_time
                prediction_times.append((k, max_iters, pred_time))
                
                # Evaluate on validation set
                val_acc = accuracy_fn(val_preds, ytest)
                val_f1 = macrof1_fn(val_preds, ytest)
                
                # Store results for plotting
                results.append((k, max_iters, val_acc, val_f1))
                
                print(f"    Validation accuracy: {val_acc:.3f}% - F1-score: {val_f1:.6f}")
                print(f"    Training time: {train_time:.4f}s - Prediction time: {pred_time:.4f}s")
                
                # Update best parameters if better
                if val_acc > best_acc:
                    best_acc = val_acc
                    best_k = k
                    best_max_iters = max_iters

        print(f"\nBest hyperparameters: K={best_k}, max_iters={best_max_iters}")
        print(f"Best validation accuracy: {best_acc:.3f}%")

        # Create visualizations

        # 1. Plot accuracy vs. K for different max_iters
        plt.figure(figsize=(12, 8))
        for max_iter in max_iters_options:
            k_values_plot = []
            accuracies = []
            for k, max_iters, acc, _ in results:
                if max_iters == max_iter:
                    k_values_plot.append(k)
                    accuracies.append(acc)
            plt.plot(k_values_plot, accuracies, marker='o', label=f'max_iters={max_iter}')

        plt.title('Accuracy vs. Number of Clusters (K)')
        plt.xlabel('Number of Clusters (K)')
        plt.ylabel('Validation Accuracy (%)')
        plt.legend()
        plt.grid(True)
        plt.savefig('kmeans_accuracy_vs_k.png')
        #plt.show(block=False)

        # 2. Plot F1-score vs. K for different max_iters
        plt.figure(figsize=(12, 8))
        for max_iter in max_iters_options:
            k_values_plot = []
            f1_scores = []
            for k, max_iters, _, f1 in results:
                if max_iters == max_iter:
                    k_values_plot.append(k)
                    f1_scores.append(f1)
            plt.plot(k_values_plot, f1_scores, marker='o', label=f'max_iters={max_iter}')

        plt.title('F1-Score vs. Number of Clusters (K)')
        plt.xlabel('Number of Clusters (K)')
        plt.ylabel('F1-Score')
        plt.legend()
        plt.grid(True)
        plt.savefig('kmeans_f1_vs_k.png')
        #plt.show(block=False)

        
        # 3. Plot training time vs. K
        plt.figure(figsize=(12, 8))
        for max_iter in max_iters_options:
            k_values_plot = []
            times = []
            for k, max_iters, time_val in training_times:
                if max_iters == max_iter:
                    k_values_plot.append(k)
                    times.append(time_val)
            plt.plot(k_values_plot, times, marker='o', label=f'max_iters={max_iter}')

        plt.title('Training Time vs. Number of Clusters (K)')
        plt.xlabel('Number of Clusters (K)')
        plt.ylabel('Training Time (seconds)')
        plt.legend()
        plt.grid(True)
        plt.savefig('kmeans_training_time_vs_k.png')
        plt.show(block=False)
            
if __name__ == "__main__":
    # Definition of the arguments that can be given through the command line (terminal).
    # If an argument is not given, it will take its default value as defined below.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        default="dummy_classifier",
        type=str,
        help="dummy_classifier / knn / logistic_regression / kmeans / nn (MS2)",
    )
    parser.add_argument(
        "--data_path", default="data", type=str, help="path to your dataset"
    )
    parser.add_argument(
        "--data_type", default="features", type=str, help="features/original(MS2)"
    )
    parser.add_argument(
        "--K", type=int, default=1, help="number of neighboring datapoints used for knn"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="learning rate for methods with learning rate",
    )
    parser.add_argument(
        "--max_iters",
        type=int,
        default=100,
        help="max iters for methods which are iterative",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="train on whole training data and evaluate on the test data, otherwise use a validation set",
    )

    # Feel free to add more arguments here if you need!

    # MS2 arguments
    parser.add_argument(
        "--nn_type",
        default="cnn",
        help="which network to use, can be 'Transformer' or 'cnn'",
    )
    parser.add_argument(
        "--nn_batch_size", type=int, default=64, help="batch size for NN training"
    )

    # "args" will keep in memory the arguments and their values,
    # which can be accessed as "args.data", for example.
    args = parser.parse_args()
    main(args)
