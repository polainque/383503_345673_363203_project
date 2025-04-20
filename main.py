import argparse

import numpy as np
import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt

import seaborn as sns
import time
import os

from src.data import load_data
from src.methods.dummy_methods import DummyClassifier
from src.methods.logistic_regression import LogisticRegression
from src.methods.knn import KNN
from src.methods.kmeans import KMeans
from src.utils import normalize_fn, append_bias_term, accuracy_fn, macrof1_fn, mse_fn

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
    s1_train = time.time()
    print(f"\nTraining {args.method}...")
    preds_train = method_obj.fit(xtrain_normalized, ytrain)
    s2_train = time.time()
    print(f"\nMethod {args.method} takes {s2_train-s1_train} seconds to train.")

    # Predict on unseen data
    s1_pred = time.time()
    print("\nEvaluating on test set...")
    preds = method_obj.predict(xtest_normalized)
    s2_pred = time.time()
    print(f"\nMethod {args.method} takes {s2_pred-s1_pred} seconds to predict.")

    # Report results: performance on train and valid/test sets
    acc = accuracy_fn(preds_train, ytrain)
    macrof1 = macrof1_fn(preds_train, ytrain)
    print(f"\nTrain set: accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")

    acc = accuracy_fn(preds, ytest)
    macrof1 = macrof1_fn(preds, ytest)
    print(f"Test set:  accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")

    if args.method == "logistic_regression" and not args.test:
        print("\nHyperparameter search for logistic regression:")
        best_acc = 0
        best_lr = args.lr
        best_max_iters = args.max_iters
        
        # We try different learning rates
        learning_rates = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1]
        max_iters_options = [10, 30, 35, 40, 45, 50, 100, 200, 300, 400, 500, 1000]
        
        # Create dictionaries to store results
        results = {max_iter: [] for max_iter in max_iters_options}
        all_accuracies = []
        
        for max_iters in max_iters_options:
            accuracies = []
            
            for lr in learning_rates:
                # Initialize and train model
                lr_model = LogisticRegression(lr=lr, max_iters=max_iters)
                lr_model.fit(xtrain_normalized, ytrain)
                
                # Evaluate on validation set
                val_preds = lr_model.predict(xtest_normalized)
                val_acc = accuracy_fn(val_preds, ytest)
                
                # Store accuracy for this combination
                accuracies.append(val_acc)
                all_accuracies.append((lr, max_iters, val_acc))
                
                # Update best parameters if better
                if val_acc > best_acc:
                    best_acc = val_acc
                    best_lr = lr
                    best_max_iters = max_iters
            
            # Store accuracies for this max_iters value
            results[max_iters] = accuracies
        
        # Create directory for plots if it doesn't exist
        os.makedirs('plots', exist_ok=True)

        print(f"Best hyperparameters: lr={best_lr:.1e}, max_iters={best_max_iters}")
        print(f"Best validation accuracy: {best_acc:.3f}%")

        # Extract and sort unique learning rates and max_iters
        lrs = sorted(set(lr for lr, _, _ in all_accuracies))

        max_iters_list = sorted(set(it for _, it, _ in all_accuracies))

        # Create accuracy matrix (rows = max_iters, cols = learning_rates)
        accuracy_matrix = np.zeros((len(max_iters_list), len(lrs)))

        # Fill accuracy matrix
        for i, max_iter in enumerate(max_iters_list):
            for j, lr in enumerate(lrs):
                for acc_lr, acc_iter, acc in all_accuracies:
                    if acc_lr == lr and acc_iter == max_iter:
                        accuracy_matrix[i, j] = acc

        plt.figure(figsize=(14, 8))
        sns.heatmap(
            accuracy_matrix,
            xticklabels=[f"{lr:.0e}" for lr in lrs],
            yticklabels=max_iters_list,
            annot=True, fmt=".2f", cmap="crest", cbar_kws={'label': 'Validation Accuracy %'}
        )
        plt.xlabel("Learning Rate")
        plt.ylabel("Max Iterations")
        plt.title("Validation Accuracy Heatmap (Logistic Regression)")
        plt.tight_layout()
        plt.savefig("plots/logistic_regression_hyperparam_heatmap.png")
        plt.show()

    if args.method == "knn" and not args.test:

        kmax = 40 if args.K < 40 else args.K
        K = 5
        k_range = np.arange(1, kmax+1)

        # Cross Validation
        t0 = time.time()
        cv_val_accs, cv_val_f1s = KNN.run_cv_for_hyperparam(xtrain_normalized, ytrain, K, k_range)
        print(f"\n5-Fold Cross Validation on {kmax} values of k in {time.time()-t0:.2f}s")

        # pick best k by validation accuracy
        best_idx = np.argmax(cv_val_accs)
        best_k   = k_range[best_idx]
        print(f"\nBest k by 5-Fold Cross Validation : {best_k} \n---> Validation accuracy = {cv_val_accs[best_idx]:.3f}\n---> F1-score = {cv_val_f1s[best_idx]:.3f}")

        # Recomputing train performance on entire training data for each k
        full_tr_accs = np.zeros_like(cv_val_accs)
        full_tr_f1s  = np.zeros_like(cv_val_f1s)
        for i, k in enumerate(k_range):
            m = KNN(k=k)
            pred = m.fit(xtrain_normalized, ytrain)
            full_tr_accs[i] = accuracy_fn(pred, ytrain)
            full_tr_f1s [i] = macrof1_fn(pred, ytrain)

        # Final model on all training data to test model on test data
        final = KNN(k=best_k)
        pred_train = final.fit(xtrain_normalized, ytrain)
        pred_test = final.predict(xtest_normalized)

        tr_acc, tr_f1 = accuracy_fn(pred_train, ytrain), macrof1_fn(pred_train, ytrain)
        te_acc, te_f1 = accuracy_fn(pred_test, ytest),  macrof1_fn(pred_test,  ytest)
        print(f"\nFINAL [k={best_k}] : TRAIN accuracy = {tr_acc:.3f}%, F1-score = {tr_f1:.6f}")
        print(f"FINAL [k={best_k}] : TEST  accuracy = {te_acc:.3f}%, F1-score = {te_f1:.6f}\n")

        # CV and full‐train plot
        plt.figure(figsize=(8,6))
        plt.plot(k_range, cv_val_accs ,color = 'r', marker = '.',label="CV-validation Accuracy")
        plt.plot(k_range, cv_val_f1s *100 ,color = 'r', linestyle = '--', marker = '.', label="CV-validation F1-score")
        plt.plot(k_range, full_tr_accs , color = 'b',marker = '.', label="Full-train Accuracy")
        plt.plot(k_range, full_tr_f1s * 100 , color = 'b', linestyle = '--',marker = '.',label="Full-train F1-score")
        plt.axvline(best_k, color='k', linestyle =':', label=f"Best k = {best_k}")
        plt.xlabel("k"); plt.ylabel("Score (%)")
        plt.title("k-NN: CV vs Full-train Accuracy & F1-score over k")
        plt.legend(); plt.grid(True)
        plt.savefig("knn_train_test_curves.png")
        plt.close()

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
                #print(f"  Testing K={k}, max_iters={max_iters}...")
                
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
                
                #print(f"    Validation accuracy: {val_acc:.3f}% - F1-score: {val_f1:.6f}")
                #print(f"    Training time: {train_time:.4f}s - Prediction time: {pred_time:.4f}s")
                
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
        plt.show()  # Display the plot

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
        plt.show()  # Display the plot

        
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
        plt.show()  # Display the plot
            
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
        "--data_type", default="features", type=str, help="features/original(MS2)/heart_disease"
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