# Threat-Intelligence-project
Threat Intelligence project code to run a machine learning algorithm on it, and then print the accuracy on the screen

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# Load the Iris dataset
iris = load_iris()

# Set the number of folds
k = 10

# Decision Tree model
decision_tree = DecisionTreeClassifier()

# Perform k-fold cross-validation for Decision Tree
decision_tree_scores = cross_val_score(decision_tree, iris.data, iris.target, cv=k)

# Print Decision Tree scores
print("Decision Tree Scores:")
for fold_idx, score in enumerate(decision_tree_scores):
    print("Fold", fold_idx+1, "Accuracy:", score)
print("Mean Accuracy:", np.mean(decision_tree_scores))
print("Standard Deviation:", np.std(decision_tree_scores))
print()

# SVM model
svm = SVC()

# Perform k-fold cross-validation for SVM
svm_scores = cross_val_score(svm, iris.data, iris.target, cv=k)

# Print SVM scores
print("SVM Scores:")
for fold_idx, score in enumerate(svm_scores):
    print("Fold", fold_idx+1, "Accuracy:", score)
print("Mean Accuracy:", np.mean(svm_scores))
print("Standard Deviation:", np.std(svm_scores))
print()

# Neural Network model
nn = MLPClassifier()

# Perform k-fold cross-validation for Neural Network
nn_scores = cross_val_score(nn, iris.data, iris.target, cv=k)

# Print Neural Network scores
print("Neural Network Scores:")
for fold_idx, score in enumerate(nn_scores):
    print("Fold", fold_idx+1, "Accuracy:", score)
print("Mean Accuracy:", np.mean(nn_scores))
print("Standard Deviation:", np.std(nn_scores))
