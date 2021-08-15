"""
Jake Stephens
Class: CS 677 - Summer 2
Date: 8/17/2021
Homework #6
Description: This program analyzes seed data.
"""

import collections
import pandas as pd
import os
import math
import ast
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from create_table import create_table
from sklearn.cluster import KMeans
import random


# ======================
# Main program execution
# ======================

cols = [
    "area",
    "perimeter",
    "compactness",
    "length_of_kernel",
    "width_of_kernel",
    "asymmetry_coefficient",
    "lenth_of_kernel_groove",
    "class",
]
# Reading csv into dataframe
df_original = pd.read_csv(
    "data/seeds_dataset.txt",
    sep=r"\s+",
    names=cols,
)

df = df_original[df_original["class"] > 1]

# ======================
# Question #1
# ======================

# Splitting the data into X and Y
scaler = StandardScaler()
y = df["class"].values.tolist()
x = df.drop(["class"], axis=1)
scaler.fit(x)
x = scaler.transform(x)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.5, random_state=11, shuffle=True
)

# SVM Linear Kernel
svm_linear = svm.SVC(kernel="linear")
svm_linear.fit(x_train, y_train)
y_predict = svm_linear.predict(x_test)
accuracy = metrics.accuracy_score(y_predict.tolist(), y_test)
svm_linear_table = create_table(y_test, y_predict, accuracy)
print("\nSVM with Linear Kernel")
print("Accuracy: " + str(accuracy))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_predict))

# SVM Gaussian Kernel
svm_gaussian = svm.SVC(kernel="rbf")
svm_gaussian.fit(x_train, y_train)
y_predict = svm_gaussian.predict(x_test)
accuracy = metrics.accuracy_score(y_predict.tolist(), y_test)
svm_gaussian_table = create_table(y_test, y_predict, accuracy)
print("\nSVM with Gaussian Kernel")
print("Accuracy: " + str(accuracy))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_predict))

# SVM Polynomial Kernel
svm_poly = svm.SVC(kernel="poly")
svm_poly.fit(x_train, y_train)
y_predict = svm_poly.predict(x_test)
accuracy = metrics.accuracy_score(y_predict.tolist(), y_test)
svm_poly_table = create_table(y_test, y_predict, accuracy)
print("\nSVM with Polynomial Kernel")
print("Accuracy: " + str(accuracy))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_predict))

# ======================
# Question #2
# ======================

# KNN Classifier
knn = make_pipeline(
    StandardScaler(), KNeighborsClassifier(n_neighbors=7, weights="distance")
)
knn.fit(x_train, y_train)
y_predict = knn.predict(x_test)
accuracy = metrics.accuracy_score(y_predict, y_test)
knn_table = create_table(y_test, y_predict, accuracy)
print("\nKNN Classifier")
print("Accuracy: " + str(accuracy))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_predict))

table = pd.DataFrame(
    {
        "tp": [
            svm_linear_table[0],
            svm_gaussian_table[0],
            svm_poly_table[0],
            knn_table[0],
        ],
        "fp": [
            svm_linear_table[1],
            svm_gaussian_table[1],
            svm_poly_table[1],
            knn_table[1],
        ],
        "tn": [
            svm_linear_table[2],
            svm_gaussian_table[2],
            svm_poly_table[2],
            knn_table[2],
        ],
        "fn": [
            svm_linear_table[3],
            svm_gaussian_table[3],
            svm_poly_table[3],
            knn_table[3],
        ],
        "accuracy": [
            svm_linear_table[4],
            svm_gaussian_table[4],
            svm_poly_table[4],
            knn_table[4],
        ],
        "tpr": [
            svm_linear_table[5],
            svm_gaussian_table[5],
            svm_poly_table[5],
            knn_table[5],
        ],
        "tnr": [
            svm_linear_table[6],
            svm_gaussian_table[6],
            svm_poly_table[6],
            knn_table[6],
        ],
    }
)
print()
print(table)

# ======================
# Question #3
# ======================

# Clustering
num_clusters = [1, 2, 3, 4, 5, 6, 7, 8]
inertia_list = []
for k in num_clusters:
    kmeans_classifier = KMeans(n_clusters=k)
    y_means = kmeans_classifier.fit_predict(df_original)
    inertia = kmeans_classifier.inertia_
    inertia_list.append(inertia)

plt.plot(inertia_list, marker="o", color="green")
# plt.show()

# Pick 2 random features:
cols.remove("class")
random_cols = random.sample(set(cols), 2)
df_random_cols = df_original[random_cols]

# Run clustering for k=2
kmeans_classifier = KMeans(n_clusters=3)
y_means = kmeans_classifier.fit_predict(df_random_cols)
centroids = kmeans_classifier.cluster_centers_

plt.clf()

for i in range(0, len(y_means)):
    if y_means[i] == 0:
        plt.scatter(df_random_cols.loc[i][random_cols[0]], df_random_cols.loc[i][random_cols[1]], c="red")
    elif y_means[i] == 1:
        plt.scatter(df_random_cols.loc[i][random_cols[0]], df_random_cols.loc[i][random_cols[1]], c="blue")
    else:
        plt.scatter(df_random_cols.loc[i][random_cols[0]], df_random_cols.loc[i][random_cols[1]], c="green")

plt.scatter(centroids[0][0], centroids[0][1],  c="black", label="Centroids")
plt.scatter(centroids[1][0], centroids[1][1],  c="black")
plt.scatter(centroids[2][0], centroids[2][1],  c="black")
x_label = random_cols[0]
y_label = random_cols[1]
plt.legend()
plt.xlabel(x_label)
plt.ylabel(y_label)
plt.tight_layout()
plt.show()
