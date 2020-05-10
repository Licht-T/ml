from __future__ import annotations
from typing import Tuple

import numpy as np
import scipy
import scipy.special

from sklearn.datasets import make_blobs

import matplotlib.pyplot as plt


class LogisticRegression:
    def __init__(self, n_mini_batch: int = 100, learning_rate: float = 0.6):
        self.learning_rate = learning_rate
        self.n_mini_batch = n_mini_batch
        pass

    def fit(self, X: np.ndarray, T: np.ndarray):
        self.label_list = list(set(T))
        n_labels = len(self.label_list)
        n_X = X.shape[0]
        n_features = X.shape[1]

        T_vectors = np.zeros((n_X, n_labels))
        T_vectors[range(n_X), T] = 1

        self.W = np.ones((n_features, n_labels))
        self.b = np.ones(n_labels)

        for _ in range(100):
            for i in range(0, n_X, self.n_mini_batch):
                i_next = i + self.n_mini_batch
                X_batch = X[i:i_next, :]
                T_vectors_batch = T_vectors[i:i_next]

                dW, db = self.gradient(X_batch, T_vectors_batch)
                self.W -= self.learning_rate * dW
                self.b -= self.learning_rate * db

    def softmax(self, X: np.ndarray) -> np.ndarray:
        return scipy.special.softmax(X @ self.W + self.b, axis=1)

    def gradient(self, X: np.ndarray, T_vectors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n_X = X.shape[0]

        Pi = self.softmax(X)
        X_weight = Pi - T_vectors

        dW = X.T.dot(X_weight) / n_X
        db = X_weight.sum(axis=0) / n_X

        return dW, db

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.softmax(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.softmax(X).argmax(axis=1)


def main():
    X, T = make_blobs(n_samples=1000, n_features=2, centers=5)

    plt.figure(figsize=(8, 8))

    logistic = LogisticRegression()

    logistic.fit(X, T)

    plt.scatter(X[:, 0], X[:, 1], marker='o', c=logistic.predict(X), s=100, edgecolor='k')
    plt.scatter(X[:, 0], X[:, 1], marker='^', c=T, s=50, edgecolor='k')

    plt.show()


if __name__ == '__main__':
    main()
