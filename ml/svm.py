from __future__ import annotations

import numpy as np

from sklearn.datasets import make_blobs

import matplotlib.pyplot as plt


class SVM:
    def __init__(self, c):
        self.c = c
        self.EPS = np.finfo(float).eps

    def fit(self, X: np.ndarray, T: np.ndarray) -> SVM:
        N = T.shape[0]
        D = X.shape[1]

        self.alpha = np.zeros(N)
        self.w = np.zeros(D)
        self.b = 0

        for _ in range(500):
            for i in range(N):
                x2 = X[i]
                y2 = self.func(x2)
                a2_old = self.alpha[i]
                t2 = T[i]

                is_a2_support = not np.isclose(a2_old, 0)
                if not (
                        (not is_a2_support and not t2 * y2 >= 1)
                        or (is_a2_support and not np.isclose(t2 * y2, 1))
                ):
                    continue

                E2 = y2 - t2

                Emax = 0.

                J = 0 if i != 0 else 1

                for j in range(N):
                    if j == i:
                        continue

                    E1 = self.func(X[j]) - T[j]

                    Ediff = abs(E1 - E2)

                    if Ediff >= Emax:
                        Emax = Ediff
                        J = j

                # print('selected: ', i, J)
                x1 = X[J]
                t1 = T[J]
                E1 = self.func(x1) - t1
                a1_old = self.alpha[J]

                L, H = self.calc_clip(t1, t2, a1_old, a2_old)

                kernels = x1.dot(x1) - 2 * x1.dot(x2) + x2.dot(x2)
                a2_unclip = (kernels * a2_old + t2 * (E1 - E2)) / kernels

                # print('Energy: ', E1 - E2)
                # print('L: ', L)
                # print('H: ', H)
                # print('a2_unclip: ', a2_unclip)
                a2_new = min(max(a2_unclip, L), H)

                a1_new = a1_old + t1 * t2 * (a2_old - a2_new)

                self.alpha[i] = a2_new
                self.alpha[J] = a1_new

                self.w = ((self.alpha * T).reshape((N, 1)) * X).sum(axis=0)

                b = 0.
                n_non_support = 0
                for k in range(N):
                    if not np.isclose(self.alpha[k], 0):
                        b += T[k] - self.w.dot(X[k])
                        n_non_support += 1

                if n_non_support != 0:
                    self.b = b / n_non_support

                # print(self.alpha)
                # print(self.w)
                # print(self.b)

        print(self.alpha)
        print(self.w)
        print(self.b)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        N = X.shape[0]
        T = np.zeros(N)

        for i in range(N):
            y = self.func(X[i])
            T[i] = 1 if y >= 0 else -1

        return T

    def func(self, x: np.ndarray) -> float:
        return self.w.dot(x) + self.b

    def calc_clip(self, t1: float, t2: float, a1_old: np.ndarray, a2_old: np.ndarray) -> tuple:
        if t1 == t2:
            L = 0
            H = a1_old + a2_old
        else:
            L = max(a2_old - a1_old, 0)
            H = np.inf

        return L, H


def main():
    X, T = make_blobs(n_samples=100, centers=2, n_features=2)
    T[T == 0] = -1

    plt.figure(figsize=(8, 7))
    # plt.scatter(X[:, 0], X[:, 1], marker='o', c=T, s=25, edgecolor='k')
    # plt.show()

    svm = SVM(1)

    # X = np.array([[1], [1.5], [2], [4], [4.8], [5]])
    # T = np.array([1, 1, 1, -1, -1, -1])

    svm.fit(X, T)
    plt.scatter(X[:, 0], X[:, 1], marker='o', c=svm.predict(X), s=25, edgecolor='k')

    Xl = np.linspace(X[:, 0].min(), X[:, 0].max(), 10)
    plt.plot(Xl, - (svm.w[0] * Xl + svm.b) / svm.w[1])

    plt.show()


if __name__ == '__main__':
    main()
