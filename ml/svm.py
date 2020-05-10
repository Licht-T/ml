from __future__ import annotations

import numpy as np

from sklearn.datasets import make_blobs

import matplotlib.pyplot as plt


class SVM:
    def __init__(self, c):
        self.c = c

    def fit(self, X: np.ndarray, T: np.ndarray) -> SVM:
        N = T.shape[0]
        D = X.shape[1]
        XX = np.inner(X, X)

        self.alpha = np.zeros(N)
        self.w = np.zeros(D)
        self.b = 0

        self.kernel = np.empty(N)

        np.seterr(all='raise')

        for _ in range(500):
            self.kernel[:] = np.nan

            for i in range(N):
                x2x2 = XX[i, i]
                y2 = self.classifier_function(X[i], i)
                a2_old = self.alpha[i]
                t2 = T[i]
                t2y2 = t2 * y2

                is_a2_support = not np.isclose(a2_old, 0)
                is_a2_bounded = np.isclose(a2_old, self.c)
                if not (
                        (not is_a2_support and not t2y2 >= 1)
                        or (is_a2_support and not is_a2_bounded and not np.isclose(t2y2, 1))
                ) or is_a2_bounded:
                    continue

                E2 = np.nan_to_num(y2 - t2, neginf=-1E100, posinf=1E100)

                E_diff_max = 0.

                J = 0 if i != 0 else 1

                for j in range(N):
                    if j == i:
                        continue

                    y1 = self.classifier_function(X[j], j)

                    E1 = np.nan_to_num(y1 - T[j], neginf=-1E100, posinf=1E100)

                    E_diff = E1 - E2

                    if abs(E_diff) >= abs(E_diff_max):
                        E_diff_max = E_diff
                        J = j

                x1x1 = XX[J, J]
                x1x2 = XX[J, i]
                t1 = T[J]
                a1_old = self.alpha[J]

                L, H = self.calc_clip(t1, t2, a1_old, a2_old)

                kernels = x1x1 - 2 * x1x2 + x2x2
                a2_unclip = a2_old + (t2 * E_diff_max) / kernels

                a2_new = min(max(a2_unclip, L), H)

                a1_new = a1_old + t1 * t2 * (a2_old - a2_new)

                self.alpha[i] = a2_new
                self.alpha[J] = a1_new
                self.w = ((self.alpha * T).reshape((N, 1)) * X).sum(axis=0)
                self.kernel[:] = np.nan

                b = 0.
                n_non_support = 0

                for j in range(N):
                    aj = self.alpha[j]
                    kj = self.w.dot(X[j])
                    self.kernel[j] = kj

                    if not np.isclose(aj, 0) and not np.isclose(aj, self.c):
                        n_non_support += 1
                        b += (T[j] - kj - b) / n_non_support
                        b = np.nan_to_num(b, neginf=-1E100, posinf=1E100)

                if n_non_support != 0:
                    self.b = b

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        N = X.shape[0]
        T = np.zeros(N)

        for i in range(N):
            y = self.classifier_function(X[i])
            T[i] = 1 if y >= 0 else -1

        return T

    def classifier_function(self, x: np.ndarray, i: int=None) -> float:
        if i is None:
            kernel = self.w.dot(x)
        else:
            kernel = self.kernel[i]

            if np.isnan(kernel):
                kernel = self.w.dot(x)
                self.kernel[i] = kernel

        return kernel + self.b

    def calc_clip(self, t1: float, t2: float, a1_old: np.ndarray, a2_old: np.ndarray) -> tuple:
        if t1 == t2:
            L = max(a1_old + a2_old - self.c, 0)
            H = max(a1_old + a2_old, self.c)
        else:
            L = max(a2_old - a1_old, 0)
            H = min(self.c - a2_old + a1_old, self.c)

        return L, H


def main():
    X, T = make_blobs(n_samples=100, centers=2, n_features=2)
    T[T == 0] = -1

    plt.figure(figsize=(8, 8))

    plt.scatter(X[:, 0], X[:, 1], marker='o', c=T, s=25, edgecolor='k')

    svm = SVM(10)

    svm.fit(X, T)

    Xl = np.linspace(X[:, 0].min(), X[:, 0].max(), 10)
    plt.plot(Xl, - (svm.w[0] * Xl + svm.b) / svm.w[1])

    plt.show()


if __name__ == '__main__':
    main()
