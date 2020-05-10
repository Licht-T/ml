from __future__ import annotations
import collections

import numpy as np
from sklearn.datasets import load_iris
from scipy.spatial import distance
from scipy.cluster.hierarchy import dendrogram
import scipy

import matplotlib.pyplot as plt


class Ward:
    def __init__(self):
        pass

    def fit(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        active_stack = collections.deque(range(n))
        chain_stack = collections.deque(maxlen=n)
        D = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(X))
        np.fill_diagonal(D, np.inf)
        size = np.ones(n, dtype=np.int)

        Z = np.empty((n-1, 4))

        z = 0
        while z < n - 1:
            if len(chain_stack) == 0:
                chain_stack.append(active_stack.pop())

            i = chain_stack.pop()
            j = np.argmin(D[i, ])

            if j not in chain_stack:
                if j in active_stack:
                    active_stack.remove(j)
                chain_stack.append(i)
                chain_stack.append(j)

                continue

            chain_stack.remove(j)

            si = size[i]
            sj = size[j]

            Dij = D[i, j]

            Z[z, 0] = i
            Z[z, 1] = j
            Z[z, 2] = Dij
            Z[z, 3] = si + sj
            z += 1

            size[i] = si + sj
            size[j] = 0
            active_stack.append(i)

            for k in range(n):
                sk = size[k]

                Dik = D[i, k]
                Djk = D[j, k]

                if sk == 0 or k == i:
                    continue

                d_new = ((si + sk) * Dik + (sj + sk) * Djk - sk * Dij) / (si + sj + sk)

                D[k, i] = d_new
                D[i, k] = d_new

            D[:, j] = np.inf
            D[j, :] = np.inf

        Z = Z[np.argsort(Z[:, 2], kind='mergesort')]
        cluster_dict = {}
        for i in range(Z.shape[0]):
            z0 = Z[i, 0]
            z1 = Z[i, 1]

            if z0 in cluster_dict:
                Z[i, 0] = cluster_dict[z0]
            if z1 in cluster_dict:
                Z[i, 1] = cluster_dict[z1]

            cluster_dict[z0] = n + i

        return Z


def main():
    iris = load_iris()
    X = iris.data

    ward = Ward()
    Z = ward.fit(X)
    print(Z)
    Z[:, 2] = np.log10(Z[:, 2] + 1)

    plt.figure(figsize=(8, 8))
    dendrogram(Z)
    plt.show()


if __name__ == '__main__':
    main()
