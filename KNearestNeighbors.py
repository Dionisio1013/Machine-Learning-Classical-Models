import pandas as pd
import numpy as np
from collections import Counter


class KNN:
    def __init__(self, k, Xreference=None, yreference=None):
        self.k = k
        self.Xreference = None
        self.yreference = None

    def fit(self, x, y):
        self.Xreference = x
        self.yreference = y

    def predict(self, X):
        predictions_array = []
        for idx in range(len(X)):
            euclidean_distances = self._euclidean_distances(
                self.Xreference, X[idx, :])

            top_idx = self._topk(euclidean_distances, self.k)

            predict = self._majority(top_idx, self.yreference)
            predictions_array.append(int(predict))

        return predictions_array

    def _euclidean_distances(self,  X, record):
        euclidean_distances = np.sqrt(np.sum((X - record)**2, axis=1))
        return euclidean_distances

    def _topk(self, euclidean_distance, k):
        # arg obtains the indices 
        # argsort does least to greatest (euclidean means least is closest)
        return np.argsort(euclidean_distance)[:k]

    def _majority(self, indces, y):
        counter = Counter(y[indces])
        value = counter.most_common(1)[0][0]
        return value
