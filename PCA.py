import pandas as pd
import numpy as np

from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np


class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, x_data):
        self.mean = np.mean(x_data, axis=0)
        X = x_data - self.mean

        # covariance matrix shape (n x n or nfeatures x nfeatures)
        covariance_matrix = self._covariance_matrix(x_data)

        # eigenvector shape (nx1)
        eigenvectors, eigenvalues = np.linalg.eig(covariance_matrix)

        eigenvectors = eigenvectors.T

        # sorting eigenvalues [start:end:steps]
        # this basically reverse the list to be greatest to least
        idxs = np.argsort(eigenvalues)[::-1]

        # we are choosing the fit variance
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]

        self.components = eigenvectors[:self.n_components]

        return covariance_matrix
# So Covariance is used incalculating the direction and variance between two variables

    def _covariance_matrix(self, x_data):
        n_rows, n_features = x_data.shape
        covariance_matrix = np.zeros((n_features, n_features))
        for i in range(n_features):
            for j in range(n_features):
                # obtain i's mean
                i_data = x_data[:, i]
                i_mean = np.mean(i_data)
                j_data = x_data[:, j]
                j_mean = np.mean(j_data)

                covariance_matrix[i, j] = np.sum(
                    (i_data - i_mean) * (j_data - j_mean)) / (n_rows - 1)

        return covariance_matrix

    def transform(self, X):
        # projects Data
        X = X - self.mean
        return np.dot(X, self.components.T)
