import numpy as np
import pandas as pd
from collections import Counter

# Classes is the blue print for creating objects
# Defines attributes (data) and methods (functions)

# Object is an instance of the class

# Methods - Funcitons to operate on attributes

# __init__ is an intializer that is called when the object is created which is used to initialize the attributes

# self refers to the individual node - current instance of the class - allowing to access attributes and methods

# helper functinos - indicated via this underscore _helperfunction(self,...):


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None


class DecisionTreeClassifier:
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None):

        # These are the hyperparameters/stopping points
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        # a way to add randomness to tree # useful for making random forests
        self.n_features = n_features
        self.root = None  # access to root in order to obtain IG later

    # Intial Step of passing data to model
    def fit(self, X, y):
        self.n_features = X.shape[1] if not self.n_features else min(
            X.shape[1], self.n_features)
        print(self.n_features)
        self.root = self._grow_tree(X, y)

    # Will recursively built tree until we reach a STOPPING POINT
    def _grow_tree(self, X, y, depth=0):
        n_samples, n_feats = X.shape
        # Obtaining number of unique y values - for classificaiton
        n_labels = len(np.unique(y))

        # This is the STOPPING POINT - This checks if a 1) tree is too long, 2) if a leaf is pure or
        # 3) if min_samples_split can only occur if there are a specific # of samples
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # Syntax np.random.choice(array, size, replace, p)
        # This will basically create a random order on how
        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)

        # finding best split
        best_feature, best_thresh = self._best_split(X, y, feat_idxs)

        left_idx, right_idx = self._split(X[:, best_feature], best_thresh)

        # this is the recurisive part where it call itself until the
        left = self._grow_tree(X[left_idx, :], y[left_idx], depth + 1)
        right = self._grow_tree(X[right_idx, :], y[right_idx], depth + 1)
        return Node(best_feature, best_thresh, left, right)

    def _most_common_label(self, y):
        counter = Counter(y)
        value = counter.most_common(1)[0][0]
        # t = [0, 1, 0, 1, 1, 2]
        # counter = Counter(t)
        # Output: Counter({1: 3, 0: 2, 2: 1})
        # puts into a dictionary
        return value

    def _best_split(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_threshold = None, None

        for feat_idx in feat_idxs:  # Iterating through each ordered column
            X_column = X[:, feat_idx]
            # for each column we are getting the unique values of the thresholds
            thresholds = np.unique(X_column)

            # Threshold represents that unique values within a column (are we >=,== to a specific filter)
            for thr in thresholds:

                gain = self._information_gain(y, X_column, thr)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = thr

        return split_idx, split_threshold

    def _information_gain(self, y, Xcolumn, thresholds):
        # parent entropy
        parent_entropy = self._entropy(y)

        # create children
        left_idx, right_idx = self._split(Xcolumn, thresholds)

        if len(left_idx) == 0 or len(right_idx) == 0:
            return 0

        # calculate the weighter entropy of children
        n = len(y)
        n_l, n_r = len(left_idx), len(right_idx)
        e_l, e_r = self._entropy(y[left_idx]), self._entropy(y[right_idx])

        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r

        informaiton_gain = parent_entropy - child_entropy
        # calculate information gain
        return informaiton_gain

    def _split(self, Xcolumn, thresholds):
        left_idxs = np.argwhere(Xcolumn <= thresholds).flatten()
        right_idxs = np.argwhere(Xcolumn > thresholds).flatten()
        return left_idxs, right_idxs

    def _entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log(p) for p in ps if p > 0])

        # This measures purity if all nodes values are close to 1 (100%), the closer it is to zero
        # The closer the log is to zero the samller then entropy


# Prediction Component

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
