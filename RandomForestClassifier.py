import numpy as np
import random
import pandas as pd
from collections import Counter
import math


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
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None, random_forest=False):

        # These are the hyperparameters/stopping points
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        # a way to add randomness to tree # useful for making random forests
        self.n_features = n_features
        self.root = None  # access to root in order to obtain IG later
        self.random_forest = random_forest

    # Intial Step of passing data to model
    def fit(self, X, y):
        self.n_features = X.shape[1] if not self.n_features else min(
            X.shape[1], self.n_features)
        print(self.n_features)

        self.root = self._grow_tree(X, y)

    # Will recursively built tree until we reach a STOPPING POINT
    def _grow_tree(self, X, y, depth=0):
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))

        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        if self.random_forest:
            feat_idxs = np.random.choice(n_feats, int(
                round(math.sqrt(self.n_features), 2)), replace=False)
        else:
            feat_idxs = np.random.choice(
                n_feats, self.n_features, replace=False)

        best_feature, best_thresh = self._best_split(X, y, feat_idxs)

        left_idx, right_idx = self._split(X[:, best_feature], best_thresh)

        # this is the recurisive part where it call itself until the
        left = self._grow_tree(X[left_idx, :], y[left_idx], depth + 1)
        right = self._grow_tree(X[right_idx, :], y[right_idx], depth + 1)
        return Node(best_feature, best_thresh, left, right)

    def _most_common_label(self, y):
        counter = Counter(y)
        value = counter.most_common(1)[0][0]
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
        parent_entropy = self._entropy(y)

        left_idx, right_idx = self._split(Xcolumn, thresholds)

        if len(left_idx) == 0 or len(right_idx) == 0:
            return 0

        n = len(y)
        n_l, n_r = len(left_idx), len(right_idx)
        e_l, e_r = self._entropy(y[left_idx]), self._entropy(y[right_idx])

        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r

        informaiton_gain = parent_entropy - child_entropy

        return informaiton_gain

    def _split(self, Xcolumn, thresholds):
        left_idxs = np.argwhere(Xcolumn <= thresholds).flatten()
        right_idxs = np.argwhere(Xcolumn > thresholds).flatten()
        return left_idxs, right_idxs

    def _entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log(p) for p in ps if p > 0])

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)


class Random_Forest:
    def __init__(self, number_of_trees=100, trees=None):
        self.number_of_trees = number_of_trees
        self.trees = trees

    def fit(self, Xtrain, ytrain):
        X_boots, y_boots = self._bootstrapping_data(Xtrain, ytrain)
        self.trees = self._create_decision_trees(
            self.number_of_trees, X_boots, y_boots)

    def _bootstrapping_data(self, X, y):
        n = len(X)
        idxes = [int(n * random.random()) for i in range(n)]

        X_boots = [X[idx] for idx in idxes]
        y_boots = [y[idx] for idx in idxes]

        return np.array(X_boots), np.array(y_boots)

    def _create_decision_trees(self, number_of_trees, X, y):
        decision_trees_list = []
        for x in range(number_of_trees):
            clf = DecisionTreeClassifier(max_depth=10, random_forest=True)
            clf.fit(X, y)

            decision_trees_list.append(clf)

        return decision_trees_list

    def predict_rt(self, Xtest):
        predicitons = np.array([x.predict(Xtest) for x in self.trees])

        final_preds = [Counter(predicitons[:, i]).most_common(
            1)[0][0] for i in range(predicitons.shape[1])]
        return np.array(final_preds)
