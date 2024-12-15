import numpy as np
import pandas as pd
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, ClassifierMixin

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class ID32(BaseEstimator, ClassifierMixin):
    def __init__(self, criterion='entropy', max_depth=10, min_samples_split=2, random_state=None):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.root = None
        self.label_encoder = LabelEncoder()
        self.feature_names = None

    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X = X.values
        elif isinstance(X, np.ndarray):
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        else:
            raise ValueError("X should be a pandas DataFrame or a NumPy array.")
        
        y = self.label_encoder.fit_transform(y)
        
        self.n_classes = len(self.label_encoder.classes_)
        self.n_features = X.shape[1]
        
        self.root = self._grow_tree(X, y)
        return self

    def _entropy(self, y):
        hist = np.bincount(y, minlength=self.n_classes)
        ps = hist / len(y)
        ps = ps[ps > 0]
        return -np.sum(ps * np.log2(ps + 1e-10))

    def _information_gain(self, y, X_column, threshold):
        parent_entropy = self._entropy(y)
        left_mask = X_column <= threshold
        right_mask = ~left_mask

        if sum(left_mask) == 0 or sum(right_mask) == 0:
            return 0

        left_entropy = self._entropy(y[left_mask])
        right_entropy = self._entropy(y[right_mask])

        p_left = sum(left_mask) / len(y)
        p_right = sum(right_mask) / len(y)

        return parent_entropy - (p_left * left_entropy + p_right * right_entropy)

    def _best_split(self, X, y):
        best_gain = -float('inf')
        best_feature = None
        best_threshold = None

        for feature_idx in range(X.shape[1]):
            X_column = X[:, feature_idx]
            thresholds = np.unique(X_column)
            
            for threshold in thresholds:
                gain = self._information_gain(y, X_column, threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold

        return best_feature, best_threshold

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        if n_labels == 1 or n_samples < self.min_samples_split or depth >= self.max_depth:
            leaf_value = self._most_common_label(y)
            return Node(value=self.label_encoder.inverse_transform([leaf_value])[0])

        best_feature, best_threshold = self._best_split(X, y)

        if best_feature is None:
            leaf_value = self._most_common_label(y)
            return Node(value=self.label_encoder.inverse_transform([leaf_value])[0])

        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask

        left = self._grow_tree(X[left_mask], y[left_mask], depth + 1)
        right = self._grow_tree(X[right_mask], y[right_mask], depth + 1)

        return Node(self.feature_names[best_feature], best_threshold, left, right)

    def _most_common_label(self, y):
        counter = Counter(y)
        return counter.most_common(1)[0][0]

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.value is not None:
            return node.value

        feature_idx = self.feature_names.index(node.feature)
        feature_value = x[feature_idx]

        if feature_value <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
