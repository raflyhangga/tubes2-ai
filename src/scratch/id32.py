import numpy as np
import pandas as pd
from collections import Counter
from sklearn.preprocessing import LabelEncoder

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class ID3:
    def __init__(self, min_samples_split=2, max_depth=10):
        self.root = None
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.feature_names = None
        self.target_attribute = None
        self.attribute_types = None
        self.label_encoder = LabelEncoder()
        
    def fit(self, data, target_attribute, attribute_types):
        self.target_attribute = target_attribute
        self.attribute_types = attribute_types
        self.feature_names = list(attribute_types.keys())
        
        X = data[self.feature_names]
        y = data[target_attribute]
        
        y_encoded = self.label_encoder.fit_transform(y)
        
        X = X.values
        
        self.n_classes = len(self.label_encoder.classes_)
        self.n_features = X.shape[1]
        
        self.root = self._grow_tree(X, y_encoded)
        return self
        
    def _entropy(self, y):
        if len(y) == 0:
            return 0
        
        hist = np.bincount(y, minlength=self.n_classes)
        ps = hist / len(y)
        ps = ps[ps > 0]  # Remove zero probabilities
        return -np.sum(ps * np.log2(ps + 1e-10))
    
    def _information_gain(self, y, X_column, threshold=None, is_numeric=True):
        parent_entropy = self._entropy(y)
        
        if is_numeric:
            left_mask = X_column <= threshold
            right_mask = ~left_mask
            
            if sum(left_mask) == 0 or sum(right_mask) == 0:
                return 0, threshold
            
            left_entropy = self._entropy(y[left_mask])
            right_entropy = self._entropy(y[right_mask])
            
            p_left = sum(left_mask) / len(y)
            p_right = sum(right_mask) / len(y)
            
            gain = parent_entropy - (p_left * left_entropy + p_right * right_entropy)
            return gain, threshold
        else:
            unique_values = np.unique(X_column)
            max_gain = -float('inf')
            best_value = None
            
            for value in unique_values:
                left_mask = X_column == value
                right_mask = ~left_mask
                
                if sum(left_mask) == 0 or sum(right_mask) == 0:
                    continue
                
                left_entropy = self._entropy(y[left_mask])
                right_entropy = self._entropy(y[right_mask])
                
                p_left = sum(left_mask) / len(y)
                p_right = sum(right_mask) / len(y)
                
                gain = parent_entropy - (p_left * left_entropy + p_right * right_entropy)
                
                if gain > max_gain:
                    max_gain = gain
                    best_value = value
            
            return max_gain, best_value
    
    def _best_split(self, X, y):
        best_gain = -float('inf')
        best_feature = None
        best_threshold = None
        
        for feature_idx, feature_name in enumerate(self.feature_names):
            X_column = X[:, feature_idx]
            is_numeric = self.attribute_types[feature_name] == 'numerical'
            
            if is_numeric:
                X_column = np.nan_to_num(X_column, nan=-np.inf)
                
                unique_values = np.unique(X_column[np.isfinite(X_column)])
                if len(unique_values) > 10:
                    thresholds = np.percentile(X_column[np.isfinite(X_column)], 
                                            np.linspace(0, 100, 10))
                else:
                    thresholds = unique_values
                
                for threshold in thresholds:
                    gain, _ = self._information_gain(y, X_column, threshold, is_numeric=True)
                    if gain > best_gain:
                        best_gain = gain
                        best_feature = feature_idx
                        best_threshold = threshold
            else:
                gain, threshold = self._information_gain(y, X_column, is_numeric=False)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))
        
        if (n_labels == 1 or n_samples < self.min_samples_split or 
            depth >= self.max_depth):
            leaf_value = self._most_common_label(y)
            return Node(value=self.label_encoder.inverse_transform([leaf_value])[0])
        
        best_feature, best_threshold = self._best_split(X, y)
        
        if best_feature is None:  # No valid split found
            leaf_value = self._most_common_label(y)
            return Node(value=self.label_encoder.inverse_transform([leaf_value])[0])
        
        feature_name = self.feature_names[best_feature]
        is_numeric = self.attribute_types[feature_name] == 'numerical'
        
        if is_numeric:
            X_column = np.nan_to_num(X[:, best_feature], nan=-np.inf)
            left_mask = X_column <= best_threshold
        else:
            left_mask = X[:, best_feature] == best_threshold
        right_mask = ~left_mask
        
        left = self._grow_tree(X[left_mask], y[left_mask], depth + 1)
        right = self._grow_tree(X[right_mask], y[right_mask], depth + 1)
        
        return Node(self.feature_names[best_feature], best_threshold, left, right)
    
    def _most_common_label(self, y):
        """Return the most common label in an array"""
        counter = Counter(y)
        return counter.most_common(1)[0][0]
    
    def predict(self, X):
        """Predict class labels for samples in X"""
        if isinstance(X, pd.DataFrame):
            X = X[self.feature_names].values
        
        return np.array([self._traverse_tree(x, self.root) for x in X])
    
    def _traverse_tree(self, x, node):
        """Traverse the tree to make a prediction"""
        if node.value is not None:
            return node.value
        
        feature_idx = self.feature_names.index(node.feature)
        feature_value = np.nan_to_num(x[feature_idx], nan=-np.inf)
        
        is_numeric = self.attribute_types[node.feature] == 'numerical'
        
        if is_numeric:
            if feature_value <= node.threshold:
                return self._traverse_tree(x, node.left)
            return self._traverse_tree(x, node.right)
        else:
            if feature_value == node.threshold:
                return self._traverse_tree(x, node.left)
            return self._traverse_tree(x, node.right)