import numpy as np
from sklearn.preprocessing import LabelEncoder

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class ID3:
    def __init__(self, min_samples_split=2, max_depth=20, min_info_gain=1e-4):
        self.root = None
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.min_info_gain = min_info_gain
        self.label_encoder = LabelEncoder()
        self.n_features_in_ = None
        
    def fit(self, X, y):
        self.n_features_in_ = X.shape[1]
        X = np.asarray(X, dtype=np.float64)
        y = self.label_encoder.fit_transform(y)
        self.n_classes = len(self.label_encoder.classes_)
        self.class_weights = np.bincount(y) / len(y)
        self.root = self._grow_tree(X, y)
        return self
    
    def _entropy(self, y):
        if len(y) == 0:
            return 0
        
        hist = np.bincount(y, minlength=self.n_classes)
        ps = hist / len(y)
        ps = ps[ps > 0]
        return -np.sum(ps * np.log2(ps + 1e-10))
    
    def _information_gain(self, y, X_column, threshold):
        parent_entropy = self._entropy(y)
        
        left_indices = [x <= threshold for x in X_column]
        n_left = sum(left_indices)
        n_right = len(y) - n_left
        
        if n_left == 0 or n_right == 0:
            return 0, threshold
        
        left_indices = np.array(left_indices)
        right_indices = np.array([not x for x in left_indices])
        
        left_entropy = self._entropy(y[left_indices])
        right_entropy = self._entropy(y[right_indices])
        
        p_left = n_left / len(y)
        weighted_entropy = p_left * left_entropy + (1 - p_left) * right_entropy
        
        return parent_entropy - weighted_entropy, threshold
    
    def _best_split(self, X, y):
        best_gain = -float('inf')
        best_feature = None
        best_threshold = None
        min_samples_per_split = max(int(0.05 * len(y)), self.min_samples_split)
        
        for feature in range(self.n_features_in_):
            X_column = X[:, feature]
            finite_mask = [np.isfinite(x) for x in X_column]
            if not any(finite_mask):
                continue
                
            # Convert to numpy array for indexing
            finite_mask = np.array(finite_mask)
            X_finite = X_column[finite_mask]
            y_finite = y[finite_mask]
            
            if len(X_finite) < min_samples_per_split:
                continue
            
            unique_values = np.unique(X_finite)
            if len(unique_values) > 10:
                thresholds = np.percentile(X_finite, [20, 40, 60, 80])
            else:
                thresholds = unique_values[:-1]
            
            for threshold in thresholds:
                gain, _ = self._information_gain(y_finite, X_finite, threshold)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        
        if best_gain < self.min_info_gain:
            return None, None
            
        return best_feature, best_threshold
    
    def _grow_tree(self, X, y, depth=0):
        n_samples = len(y)
        unique_classes = np.unique(y)
        
        if (depth >= self.max_depth or 
            n_samples < self.min_samples_split or 
            len(unique_classes) == 1 or 
            np.all(np.isnan(X))):
            return Node(value=self._most_common_label(y))
        
        best_feature, best_threshold = self._best_split(X, y)
        
        if best_feature is None:
            return Node(value=self._most_common_label(y))
        
        left_mask = [x <= best_threshold for x in X[:, best_feature]]
        right_mask = [not x for x in left_mask]
        
        left_mask = np.array(left_mask)
        right_mask = np.array(right_mask)
        
        if sum(left_mask) < self.min_samples_split or sum(right_mask) < self.min_samples_split:
            return Node(value=self._most_common_label(y))
        
        left = self._grow_tree(X[left_mask], y[left_mask], depth + 1)
        right = self._grow_tree(X[right_mask], y[right_mask], depth + 1)
        
        return Node(best_feature, best_threshold, left, right)
    
    def _most_common_label(self, y):
        if len(y) == 0:
            return np.argmax(self.class_weights)
        counts = np.bincount(y, minlength=self.n_classes)
        return np.argmax(counts)
    
    def predict(self, X):
        if self.root is None:
            raise ValueError("Model not fitted. Call fit first.")
            
        X = np.asarray(X, dtype=np.float64)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {X.shape[1]}")
        
        predictions = np.array([self._traverse_tree(x, self.root) for x in X])
        return self.label_encoder.inverse_transform(predictions)
    
    def _traverse_tree(self, x, node):
        if node.value is not None:
            return node.value
            
        feature_value = x[node.feature]
        if np.isnan(feature_value):
            return self._most_common_label(np.array([]))
            
        if feature_value <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)