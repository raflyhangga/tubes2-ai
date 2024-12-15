import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Union

class KDTree:
    def __init__(self, data: np.ndarray, depth: int = 0, indices: Optional[np.ndarray] = None):
        if indices is None:
            indices = np.arange(len(data))
        
        if len(indices) == 0:
            self.location = None
            self.left = self.right = None
            self.axis = None
            return
            
        k = data.shape[1]
        self.axis = depth % k
        
        idx = indices[data[indices, self.axis].argsort()]
        median_idx = len(idx) // 2
        
        self.location = data[idx[median_idx]]
        self.left = KDTree(data, depth + 1, idx[:median_idx])
        self.right = KDTree(data, depth + 1, idx[median_idx + 1:])
    
    def query(self, point: np.ndarray, k: int = 1, depth: int = 0, 
              heap: Optional[List[Tuple[np.ndarray, float]]] = None,
              metric: str = "minkowski", power: float = 2) -> List[Tuple[np.ndarray, float]]:
        if self.location is None:
            return heap or []
            
        if heap is None:
            heap = []
            
        axis = self.axis
        dist = self._calculate_distance(point, self.location, metric, power)
        
        if len(heap) < k:
            heap.append((self.location, dist))
            heap.sort(key=lambda x: x[1], reverse=True) 
        elif dist < heap[0][1]: 
            heap[0] = (self.location, dist)
            # Maintain heap property
            for i in range(len(heap)-1):
                if heap[i][1] < heap[i+1][1]:
                    heap[i], heap[i+1] = heap[i+1], heap[i]
                else:
                    break
                    
        if point[axis] < self.location[axis]:
            first, second = self.left, self.right
        else:
            first, second = self.right, self.left
            
        if first is not None:
            heap = first.query(point, k, depth + 1, heap, metric, power)
            
        if second is not None and (len(heap) < k or abs(point[axis] - self.location[axis]) < heap[0][1]):
            heap = second.query(point, k, depth + 1, heap, metric, power)
            
        return sorted(heap, key=lambda x: x[1])
    
    @staticmethod
    def _calculate_distance(x1: np.ndarray, x2: np.ndarray, metric: str, power: float) -> float:
        diff = np.abs(x1 - x2)
        if metric == "manhattan":
            return np.sum(diff)
        elif metric == "euclidean":
            return np.sqrt(np.sum(diff * diff))
        elif metric == "minkowski":
            return np.sum(diff ** power) ** (1 / power)
        elif metric == "chebyshev":
            return np.max(diff)
        raise ValueError(f"Invalid distance metric: {metric}")


class KNearestNeighbors:
    def __init__(self, n_neighbors: int = 5, metric: str = "minkowski", power: float = 2.0):
        self._n_neighbors = n_neighbors
        self._metric = metric
        self._power = power if metric == "minkowski" else None
        self._tree = None
        self._y = None
        self._classes = None
        
    def fit(self, x: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> 'KNearestNeighbors':
        self._x = x.to_numpy() if isinstance(x, pd.DataFrame) else x
        self._tree = KDTree(self._x)
        self._classes, self._y = np.unique(y, return_inverse=True)
        return self
    
    def predict(self, x: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        x_array = x.to_numpy() if isinstance(x, pd.DataFrame) else x
        
        predictions = np.empty(len(x_array), dtype=int)
        for i, x_i in enumerate(x_array):
            neighbors = self._tree.query(x_i, k=self._n_neighbors, metric=self._metric, power=self._power)
            indices = np.array([np.where((self._x == neighbor[0]).all(axis=1))[0][0] for neighbor in neighbors])
            predictions[i] = np.bincount(self._y[indices]).argmax()
            
        return self._classes[predictions]