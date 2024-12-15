import numpy as np
import pandas as pd


class KDTree:

    def __init__(self, data, depth=0):
        if len(data) > 0:
            k = data.shape[1]
            axis = depth % k
            sorted_data = data[data[:, axis].argsort()]
            median_index = len(sorted_data) // 2

            self.location = sorted_data[median_index]
            self.left = KDTree(sorted_data[:median_index], depth + 1)
            self.right = KDTree(sorted_data[median_index + 1:], depth + 1)
        else:
            self.location = None
            self.left = None
            self.right = None

    def query(self, point, k=1, depth=0, best=None, metric="minkowski", power=2):
        if self.location is None:
            return best

        if best is None:
            best = []

        k_dim = len(point)
        axis = depth % k_dim

        next_best = None
        next_branch = None

        if point[axis] < self.location[axis]:
            next_branch = self.left
            next_best = self.right
        else:
            next_branch = self.right
            next_best = self.left

        best = next_branch.query(point, k, depth + 1, best, metric, power)

        dist = self.calculate_distance(point, self.location, metric, power)
        if len(best) < k or dist < best[-1][1]:
            best.append((self.location, dist))
            best = sorted(best, key=lambda x: x[1])[:k]

        if len(best) < k or abs(point[axis] - self.location[axis]) < best[-1][1]:
            best = next_best.query(point, k, depth + 1, best, metric, power)

        return best

    def calculate_distance(self, x1, x2, metric, power):
        if metric == "manhattan":
            return np.sum(np.abs(x1 - x2))
        elif metric == "euclidean":
            return np.sqrt(np.sum((x1 - x2) ** 2))
        elif metric == "minkowski":
            return np.sum(np.abs(x1 - x2) ** power) ** (1 / power)
        elif metric == "chebyshev":
            return np.max(np.abs(x1 - x2))
        else:
            raise ValueError("Invalid distance type")


class KNearestNeighbors:

    def __init__(self, n_neighbors: int = 5, metric: str = "minkowski", power: float = 2.0):
        self.__n_neighbors: int = n_neighbors
        self.__metric: str = metric
        self.__power: float = power if metric == "minkowski" else None
        self.__tree = None
        self.__y = None
        self.__classes = None

    def fit(self, x: pd.DataFrame, y: pd.Series):
        self.__x: np.ndarray = x.to_numpy(copy=True)
        self.__tree = KDTree(self.__x)
        self.__classes, self.__y = np.unique(y, return_inverse=True)
        return self
    
    def predict(self, x: pd.DataFrame):
        x = x.to_numpy(copy=True)
        y_pred = np.array([self.__predict_single(x_i) for x_i in x])
        y_pred = self.__classes[y_pred]
        return y_pred
    
    def __predict_single(self, x: np.ndarray):
        neighbors = self.__tree.query(x, k=self.__n_neighbors, metric=self.__metric, power=self.__power)
        k_nearest_y = np.array([self.__y[np.where((self.__x == neighbor[0]).all(axis=1))[0][0]] for neighbor in neighbors])
        most_common = np.bincount(k_nearest_y).argmax()
        return most_common


# Examples
if __name__ == "__main__":

    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from sklearn.neighbors import KNeighborsClassifier

    x, y = load_iris(return_X_y=True, as_frame=True)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    print("y_test:")
    print(y_test)

    # using scratch
    knn_scratch = KNearestNeighbors(n_neighbors=5, metric="minkowski", power=3)
    knn_scratch.fit(x_train, y_train)
    y_pred_scratch = knn_scratch.predict(x_test)
    acc = accuracy_score(y_test, y_pred_scratch)
    print("y_pred_scratch:")
    print(y_pred_scratch)
    print(f"Accuracy scratch: {acc}")

    # using sklearn
    knn_sklearn = KNeighborsClassifier(n_neighbors=5, metric="minkowski", p=3)
    knn_sklearn.fit(x_train, y_train)
    y_pred_sklearn = knn_sklearn.predict(x_test)
    acc = accuracy_score(y_test, y_pred_sklearn)
    print("y_pred_sklearn:")
    print(y_pred_sklearn)
    print(f"Accuracy sklearn: {acc}")

    # comparing predictions
    print("Is both predictions same?", np.array_equal(y_pred_scratch, y_pred_sklearn))
