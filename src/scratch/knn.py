import numpy as np
import pandas as pd


class KNearestNeighbors:

    def __init__(self, n_neighbors: int = 3, metric: str = "euclidean", power: float = None):
        self.__n_neighbors: int = n_neighbors
        
        if metric == "minkowski" and power is None:
            raise ValueError("Power must be provided for minkowski distance")
        
        if metric == "minkowski":
            self.__power: float = power
        elif metric == "manhattan":
            self.__power: float = 1.0
        elif metric == "euclidean":
            self.__power: float = 2.0
        elif metric == "chebyshev":
            self.__power: float = np.inf
        else:
            raise ValueError("Invalid distance type")

    def fit(self, x: pd.DataFrame, y: pd.Series):
        self.__x: np.ndarray = x.to_numpy(copy=True)
        self.__y: np.ndarray = y.to_numpy(copy=True)
        return self
    
    def predict(self, x: pd.DataFrame):
        x = x.to_numpy(copy=True)
        y_pred = np.array([self.__predict_single(x_i) for x_i in x])
        return y_pred
    
    def __predict_single(self, x: np.ndarray):
        distances = np.array([self.__calculate_distance(x, x_i) for x_i in self.__x])
        k_nearest_x_indices = np.argsort(distances)[:self.__n_neighbors]
        k_nearest_y = np.array([self.__y[i] for i in k_nearest_x_indices])
        most_common = np.bincount(k_nearest_y).argmax()
        return most_common
    
    def __calculate_distance(self, x1: np.ndarray, x2: np.ndarray):
        distance = np.linalg.norm(x=x1 - x2, ord=self.__power)
        return distance


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
