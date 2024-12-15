import numpy as np
import pickle

class GaussianNaiveBayes:
    def __init__(self, var_smoothing: float = 1e-9):
        self.var_smoothing = var_smoothing
        self.classes = None
        self.class_prior = None
        self.theta = None  # mean of each feature per class
        self.var = None   # variance of each feature per class
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'GaussianNaiveBayes':
        X = np.array(X, dtype=np.float64)
        y = np.array(y)
        
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        n_samples, n_features = X.shape
        
        self.theta = np.zeros((n_classes, n_features))
        self.var = np.zeros((n_classes, n_features))
        
        class_count = np.zeros(n_classes, dtype=np.float64)
        for i, y_i in enumerate(self.classes):
            class_count[i] = np.sum(y == y_i)
        self.class_prior = class_count / n_samples
        
        for i, y_i in enumerate(self.classes):
            X_i = X[y == y_i]
            if X_i.shape[0]:
                self.theta[i, :] = np.mean(X_i, axis=0)
                self.var[i, :] = np.var(X_i, axis=0, ddof=1)
            else:
                self.theta[i, :] = np.mean(X, axis=0)
                self.var[i, :] = np.var(X, axis=0, ddof=1)
        
        self.var += self.var_smoothing * np.var(X, axis=0, ddof=1).max()
        
        return self
    
    def _joint_log_likelihood(self, X: np.ndarray) -> np.ndarray:
        joint_log_likelihood = np.zeros((X.shape[0], len(self.classes)))
        
        for i, y_i in enumerate(self.classes):
            jointi = -0.5 * np.sum(np.log(2. * np.pi * self.var[i, :]))
            jointi -= 0.5 * np.sum(((X - self.theta[i, :]) ** 2) / 
                                 self.var[i, :], axis=1)
            jointi += np.log(self.class_prior[i])
            joint_log_likelihood[:, i] = jointi
            
        return joint_log_likelihood
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.array(X, dtype=np.float64)
        
        log_prob = self._joint_log_likelihood(X)
        
        log_prob_max = log_prob.max(axis=1)
        exp_proba = np.exp(log_prob.T - log_prob_max).T
        proba = exp_proba / exp_proba.sum(axis=1)[:, np.newaxis]
        
        return proba
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes[np.argmax(self.predict_proba(X), axis=1)]
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return np.mean(self.predict(X) == y)
    
    def save_model(self, filename: str) -> None:
        model_params = {
            'classes': self.classes,
            'class_prior': self.class_prior,
            'theta': self.theta,
            'var': self.var,
            'var_smoothing': self.var_smoothing
        }
        with open(filename, 'wb') as f:
            pickle.dump(model_params, f)
    
    def load_model(self, filename: str) -> 'GaussianNaiveBayes':
        with open(filename, 'rb') as f:
            model_params = pickle.load(f)
        
        self.classes = model_params['classes']
        self.class_prior = model_params['class_prior']
        self.theta = model_params['theta']
        self.var = model_params['var']
        self.var_smoothing = model_params['var_smoothing']
        
        return self