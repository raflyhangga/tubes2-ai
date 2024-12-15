import numpy as np
import pickle

class GaussianNaiveBayes:
    def __init__(self, var_smoothing: float = 1e-9):
        self.var_smoothing = var_smoothing
        self.classes_ = None         # Array of class labels
        self.class_prior_ = None     # P(y) - probability of each class
        self.theta_ = None           # μy - mean of each feature per class
        self.var_ = None            # σ²y - variance of each feature per class
        self.epsilon_ = None        # Smoothing value for variances
        self.n_features_ = None     # Number of features
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'GaussianNaiveBayes':
        X = np.array(X, dtype=np.float64)
        y = np.array(y)
        
        n_samples, self.n_features_ = X.shape
        
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        
        self.theta_ = np.zeros((n_classes, self.n_features_))
        self.var_ = np.zeros((n_classes, self.n_features_))
        
        # P(y) = number of samples in class y / total number of samples
        class_count = np.array([np.sum(y == c) for c in self.classes_])
        self.class_prior_ = class_count / n_samples
        
        #  μy (theta) and σ²y (var)
        for i, c in enumerate(self.classes_):
            X_c = X[y == c]
            if X_c.shape[0] > 0:
                # μy = mean of features for class y
                self.theta_[i, :] = np.mean(X_c, axis=0)
                # σ²y = variance of features for class y
                self.var_[i, :] = np.var(X_c, axis=0)
        
        # Smoothing
        self.epsilon_ = self.var_smoothing * np.var(X, axis=0).max()
        self.var_ += self.epsilon_
        
        return self
    
    def _joint_log_likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the joint log likelihood for all classes.
        
        P(x_i | y) = 1/sqrt(2πσ²y) * exp(-(x_i - μy)²/(2σ²y))
        log P(x_i | y) = -0.5 log(2πσ²y) - (x_i - μy)²/(2σ²y)
        """
        joint_log_likelihood = np.zeros((X.shape[0], len(self.classes_)))
        
        for i, c in enumerate(self.classes_):
            # -0.5 log(2πσ²y)
            log_proba = -0.5 * np.sum(np.log(2. * np.pi * self.var_[i, :]))
            # -(x_i - μy)²/(2σ²y)
            log_proba -= 0.5 * np.sum(
                ((X - self.theta_[i, :]) ** 2) / self.var_[i, :], axis=1)
            # log of class prior probability
            log_proba += np.log(self.class_prior_[i])
            
            joint_log_likelihood[:, i] = log_proba
            
        return joint_log_likelihood
    
    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.array(X, dtype=np.float64)
        return self._joint_log_likelihood(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        log_prob = self.predict_log_proba(X)
        # Log-sum-exp trick for numerical stability
        log_prob_max = log_prob.max(axis=1)
        exp_proba = np.exp(log_prob.T - log_prob_max).T
        proba = exp_proba / exp_proba.sum(axis=1)[:, np.newaxis]
        
        return proba
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return np.mean(self.predict(X) == y)
    
    def save_model(self, filename: str) -> None:
        model_params = {
            'classes_': self.classes_,
            'class_prior_': self.class_prior_,
            'theta_': self.theta_,
            'var_': self.var_,
            'epsilon_': self.epsilon_,
            'n_features_': self.n_features_,
            'var_smoothing': self.var_smoothing
        }
        with open(filename, 'wb') as f:
            pickle.dump(model_params, f)
    
    def load_model(self, filename: str) -> 'GaussianNaiveBayes':
        with open(filename, 'rb') as f:
            model_params = pickle.load(f)
        
        self.classes_ = model_params['classes_']
        self.class_prior_ = model_params['class_prior_']
        self.theta_ = model_params['theta_']
        self.var_ = model_params['var_']
        self.epsilon_ = model_params['epsilon_']
        self.n_features_ = model_params['n_features_']
        self.var_smoothing = model_params['var_smoothing']
        
        return self