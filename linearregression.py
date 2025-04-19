import numpy as np

class LinearRegression:
    """
    A simple implementation of Linear Regression similar to sklearn's LinearRegression.

    """
    
    def __init__(self, fit_intercept=True, normalize=False):
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.coef_ = None
        self.intercept_ = None
        self._X_mean = None
        self._X_std = None
        
    def _normalize_features(self, X):
        if self._X_mean is None or self._X_std is None:
            self._X_mean = np.mean(X, axis=0)
            self._X_std = np.std(X, axis=0)
            self._X_std[self._X_std == 0] = 1.0
            
        return (X - self._X_mean) / self._X_std
    
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        
        if self.normalize:
            X = self._normalize_features(X)
        
        if self.fit_intercept:
            X_with_intercept = np.column_stack((np.ones(X.shape[0]), X))
        else:
            X_with_intercept = X
            
        # Calculate coefficients using the normal equation
        # (X^T * X)^(-1) * X^T * y
        XTX = np.dot(X_with_intercept.T, X_with_intercept)
        XTX_inv = np.linalg.inv(XTX)
        XTy = np.dot(X_with_intercept.T, y)
        coeffs = np.dot(XTX_inv, XTy)
        
        if self.fit_intercept:
            self.intercept_ = coeffs[0]
            self.coef_ = coeffs[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = coeffs
            
        return self
    
    def predict(self, X):
        X = np.array(X)
        
        if self.normalize:
            X = (X - self._X_mean) / self._X_std
            
        return np.dot(X, self.coef_) + self.intercept_
    
    def score(self, X, y):
        y_pred = self.predict(X)
        u = ((y - y_pred) ** 2).sum()
        v = ((y - y.mean()) ** 2).sum()
        return 1 - u/v


# Example usage
if __name__ == "__main__":
    # Generate some synthetic data
    np.random.seed(42)
    X = 2 * np.random.rand(100, 3)
    y = 4 + 3 * X[:, 0] - 2 * X[:, 1] + 0.5 * X[:, 2] + np.random.randn(100)
    
    # Split into train and test sets
    X_train, X_test = X[:80], X[80:]
    y_train, y_test = y[:80], y[80:]
    
    # Fit the model
    model = LinearRegression(fit_intercept=True, normalize=True)
    model.fit(X_train, y_train)
    
    # Print the results
    print(f"Intercept: {model.intercept_}")
    print(f"Coefficients: {model.coef_}")
    print(f"R² score: {model.score(X_test, y_test)}")
    
    from sklearn.linear_model import LinearRegression as SklearnLR
    
    sklearn_model = SklearnLR()
    sklearn_model.fit(X_train, y_train)
    
    print("\nScikit-learn results:")
    print(f"Intercept: {sklearn_model.intercept_}")
    print(f"Coefficients: {sklearn_model.coef_}")
    print(f"R² score: {sklearn_model.score(X_test, y_test)}")