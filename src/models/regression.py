import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class LinearRegression:

    def __init__(self):
        self.weight = np.random.uniform(-1, 1)  # Random initialization
        self.bias = np.random.uniform(-1, 1)    # Random initialization
        self.X = pd.Series(dtype=float)
        self.y = pd.Series(dtype=float)

    def fit(self, X: pd.Series, y: pd.Series, epochs: int = 3000, learning_rate: float = 0.01) -> None:
        self.X = X
        self.y = y

        for epoch in range(epochs):
            # Predicted values
            y_pred = self.weight * X + self.bias

            # Compute gradients
            d_weight = -(2 / len(X)) * np.dot(X, (y - y_pred))
            d_bias = -(2 / len(X)) * np.sum(y - y_pred)

            # Update parameters using gradient descent
            self.weight -= learning_rate * d_weight
            self.bias -= learning_rate * d_bias
            
            # Output performance every 100 epochs
            if epoch % 100 == 0:
                print(f"Epoch: {epoch}, Weight: {self.weight:.3f}, Bias: {self.bias:.3f}, SSE: {self.sse(y, y_pred):.3f}")

            # Early stopping condition for overfitting
            if self.sse(y, y_pred) < 0.001:
                print(f'Model reached a < 0.001 error score, learning terminated to avoid overfitting.')
                break

    def sse(self, y: pd.Series, y_pred: pd.Series) -> float:
        '''Calculate Sum of Squared Errors (SSE)'''
        return np.sum((y - y_pred) ** 2)

    def predict(self, X: pd.Series) -> pd.Series:
        '''Predict values for new data'''
        return self.weight * X + self.bias

    def info(self):
        '''Display model statistics and parameters'''
        statistics_data = {
            'X': [
                np.mean(self.X).round(3),
                np.median(self.X).round(3),
                np.std(self.X, ddof=0).round(3),  
                np.std(self.X, ddof=1).round(3), 
                np.var(self.X).round(3)
            ],
            'y': [
                np.mean(self.y).round(3),
                np.median(self.y).round(3),
                np.std(self.y, ddof=0).round(3),
                np.std(self.y, ddof=1).round(3),
                np.var(self.y).round(3)
            ]
        }

        statistics = pd.DataFrame(data=statistics_data, index=['Mean', 'Median', 'StDev', 'Sample StDev', 'Variance'])
        print(f'\nData Statistics: \n{statistics}\n')
        print(f'Model Parameters: \nWeight: {round(self.weight, 3)}\nBias: {round(self.bias, 3)}\nSSE: {round(self.sse(self.y, (self.weight * self.X + self.bias)), 3)}\nModel: y = {round(self.weight, 3)}x + {round(self.bias, 3)}\n')

    def display_data(self):
        '''Display data and the regression line'''
        x_min = self.X.min()
        x_max = self.X.max()

        plt.scatter(self.X, self.y, color='blue', label='Data Points')

        x_values = np.linspace(x_min, x_max, 100)
        y_values = self.weight * x_values + self.bias
        plt.plot(x_values, y_values, color='red', label='Regression Line')

        plt.xlabel('X')
        plt.ylabel('y')
        plt.title('Data and Regression Line')
        plt.legend()

        plt.show()

class RidgeRegression(LinearRegression):
    def __init__(self, lambda_reg: float = 0.1):
        super().__init__()
        self.lambda_reg = lambda_reg  # Regularization strength

    def fit(self, X: pd.Series, y: pd.Series, epochs: int = 3000, learning_rate: float = 0.01) -> None:
        self.X = X
        self.y = y

        for epoch in range(epochs):
            # Predicted values
            y_pred = self.weight * X + self.bias

            # Compute gradients with L2 regularization
            d_weight = -(2 / len(X)) * np.dot(X, (y - y_pred)) + 2 * self.lambda_reg * self.weight
            d_bias = -(2 / len(X)) * np.sum(y - y_pred)

            # Update parameters using gradient descent
            self.weight -= learning_rate * d_weight
            self.bias -= learning_rate * d_bias
            
            if epoch % 100 == 0:
                print(f"Epoch: {epoch}, Weight: {self.weight:.3f}, Bias: {self.bias:.3f}, SSE: {self.sse(y, y_pred):.3f}")

    def info(self):
        '''Display model statistics and parameters'''
        print("Ridge Regression Model")
        super().info()

class LassoRegression(LinearRegression):
    def __init__(self, alpha: float = 0.1):
        super().__init__()
        self.alpha = alpha  # L1 regularization strength

    def fit(self, X: pd.Series, y: pd.Series, epochs: int = 3000, learning_rate: float = 0.01) -> None:
        self.X = X
        self.y = y

        for epoch in range(epochs):
            # Predicted values
            y_pred = self.weight * X + self.bias

            # Compute gradients with L1 regularization
            d_weight = -(2 / len(X)) * np.dot(X, (y - y_pred)) + self.alpha * np.sign(self.weight)
            d_bias = -(2 / len(X)) * np.sum(y - y_pred)

            # Update parameters using gradient descent
            self.weight -= learning_rate * d_weight
            self.bias -= learning_rate * d_bias
            
            if epoch % 100 == 0:
                print(f"Epoch: {epoch}, Weight: {self.weight:.3f}, Bias: {self.bias:.3f}, SSE: {self.sse(y, y_pred):.3f}")

    def info(self):
        '''Display model statistics and parameters'''
        print("Lasso Regression Model")
        super().info()
