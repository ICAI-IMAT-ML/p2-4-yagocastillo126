import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns


class LinearRegressor:
    """
    Extended Linear Regression model with support for categorical variables and gradient descent fitting.
    """

    def __init__(self):
        self.coefficients = None
        self.intercept = None

    """
    This next "fit" function is a general function that either calls the *fit_multiple* code that
    you wrote last week, or calls a new method, called *fit_gradient_descent*, not implemented (yet)
    """

    def fit(self, X, y, method="least_squares", learning_rate=0.01, iterations=1000):
        """
        Fit the model using either normal equation or gradient descent.

        Args:
            X (np.ndarray): Independent variable data (2D array).
            y (np.ndarray): Dependent variable data (1D array).
            method (str): method to train linear regression coefficients.
                          It may be "least_squares" or "gradient_descent".
            learning_rate (float): Learning rate for gradient descent.
            iterations (int): Number of iterations for gradient descent.

        Returns:
            None: Modifies the model's coefficients and intercept in-place.
        """
        if method not in ["least_squares", "gradient_descent"]:
            raise ValueError(
                f"Method {method} not available for training linear regression."
            )
        if np.ndim(X) == 1:
            X = X.reshape(-1, 1)

        if method == "least_squares":
            self.fit_multiple(X, y)
        elif method == "gradient_descent":
            self.fit_gradient_descent(X, y, learning_rate, iterations)

    def fit_multiple(self, X, y):
        """
        Fit the model using multiple linear regression (more than one independent variable).

        This method applies the matrix approach to calculate the coefficients for
        multiple linear regression.

        Args:
            X (np.ndarray): Independent variable data (2D array), with bias.
            y (np.ndarray): Dependent variable data (1D array).

        Returns:
            None: Modifies the model's coefficients and intercept in-place.
        """
        # Replace this code with the code you did in the previous laboratory session
        # Agrega una columna de unos a X para el término de intercepción (b0)
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        # Calcula los coeficientes óptimos usando la ecuación normal
        beta = np.linalg.pinv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
        # Extrae el término de intercepción y los coeficientes
        self.intercept = beta[0]
        self.coefficients = beta[1:]

    def fit_gradient_descent(self, X, y, learning_rate=0.01, iterations=1000):
        """
        Fit the model using either normal equation or gradient descent.

        Args:
            X (np.ndarray): Independent variable data (2D array), with bias.
            y (np.ndarray): Dependent variable data (1D array).
            learning_rate (float): Learning rate for gradient descent.
            iterations (int): Number of iterations for gradient descent.

        Returns:
            None: Modifies the model's coefficients and intercept in-place.
        """

        # Initialize the parameters to very small values (close to 0)
        m ,n = X.shape
        self.coefficients = (
            np.random.rand(n) * 0.01
        )  # Small random numbers
        self.intercept = np.random.rand() * 0.01
        self.valores_loss = []  # Para los graficos de rolling in the deep
        self.parametros = []  # Para los graficos de rolling in the deep
        # Implement gradient descent
        for epoch in range(iterations):
            predictions = self.predict(X)
            error = predictions - y


            mse = np.mean(error**2)
            self.valores_loss.append(mse)

            self.parametros.append((self.intercept, *self.coefficients))

            # TODO: Write the gradient values and the updates for the paramenters
            gradient = (2 * learning_rate / m) * X.T.dot(error)
            intercept_gradient = (2 * learning_rate / m) * np.sum(error)
            self.intercept -= intercept_gradient 
            self.coefficients -= gradient 

            # Calculate and print the loss every 10 epochs
            if epoch % 10000 == 0:
                mse = np.mean(error ** 2)
                print(f"Epoch {epoch}: MSE = {mse}")

    def predict(self, X):
        """
        Predict the dependent variable values using the fitted model.

        Args:
            X (np.ndarray): Independent variable data (1D or 2D array).
            fit (bool): Flag to indicate if fit was done.

        Returns:
            np.ndarray: Predicted values of the dependent variable.

        Raises:
            ValueError: If the model is not yet fitted.
        """

        # Paste your code from last week

        if self.coefficients is None or self.intercept is None:
            raise ValueError("Model is not yet fitted")

        if np.ndim(X) == 1:
            # Predict when X is only one variable
            predictions = self.intercept + self.coefficients[0]*X
        else:
            # Predict when X is more than one variable
            X_b = np.c_[np.ones((X.shape[0], 1)), X]
            predictions = X_b.dot(np.r_[self.intercept, self.coefficients])
        return predictions


def evaluate_regression(y_true, y_pred):
    """
    Evaluates the performance of a regression model by calculating R^2, RMSE, and MAE.

    Args:
        y_true (np.ndarray): True values of the dependent variable.
        y_pred (np.ndarray): Predicted values by the regression model.

    Returns:
        dict: A dictionary containing the R^2, RMSE, and MAE values.
    """

    # R^2 Score
    # Calculate R^2
    ss_total = (np.sum((y_true - y_pred) ** 2))
    ss_residual = (np.sum((y_true - np.mean(y_true)) ** 2))
    r_squared = 1 - (ss_total/ss_residual)

    # Root Mean Squared Error
    # Calculate RMSE
    rmse = np.sqrt((np.sum((y_true - y_pred)**2)/len(y_true)))

    # Mean Absolute Error
    # Calculate MAE
    mae = (np.sum(abs(y_true - y_pred))/len(y_true))

    return {"R2": r_squared, "RMSE": rmse, "MAE": mae}


def one_hot_encode(X, categorical_indices, drop_first=False):
    """
    One-hot encode the categorical columns specified in categorical_indices. This function
    shall support string variables.

    Args:
        X (np.ndarray): 2D data array.
        categorical_indices (list of int): Indices of columns to be one-hot encoded.
        drop_first (bool): Whether to drop the first level of one-hot encoding to avoid multicollinearity.

    Returns:
        np.ndarray: Transformed array with one-hot encoded columns.
    """
import numpy as np

def one_hot_encode(X, categorical_indices, drop_first=False):
    """
    One-hot encode the categorical columns specified in categorical_indices. This function
    supports categorical variables including strings.

    Args:
        X (np.ndarray): 2D data array.
        categorical_indices (list of int): Indices of columns to be one-hot encoded.
        drop_first (bool): Whether to drop the first level of one-hot encoding to avoid multicollinearity.

    Returns:
        np.ndarray: Transformed array with one-hot encoded columns.
    """
    transformed_columns = [[] for _ in range(X.shape[1])]
    
    for col in categorical_indices:
        unique_values = np.unique(X[:, col])
        if drop_first:
            unique_values = unique_values[1:]  # Exclude the first category

        for val in unique_values:
            transformed_columns[col].append((X[:, col] == val).astype(float))
    
    result = []
    for col_index in range(X.shape[1]):
        if col_index in categorical_indices:
            result.extend(transformed_columns[col_index])
        else:
            result.append(X[:, col_index].astype(float))
    
    return np.column_stack(result)