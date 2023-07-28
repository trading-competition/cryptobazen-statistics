from sklearn.preprocessing import RobustScaler
import numpy as np


class LogRobustScaler:
    """
    LogRobustScaler applies a log transformation and a robust scaling to input data.

    This class is designed to transform data for use with machine learning models
    and then inverse transform the output of models for interpretation. The log
    transformation is applied first, followed by robust scaling which is less
    sensitive to outliers. The transformations are inverted in reverse order.

    ...

    Attributes
    ----------
    _scaler : sklearn.preprocessing.RobustScaler
        RobustScaler instance for scaling data after the log transformation.

    Methods
    -------
    fit_transform(x: np.ndarray) -> np.ndarray:
        Applies log transformation and robust scaling to the data.
    transform(x: np.ndarray) -> np.ndarray:
        Transforms the data using the previously fitted scaler.
    inverse_transform(x: np.ndarray) -> np.ndarray:
        Applies the inverse of the log transformation and robust scaling to the data.
    """

    def __init__(self):
        """
        Initializes the LogRobustScaler with a RobustScaler instance.
        """
        self._scaler = RobustScaler()

    def fit_transform(self, x):
        """
        Function to fit and transform the data with a log transformation and a robust scaler. Used while training the model

        Parameters
        ----------
        x : np.ndarray
            The input data to transform.

        Returns
        -------
        np.ndarray
            The transformed data.
        """
        log_x = np.log1p(x)
        return self._scaler.fit_transform(log_x)

    def transform(self, x):
        """
        Function to transform the data with a log transformation and a robust scaler. Used while predicting with the model

        Parameters
        ----------
        x : np.ndarray
            The input data to transform.

        Returns
        -------
        np.ndarray
            The transformed data.
        """
        log_x = np.log1p(x)
        return self._scaler.transform(log_x)

    def inverse_transform(self, x):
        """
        Function to inverse transform the data with a log transformation and a robust scaler. Used while predicting with the model

        Parameters
        ----------
        x : np.ndarray
            The model's output to inverse transform.

        Returns
        -------
        np.ndarray
            The inverse transformed data.
        """
        inversed = self._scaler.inverse_transform(x)
        return np.expm1(inversed)


class CyclicalTransformer:
    """
    CyclicalTransformer applies a cyclical transformation to input data.

    The transformation is applied by taking the sin and cos of the data,
    effectively transforming the data to two cyclical features. It's useful
    when dealing with cyclical data like hours in a day or months in a year.

    ...

    Attributes
    ----------
    max_val : float
        The maximum value used for the cyclical transformation.

    Methods
    -------
    transform(x: np.ndarray) -> tuple:
        Applies cyclical transformation to the data.
    """

    def __init__(self, max_val):
        """
        Initializes the CyclicalTransformer with the provided maximum value.

        Parameters
        ----------
        max_val : float
            The maximum value to use in the cyclical transformation.
        """
        self.max_val = max_val

    def transform(self, x):
        """
        Function to transform the data with a cyclical transformation.

        Parameters
        ----------
        x : np.ndarray
            The input data to transform.

        Returns
        -------
        tuple
            The transformed data as a tuple of two numpy arrays (x_sin, x_cos).
        """
        x_sin = np.sin(2 * np.pi * x / self.max_val)
        x_cos = np.cos(2 * np.pi * x / self.max_val)
        return x_sin, x_cos

    def fit_transform(self, x):
        """
        Function to transform the data with a cyclical transformation.
        This function is identical to `transform`, included for consistency with other preprocessing classes.

        Parameters
        ----------
        x : np.ndarray
            The input data to transform.

        Returns
        -------
        tuple
            The transformed data as a tuple of two numpy arrays (x_sin, x_cos).
        """
        return self.transform(x)
