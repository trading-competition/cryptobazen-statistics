from sklearn.preprocessing import RobustScaler
import numpy as np


class PercentageChangeScaler:
    def __init__(self, quantile_range=(25, 75), with_centering=True, with_scaling=True, copy=True):
        """
            RelativeChangeScaler is designed to normalize financial time series data,
            such as Bitcoin prices, by focusing on the relative changes between
            consecutive price points. This scaler emphasizes the proportional changes
            rather than absolute price values, thereby capturing the inherent volatility
            of the financial instrument.

            The scaler works by calculating the percentage change between consecutive
            price points, applying a RobustScaler to these changes to mitigate the
            influence of extreme outliers, and then rescaling the output to a
            predefined range, typically between -1 and 1.

            The advantage of RelativeChangeScaler over traditional scalers is its
            ability to preserve the relative movements within the market, reflecting
            true market conditions, including volatility. It avoids the pitfalls of
            over-smoothing or distorting the data, which can occur with scalers that
            only standardize based on distribution statistics like the mean and
            standard deviation. By using the median and interquartile range, it
            remains robust to the influence of sudden spikes or drops, which are
            common in cryptocurrency markets.

            This scaler is particularly useful for machine learning models that are
            sensitive to the scale and distribution of the input data and that require
            capturing the dynamic range of market movements for accurate predictions.

            Parameters:
            -----------
            quantile_range : tuple, default=(25, 75)
                Quantile range used for RobustScaler to calculate the interquartile range.
            with_centering : boolean, default=True
                Whether to center the data before scaling.
            with_scaling : boolean, default=True
                Whether to scale the data to the interquartile range.
            copy : boolean, default=True
                Whether to copy the input data, or perform in-place scaling.

            Methods:
            --------
            fit_transform(prices):
                Calculate the relative changes, apply RobustScaler, and rescale to [-1, 1].
            inverse_transform(scaled_data):
                Inverse the transformation from scaled data back to the original price scale.
            """
        self.robust_scaler = RobustScaler(quantile_range=quantile_range,
                                          with_centering=with_centering,
                                          with_scaling=with_scaling,
                                          copy=copy)
        self.scale_ = None
        self.center_ = None
        self.starting_price = None
        self.starting_time = None
        self.min = None
        self.max = None

    def fit_transform(self, prices):
        # Compute the percentage changes
        percentage_changes = self._calculate_percentage_changes(prices)
        # Fit and transform the percentage changes using RobustScaler
        scaled_changes = self.robust_scaler.fit_transform(percentage_changes)
        self.scale_ = self.robust_scaler.scale_
        self.center_ = self.robust_scaler.center_
        # Rescale to the range [-1, 1]
        min_max_scaled_changes = self._min_max_scale(scaled_changes)
        return min_max_scaled_changes.flatten()

    def inverse_transform(self, scaled_data):
        # Inverse the min-max scale
        unscaled_data = self._min_max_inverse_scale(scaled_data)
        # Inverse transform the data using the inverse function of the RobustScaler
        inverse_scaled_changes = self.robust_scaler.inverse_transform(unscaled_data)
        inverse_scaled_changes = np.round(inverse_scaled_changes, decimals=6)
        # Convert percentage changes back to prices
        prices = self._calculate_prices_from_changes(inverse_scaled_changes)
        return prices

    def _calculate_percentage_changes(self, prices):
        # Calculate the percentage changes from the prices
        self.starting_price = np.array(prices)[0]
        self.starting_time = np.array(prices.index)[0]
        new = np.array(prices[1:])
        old = np.array(prices[:-1])
        changes = np.round(new - old, decimals=2)
        percentual_changes = np.round(changes / old * 100, decimals=6)
        # Add a 0 at the beginning to keep the same length
        percentual_changes = np.insert(percentual_changes, 0, 0)
        percentual_changes = percentual_changes.reshape(-1, 1)
        return percentual_changes

    def _calculate_prices_from_changes(self, changes):
        # Calculate the prices back from percentage changes
        prices = [self.starting_price]
        for change in changes:
            prices.append(prices[-1] * (1 + change / 100))
        return np.round(np.array(prices), decimals=2)[1:]

    def _min_max_scale(self, data):
        # Scale the data to the range [-1, 1]
        self.max = np.max(data)
        self.min = np.min(data)
        scaled_data = 2 * (data - self.min) / (self.max - self.min) - 1
        return scaled_data

    def _min_max_inverse_scale(self, scaled_data):
        inversed_data = (scaled_data + 1) / 2 * (self.max - self.min) + self.min
        return inversed_data


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

