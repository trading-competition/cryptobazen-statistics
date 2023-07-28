from sklearn.preprocessing import RobustScaler
import numpy as np


class LogRobustScaler:
    def __init__(self):
        self._scaler = RobustScaler()

    def fit_transform(self, x):
        """
        Function to fit and transform the data with a log transformation and a robust scaler. Used while training the model
        :param x:
        :return:
        """
        log_x = np.log1p(x)
        return self._scaler.fit_transform(log_x)

    def transform(self, x):
        """
        Function to transform the data with a log transformation and a robust scaler. Used while predicting with the model
        :param x:
        :return:
        """
        log_x = np.log1p(x)
        return self._scaler.transform(log_x)

    def inverse_transform(self, x):
        """
        Function to inverse transform the data with a log transformation and a robust scaler. Used while predicting with the model
        :param x:
        :return:
        """
        inversed = self._scaler.inverse_transform(x)
        return np.expm1(inversed)