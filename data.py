from scipy.stats import tmin, tmax, tmean, tvar, skew, kurtosis
import pandas as pd
import numpy as np

class Stock():
    def __init__(self, stock_ticker, dataframe):
        self.ticker = stock_ticker
        self.series_data = dataframe
        self.mean_log_return = 0
        self.variance_log_return = 0
        self.min_log_return = 0
        self.max_log_return = 0
        self.skewness_log_return = 0
        self.kurtosis_log_return = 0
        self.preprocess()
    
    def add_returns(self):
        self.series_data['Return'] = (self.series_data['Price'] - self.series_data['Price'].shift(1)) / self.series_data['Price'].shift(1)
        self.series_data['Log_Return'] = np.log(self.series_data['Price'] / self.series_data['Price'].shift(1))
        log_returns = self.series_data['Log_Return'].dropna()
        self.min_log_return = tmin(log_returns)
        self.max_log_return = tmax(log_returns)
        self.mean_log_return = tmean(log_returns)
        self.variance_log_return = tvar(log_returns)
        self.skewness_log_return = skew(log_returns)
        self.kurtosis_log_return = kurtosis(log_returns)
    
    def preprocess(self):
        self.series_data = self.series_data[::-1].reset_index(drop=True) # reverses historical data
        for column in ['Price', 'Open', 'High', 'Low']: # cast columns with strings into floats
            self.series_data[column] = (
                self.series_data[column]
                .str.replace(',', '', regex=False)
                .astype(float)
            )
        self.series_data['Vol.'] = (
            self.series_data['Vol.']
            .str.replace('M', '', regex=False)
            .astype(float)
        )
        self.series_data['Date'] = pd.to_datetime(self.series_data['Date'])
        self.series_data = self.series_data.drop('Change %', axis='columns')
        self.add_returns()
        self.series_data.dropna(inplace=True)

