from bs4 import BeautifulSoup
import yfinance as yf
import pandas as pd
import requests

class Stock():
    def __init__(self, stock_ticker):
        self.ticker = stock_ticker
        self.series_data = None
        self.mean_log_return = 0
        self.variance_log_return = 0
        self.min_log_return = 0
        self.max_log_return = 0
        self.skewness_log_return = 0
        self.kurtosis_log_return = 0
    
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

