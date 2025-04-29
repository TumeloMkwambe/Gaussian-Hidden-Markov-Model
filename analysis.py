from scipy.stats import tmin, tmax, tmean, tvar, skew, kurtosis
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class Analysis:
    def __init__(self):
        pass

    def open_close_plot(self, stock):
        stock_data = {}
        stock_data['Open'] = stock.series_data['Open']
        stock_data['Close'] = stock.series_data['Close']
        
        plt.figure(figsize=(12, 6))
        for price, series in stock_data.items():
            plt.plot(series.index, series.values, label=price)
        
        plt.title(f"{stock.ticker} Open and Close Price")
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.tight_layout()
        plt.show()

    def add_log_return(self, stock):
        price_column = 'Close' if 'Close' in stock.series_data.columns else 'Close'
        stock.series_data['Log_Return'] = np.log(stock.series_data[price_column] / stock.series_data[price_column].shift(1))
        log_returns = stock.series_data['Log_Return'].dropna()
        stock.min_log_return = tmin(log_returns)
        stock.max_log_return = tmax(log_returns)
        stock.mean_log_return = tmean(log_returns)
        stock.variance_log_return = tvar(log_returns)
        stock.skewness_log_return = skew(log_returns)
        stock.kurtosis_log_return = kurtosis(log_returns)

    def log_return_distribution(self, stock):
        plt.figure(figsize=(10, 6))
        sns.histplot(stock.series_data['Log_Return'], kde=True, color='mediumseagreen')
        plt.title(f'{stock.ticker} Log Return Distribution with KDE')
        plt.xlabel('Log Returns')
        plt.ylabel('Density')
        plt.grid(True, alpha=0.5)
        plt.show()
        
        plt.figure(figsize=(10, 6))
        sns.ecdfplot(stock.series_data['Log_Return'], color='darkviolet')
        plt.title(f'{stock.ticker} Empirical CDF of Log Returns')
        plt.xlabel('Log Returns')
        plt.ylabel('Cumulative Probability')
        plt.grid(True, alpha=0.5)
        plt.show()
    
    def summary_statistics(self, stock):
        print(f"Minimum of Log Returns: {stock.min_log_return}")
        print(f"Maximum of Log Returns: {stock.max_log_return}")
        print(f"Mean of Log Returns: {stock.mean_log_return}")
        print(f"Variance of Log Returns: {stock.variance_log_return}")
        print(f"Skewness of Log Returns: {stock.skewness_log_return}")
        print(f"Kurtosis of Log Returns: {stock.kurtosis_log_return}")