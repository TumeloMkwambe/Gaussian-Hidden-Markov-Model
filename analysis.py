from scipy.stats import tmin, tmax, tmean, tvar, skew, kurtosis, norm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import numpy as np

class Analysis:
    def __init__(self):
        pass

    def price_plot(self, stock, column):
        stock_data = stock.series_data[[column, 'Date']]

        plt.figure(figsize=(12, 6))
        plt.plot(stock_data['Date'], stock_data[column], label=column)

        plt.title(f"{stock.ticker} {column}")
        plt.xlabel('Date')
        plt.ylabel(column)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def rolling_plot(self, stock, column):
        rolling_mean = stock.series_data[column].rolling(window=30).mean()
        rolling_std = stock.series_data[column].rolling(window=30).std()

        plt.figure(figsize=(14, 6))
        plt.plot(stock.series_data['Date'], stock.series_data[column], label='Original')
        plt.plot(stock.series_data['Date'], rolling_mean, label='Rolling Mean')
        plt.plot(stock.series_data['Date'], rolling_std, label='Rolling Std')
        plt.legend()
        plt.title(f"Rolling Mean & Std Deviation of {column}")
        plt.show()

    def add_log_return(self, stock):
        price_column = 'Price' if 'Price' in stock.series_data.columns else 'Price'
        stock.series_data['Return'] = stock.series_data[price_column] / stock.series_data[price_column].shift(1)
        stock.series_data['Log_Return'] = np.log(stock.series_data[price_column] / stock.series_data[price_column].shift(1))
        log_returns = stock.series_data['Log_Return'].dropna()
        stock.min_log_return = tmin(log_returns)
        stock.max_log_return = tmax(log_returns)
        stock.mean_log_return = tmean(log_returns)
        stock.variance_log_return = tvar(log_returns)
        stock.skewness_log_return = skew(log_returns)
        stock.kurtosis_log_return = kurtosis(log_returns)

    def distribution(self, stock, column):
        plt.figure(figsize=(10, 6))
        sns.histplot(stock.series_data[column], kde=True, color='green')
        plt.title(f'{stock.ticker} {column} Distribution')
        plt.xlabel(column)
        plt.ylabel('Density')
        plt.grid(True, alpha=0.5)
        plt.show()
        
        plt.figure(figsize=(10, 6))
        sns.ecdfplot(stock.series_data[column], color='darkviolet')
        plt.title(f'{stock.ticker} Empirical CDF of {column}')
        plt.xlabel(column)
        plt.ylabel('Cumulative Probability')
        plt.grid(True, alpha=0.5)
        plt.show()
    
    def normal_distribution(self, x, mean_array, variance_array):
        std_array = np.sqrt(variance_array)
        return [norm.pdf(x, loc=mu, scale=std) for mu, std in zip(mean_array, std_array)]

    def plot_distributions(self, mean_array, variance_array):
        std_array = np.sqrt(variance_array)
        x_min = np.min(mean_array - 4 * std_array)
        x_max = np.max(mean_array + 4 * std_array)
        x = np.linspace(x_min, x_max, 1000)

        pdfs = self.normal_distribution(x, mean_array, variance_array)

        cmap = cm.get_cmap('tab10')
        colors = [cmap(i % 10) for i in range(len(mean_array))]

        for i, pdf in enumerate(pdfs):
            plt.plot(
                x, pdf,
                label=f'Dist {i+1} ($\\mu$={mean_array[i]:.6f}, $\\sigma^2$={variance_array[i]:.1e})',
                color=colors[i]
            )

        plt.title("Comparison of Normal Distributions")
        plt.xlabel("x")
        plt.ylabel("Probability Density")
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def summary_statistics(self, stock):
        print(f"Minimum of Log Return: {stock.min_log_return}")
        print(f"Maximum of Log Return: {stock.max_log_return}")
        print(f"Mean of Log Return: {stock.mean_log_return}")
        print(f"Variance of Log Return: {stock.variance_log_return}")
        print(f"Skewness of Log Return: {stock.skewness_log_return}")
        print(f"Kurtosis of Log Return: {stock.kurtosis_log_return}")