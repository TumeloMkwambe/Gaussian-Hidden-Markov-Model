import numpy as np
import seaborn as sns
import matplotlib.cm as cm
from scipy.stats import  norm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

class Analysis:
    def __init__(self):
        pass

    def price_plot(self, stock, column):
        stock_data = stock.series_data[[column, 'Date']]

        plt.figure(figsize=(12, 6))
        plt.plot(stock_data['Date'], stock_data[column], label=column, color='Green')

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
        plt.xlabel("Date")
        plt.ylabel(column)
        plt.legend()
        plt.title(f"Rolling Mean & Std Deviation of {column}")
        plt.show()

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
                label=f'State {i} ($\\mu$={mean_array[i]:.6f}, $\\sigma^2$={variance_array[i]:.1e})',
                color=colors[i]
            )

        plt.title("Comparison of Normal Distributions")
        plt.xlabel("x")
        plt.ylabel("Probability Density")
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def mean_variance_plot(self, mean_array, variance_array):
        cmap = cm.get_cmap('tab10')
        colors = [cmap(i % 10) for i in range(len(mean_array))]

        plt.figure(figsize=(8, 6))
        plt.scatter(mean_array, variance_array, c=colors, s=100, edgecolors='black')

        for i, (mu, var) in enumerate(zip(mean_array, variance_array)):
            plt.text(mu, var, f'State {i}', fontsize=9, ha='right', va='bottom')

        plt.title("Means vs Variances of Hidden States")
        plt.xlabel("Mean ($\\mu$)")
        plt.ylabel("Variance ($\\sigma^2$)")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def states_plot_points(self, stock, column, state_sequence):
        stock_data = stock.series_data[[column, 'Date']].copy()
        stock_data = stock_data.reset_index(drop=True)

        if len(stock_data) != len(state_sequence):
            raise ValueError("The length of the stock data and the state sequence must be the same.")

        fig, ax = plt.subplots(figsize=(12, 6))

        cmap = cm.get_cmap('tab10')
        unique_states = np.unique(state_sequence)
        colors = {state: cmap(i % 10) for i, state in enumerate(unique_states)}
        legend_handles = {}

        for i in range(len(stock_data)):
            date = stock_data['Date'][i]
            value = stock_data[column][i]
            state = state_sequence[i]
            color = colors[state]
            ax.scatter(date, value, color=color, s=20, label=f"State {state}" if state not in legend_handles else "")
            if state not in legend_handles:
                legend_handles[state] = True

        ax.set_title(f"{stock.ticker} {column} with Hidden States")
        ax.set_xlabel("Date")
        ax.set_ylabel(column)
        ax.legend(title="States")
        ax.grid(True)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        fig.autofmt_xdate()
        fig.tight_layout()
        plt.show()

    def summary_statistics(self, stock):
        print(f"Minimum of Log Return: {stock.min_log_return}")
        print(f"Maximum of Log Return: {stock.max_log_return}")
        print(f"Mean of Log Return: {stock.mean_log_return}")
        print(f"Variance of Log Return: {stock.variance_log_return}")
        print(f"Skewness of Log Return: {stock.skewness_log_return}")
        print(f"Kurtosis of Log Return: {stock.kurtosis_log_return}")
