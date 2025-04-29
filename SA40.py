from sklearn.preprocessing import MinMaxScaler
from scipy.stats import norm, tmin, tmax, tmean, tvar, skew, kurtosis
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import yfinance as yf
import seaborn as sns
import pandas as pd
import numpy as np
import requests
import copy

class Stock:
    def __init__(self, stock_ticker):
        self.ticker = stock_ticker
        self.meta_data = yf.Ticker(stock_ticker)
        self.series_data = yf.download(stock_ticker, start='2020-01-01', end='2024-12-31', auto_adjust=True)
        self.mean_log_return = 0
        self.variance_log_return = 0
        self.min_log_return = 0
        self.max_log_return = 0
        self.skewness_log_return = 0
        self.kurtosis_log_return = 0
        

class Index(Stock):
    def __init__(self, stock_ticker):
        super().__init__(stock_ticker)
        self.constituents = []
        self.scrap_data()
        
    def scrap_data(self):
        url = 'https://companiesmarketcap.com/south-africa/largest-companies-in-south-africa-by-market-cap/'
        headers = {
            "User-Agent": "Mozilla/5.0"
        }
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'lxml')
        table = soup.select_one('table.default-table.table.marketcap-table.dataTable')
        tbody = table.find('tbody')
        for tr in tbody.find_all('td', class_='name-td'):
            ticker = tr.find('div', class_='company-code').text
            constituent = Stock(ticker)
            constituent.meta_data = yf.Ticker(constituent.ticker)
            constituent.series_data = yf.download(constituent.ticker, start='2020-01-01', end='2024-12-31', auto_adjust=True)
            self.constituents.append(constituent)
            if len(self.constituents) == 40:
                break

        return 0
    
FTSE_JSE = Index('^J141.JO')


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

AAnalysis = Analysis()
AAnalysis.add_log_return(FTSE_JSE)
FTSE_JSE.series_data

print(f"Minimum of Log Returns: {FTSE_JSE.min_log_return}")
print(f"Maximum of Log Returns: {FTSE_JSE.max_log_return}")
print(f"Mean of Log Returns: {FTSE_JSE.mean_log_return}")
print(f"Variance of Log Returns: {FTSE_JSE.variance_log_return}")
print(f"Skewness of Log Returns: {FTSE_JSE.skewness_log_return}")
print(f"Kurtosis of Log Returns: {FTSE_JSE.kurtosis_log_return}")

AAnalysis.open_close_plot(FTSE_JSE)
AAnalysis.log_return_distribution(FTSE_JSE)


class State:
    def __init__(self, Dataset):
        self.initial_probability = np.random.rand()
        self.forward_probabilities = np.zeros(len(Dataset))
        self.backward_probabilities = np.zeros(len(Dataset))
        self.mean = np.random.rand()
        self.variance = np.random.rand()
        self.state_occupations = np.zeros(len(Dataset))

class Hidden_Markov_Model:
    def __init__(self, Dataset, number_of_states):
        self.number_of_states = number_of_states
        self.Dataset = Dataset
        self.States = np.array([State(Dataset) for i in range(number_of_states)])
        self.transition_matrix = np.random.rand(number_of_states, number_of_states)
        self.transition_matrix = self.transition_matrix / np.sum(self.transition_matrix, axis=1, keepdims=True)
        initial_probabilities = np.array([state.initial_probability for state in self.States])
        for state in self.States:
            state.initial_probability /= np.sum(initial_probabilities)

        self.state_transitions = np.zeros((len(Dataset)-1, number_of_states, number_of_states))
        self.forward_scales = np.zeros(len(Dataset))
        self.log_likelihood = 0

    def normalizing(self, array, t):
        normalizing_factor = 0
        for i in range(self.number_of_states):
            normalizing_factor += getattr(self.States[i], array)[t]
        if normalizing_factor > 0:
            for i in range(self.number_of_states):
                getattr(self.States[i], array)[t] = getattr(self.States[i], array)[t] / normalizing_factor
            if array == 'forward_probabilities':
                self.forward_scales[t] = 1 / normalizing_factor
        else:
            self.forward_scales[t] = 0
            for i in range(self.number_of_states):
                getattr(self.States[i], array)[t] = 0


    def forward_algorithm(self, t):
        for i in range(self.number_of_states):
            forward_sum = 0
            for j in range(self.number_of_states):
                forward_sum += self.States[j].forward_probabilities[t-1] * self.transition_matrix[j][i]
            self.States[i].forward_probabilities[t] = forward_sum * norm.pdf(self.Dataset[t], loc=self.States[i].mean, scale=np.sqrt(self.States[i].variance))
        self.normalizing('forward_probabilities', t)

    def backward_algorithm(self, t):
        for i in range(self.number_of_states):
            backward_sum = 0
            for j in range(self.number_of_states):
                backward_sum += self.transition_matrix[i][j] * norm.pdf(self.Dataset[t+1], loc=self.States[j].mean, scale=np.sqrt(self.States[j].variance)) * self.States[j].backward_probabilities[t+1]
            self.States[i].backward_probabilities[t] = backward_sum
        self.normalizing('backward_probabilities', t)

    def state_occupation(self, t):
        denominator = 0
        for j in range(self.number_of_states):
            denominator += self.States[j].forward_probabilities[t] * self.States[j].backward_probabilities[t]
        if denominator > 0:
            for i in range(self.number_of_states):
                self.States[i].state_occupations[t] = self.States[i].forward_probabilities[t] * self.States[i].backward_probabilities[t] / denominator
        else:
            for i in range(self.number_of_states):
                self.States[i].state_occupations[t] = 0

    def state_transition(self, t):
        denominator = 0
        for k in range(self.number_of_states):
            for l in range(self.number_of_states):
                denominator += self.States[k].forward_probabilities[t] * self.transition_matrix[k][l] * norm.pdf(self.Dataset[t+1], loc=self.States[l].mean, scale=np.sqrt(self.States[l].variance)) * self.States[l].backward_probabilities[t+1]
        if denominator > 0:
            for i in range(self.number_of_states):
                for j in range(self.number_of_states):
                    self.state_transitions[t][i][j] = self.States[i].forward_probabilities[t] * self.transition_matrix[i][j] * norm.pdf(self.Dataset[t+1], loc=self.States[j].mean, scale=np.sqrt(self.States[j].variance)) * self.States[j].backward_probabilities[t+1] / denominator
        else:
            for i in range(self.number_of_states):
                for j in range(self.number_of_states):
                    self.state_transitions[t][i][j] = 0

    def e_step(self):
        for i in range(self.number_of_states):
            self.States[i].forward_probabilities[0] = self.States[i].initial_probability * norm.pdf(self.Dataset[0], loc=self.States[i].mean, scale=np.sqrt(self.States[i].variance))
            self.States[i].backward_probabilities[len(self.Dataset)-1] = 1
        self.normalizing('forward_probabilities', 0)

        for t in range(1, len(self.Dataset)):
            self.forward_algorithm(t)
        for t in range(len(self.Dataset)-2, -1, -1):
            self.backward_algorithm(t)
        for t in range(len(self.Dataset)):
            self.state_occupation(t)
        for t in range(len(self.Dataset)-1):
            self.state_transition(t)

    def m_step(self):
        for i in range(self.number_of_states):
            self.States[i].initial_probability = self.States[i].state_occupations[0]

        for i in range(self.number_of_states):
            denominator = np.sum(self.States[i].state_occupations[:-1])
            if denominator > 0:
                for j in range(self.number_of_states):
                    self.transition_matrix[i][j] = np.sum(self.state_transitions[:, i, j]) / denominator
            else:
                self.transition_matrix[i][:] = 1 / self.number_of_states

        for i in range(self.number_of_states):
            denominator = np.sum(self.States[i].state_occupations)
            if denominator > 0:
                self.States[i].mean = np.sum(self.States[i].state_occupations * self.Dataset) / denominator
            else:
                self.States[i].mean = np.random.rand()

        for i in range(self.number_of_states):
            denominator = np.sum(self.States[i].state_occupations)
            if denominator > 0:
                self.States[i].variance = np.sum(self.States[i].state_occupations * (self.Dataset - self.States[i].mean) ** 2) / denominator
                if self.States[i].variance <= 0:
                    self.States[i].variance = np.random.rand()
            else:
                self.States[i].variance = np.random.rand()

    def baum_welch_algorithm(self, threshold, max_iterations=100):
        self.e_step()
        self.m_step()
        new_log_likelihood = -np.sum(np.log(self.forward_scales + 1e-100))
        difference = float('inf')
        iteration = 0

        while difference > threshold and iteration < max_iterations:
            iteration += 1
            previous_log_likelihood = new_log_likelihood
            self.e_step()
            self.m_step()
            new_log_likelihood = -np.sum(np.log(self.forward_scales + 1e-100))
            difference = np.abs(new_log_likelihood - previous_log_likelihood)
            print(f'Iteration {iteration} . . . Log-Likelihood = {new_log_likelihood}')

        self.log_likelihood = new_log_likelihood

FTSE_JSE.series_data

Dataset = FTSE_JSE.series_data['Log_Return'].dropna().to_numpy()

model = Hidden_Markov_Model(Dataset, 3)

for i in range(model.number_of_states):
    print(f'==================================== {i} =====================================')
    print(f'Initial Probability: {model.States[i].initial_probability}')
    print(f'Mean: {model.States[i].mean}')
    print(f'Variance: {model.States[i].variance}')
    print('\n')
print(f'Transition Matrix: {model.transition_matrix}')
print(f'Log Likelihood: {model.log_likelihood}')


model.baum_welch_algorithm(0.001)


for i in range(model.number_of_states):
    print(f'==================================== {i} =====================================')
    print(f'Initial Probability: {model.States[i].initial_probability}')
    print(f'Mean: {model.States[i].mean}')
    print(f'Variance: {model.States[i].variance}')
    print('\n')
print(f'Transition Matrix: {model.transition_matrix}')
print(f'Log Likelihood: {model.log_likelihood}')

import numpy as np
from hmmlearn import hmm

class HMM_Model:
    def __init__(self, n_components, n_features):
        self.model = hmm.GaussianHMM(n_components=n_components, covariance_type="full", n_iter=100, verbose=True)

    def fit(self, X):
        self.model.fit(X)

    def predict(self, X):
        return self.model.predict(X)

    def get_parameters(self, X):
        print(f'likelihood = {self.model.score(X)}')
        print(f'initial_probabilities = {self.model.startprob_}')
        print(f'means = {self.model.means_}')
        print(f'covariances = {self.model.covars_}')
        print(f'transition_matrix = {self.model.transmat_}')

hmm_model = HMM_Model(n_components=2, n_features=1)
hmm_model.fit(Dataset.reshape(-1, 1))

hmm_model.get_parameters(Dataset.reshape(-1, 1))

for i in range(model.number_of_states):
    print(f'==================================== {i} =====================================')
    print(f'Initial Probability: {model.States[i].initial_probability}')
    print(f'Mean: {model.States[i].mean}')
    print(f'Variance: {model.States[i].variance}')
    print('\n')
print(f'Transition Matrix: {model.transition_matrix}')
print(f'Log Likelihood: {model.log_likelihood}')

