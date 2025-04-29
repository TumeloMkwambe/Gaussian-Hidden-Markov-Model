from bs4 import BeautifulSoup
import yfinance as yf
import requests

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