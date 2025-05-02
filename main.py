from data import Stock
import pandas as pd
from analysis import Analysis
from hidden_markov_model import Hidden_Markov_Model


sa40 = Stock('JTOPI')
sa40.series_data = pd.read_csv('South_Africa_Top_40_Historical_Data.csv')
sa40.preprocess()

analysis = Analysis()
analysis.add_log_return(sa40)
sa40.series_data.dropna(inplace=True)
analysis.price_plot(sa40)