from data import Stock
import pandas as pd
from analysis import Analysis
from hidden_markov_model import Hidden_Markov_Model

sa40 = Stock('JTOPI', pd.read_csv('South_Africa_Top_40_Historical_Data.csv'))
print(sa40.series_data)