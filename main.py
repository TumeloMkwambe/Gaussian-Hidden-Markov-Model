from index import Index
from analysis import Analysis
from hidden_markov_model import Hidden_Markov_Model

'''
ANALYSIS
'''
FTSE_JSE = Index('^J141.JO')
AAnalysis = Analysis()
AAnalysis.add_log_return(FTSE_JSE)
AAnalysis.summary_statistics(FTSE_JSE)
AAnalysis.open_close_plot(FTSE_JSE)
AAnalysis.log_return_distribution(FTSE_JSE)

print(FTSE_JSE.series_data)
Dataset = FTSE_JSE.series_data['Log_Return'].dropna().to_numpy()



'''
MODEL
'''

model = Hidden_Markov_Model(Dataset, 3)
model.baum_welch_algorithm(0.001)
model.get_parameters()
