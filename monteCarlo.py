import pickle
from returns import Stock
import numpy as np
import pandas as pd
from matplotlib import pyplot

def inverse_cdf(y, sorted_array):
    probability_step = 1/(sorted_array.size-1)
    position = y/probability_step
    index_value = position.astype(int)
    interpolation_precentage = position % 1
    x = (1-interpolation_precentage)*sorted_array[index_value] + interpolation_precentage*sorted_array[index_value+1]
    return x


save_location = r'C:\Users\joona\OneDrive\Tiedostot\Simulaatioprojekti\Data\StockObject.dat'

with open(save_location, 'rb') as f:
    omxhpi = pickle.load(f)
    stock_list = pickle.load(f)


omxhpi.log_returns = omxhpi.log_returns.sort_values(by='Closing price', ignore_index = True)
e = inverse_cdf(np.linspace(0,1,1000, endpoint=False), np.array(omxhpi.log_returns['Closing price']))

fig, ax = pyplot.subplots(2)
ax[0].hist(omxhpi.log_returns['Closing price'])
ax[1].hist(e)

pyplot.show()


