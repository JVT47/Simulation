import pickle
from returns import Stock
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

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


omxhpi.log_returns = omxhpi.log_returns.sort_values(by='Closing price', ignore_index=True)

index_simulation = np.random.rand(1000,60)
index_simulation = inverse_cdf(index_simulation, np.array(omxhpi.log_returns['Closing price']))
index_simulation = np.sum(index_simulation, axis=1)

for n, stock in enumerate(stock_list):
    stock.log_returns = stock.log_returns.sort_values(by='Closing price', ignore_index=True)

    stock_simulation = np.random.rand(1000, 60)
    stock_simulation = inverse_cdf(stock_simulation, np.array(stock.log_returns['Closing price']))
    stock_simulation = np.sum(stock_simulation, axis=1)

    final_distribution = index_simulation + stock_simulation

    ax = plt.subplot(2,2, n+1)
    ax.hist(final_distribution)
    ax.set_title(stock.name)

plt.show()