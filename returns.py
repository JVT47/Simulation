import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

DELIMETER = ";"
DECIMAL = ","

def log_return(file_path: str):
    stock = pd.read_csv(file_path, delimiter=DELIMETER, decimal=DECIMAL)
    stock = stock[["Pvä", "Päätös"]]
    stock["Päätös"] = stock["Päätös"].apply(np.log)
    stock["Päätös"] = stock["Päätös"].diff(periods=-1)
    return stock

omxhpi = log_return(r'C:\Users\joona\OneDrive\Tiedostot\Simulaatioprojekti\Data\OMXHPI.csv')

print(omxhpi.head())

fig, ax = plt.subplots()
ax.hist(omxhpi)

plt.show()
