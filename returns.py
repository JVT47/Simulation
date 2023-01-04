import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

omxhpi = pd.read_csv(r'C:\Users\joona\OneDrive\Tiedostot\Simulaatioprojekti\Data\OMXHPI.csv',
                     delimiter=";",
                     decimal=",")
omxhpi = omxhpi[["Päätös"]]

ln_omxhpi = np.log(omxhpi)

log_return_omxhpi = ln_omxhpi.diff(periods=-1)

print(omxhpi.head())
print(ln_omxhpi.head())
print(log_return_omxhpi.head())

fig, ax = plt.subplots()
ax.hist(log_return_omxhpi)

plt.show()
