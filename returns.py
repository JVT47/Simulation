import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

DELIMETER = ";"
DECIMAL = ","

def log_return(file_path: str):
    stock = pd.read_csv(file_path, delimiter=DELIMETER, decimal=DECIMAL)
    stock = stock[["Date", "Closing price"]]
    stock["Closing price"] = stock["Closing price"].apply(np.log)
    stock["Closing price"] = stock["Closing price"].diff(periods=-1)
    return stock.dropna()

omxhpi = log_return(r'C:\Users\joona\OneDrive\Tiedostot\Simulaatioprojekti\Data\OMXHPI.csv')
siili_solutions = log_return(r'C:\Users\joona\OneDrive\Tiedostot\Simulaatioprojekti\Data\SIILI-2022-01-05-2023-01-05.csv')

index_and_stock_log_returns = omxhpi.merge(siili_solutions, how='inner', on='Date')

from sklearn import linear_model

X = index_and_stock_log_returns['Closing price_x']
X = np.array(X).reshape(-1,1)
y = index_and_stock_log_returns['Closing price_y']
y = np.array(y)

reg = linear_model.LinearRegression()
reg.fit(X,y)
print(reg.score(X,y))
print(reg.coef_)
print(reg.intercept_)

y_pred = reg.predict(X)

figure, axis = plt.subplots(1,2)

axis[0].scatter(X,y)
axis[0].plot(X, y_pred)

residual = y-y_pred

axis[1].hist(residual)

plt.show()