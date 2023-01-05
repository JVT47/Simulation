import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model

DELIMETER = ";"
DECIMAL = ","

def log_return(file_path: str):
    stock = pd.read_csv(file_path, delimiter=DELIMETER, decimal=DECIMAL)
    stock = stock[["Date", "Closing price"]]
    stock["Closing price"] = stock["Closing price"].apply(np.log)
    stock["Closing price"] = stock["Closing price"].diff(periods=-1)
    return stock.dropna()

def linear_regression(log_returns_X: pd.DataFrame, log_returns_y: pd.DataFrame):
    log_returns_X_y = log_returns_X.merge(log_returns_y, how='inner', on='Date')
    X = np.array(log_returns_X_y["Closing price_x"]).reshape(-1,1)
    y = np.array(log_returns_X_y["Closing price_y"])

    reg = linear_model.LinearRegression()
    reg.fit(X,y)

    residuals = y - reg.predict(X)

    return (reg.coef_[0], reg.intercept_, residuals)

class Stock:
    def __init__(self, file_path: str):
        self.name = file_path.split("'\'")[-1]
        self.log_returns = log_return(file_path)
        self.coef = 1
        self.intercept = 0
        self.residuals = np.array([0])
    
    def linear_reggression(self, other_stock):
        self.coef, self.intercept, self.residuals = linear_regression(other_stock.log_returns, 
                                                                      self.log_returns)



omxhpi = Stock(r'C:\Users\joona\OneDrive\Tiedostot\Simulaatioprojekti\Data\OMXHPI.csv')
siili_solutions = Stock(r'C:\Users\joona\OneDrive\Tiedostot\Simulaatioprojekti\Data\SIILI-2022-01-05-2023-01-05.csv')

siili_solutions.linear_reggression(omxhpi)
print(siili_solutions.residuals)

print(siili_solutions.coef)
print(siili_solutions.intercept)

figure, axis = plt.subplots(1,1)

axis.hist(siili_solutions.residuals)

plt.show()