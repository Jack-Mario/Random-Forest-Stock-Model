import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.holtwinters import SimpleExpSmoothing

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

raw_data = pd.read_csv('volvb_20260201_172407.csv',
                sep=';',
                decimal='.',
                thousands=',',
                parse_dates=['Date'])

data = raw_data.drop(columns=['Bid', 'Ask', 'Average price', 'Turnover', 'Trades'])
#  Date  Opening price  High price  Low price  Closing price  Total volume
print(data.head())
# print(data.shape[0])

train, test = train_test_split(data, test_size=0.2)

a = 0.2
train["High price smooth"] = SimpleExpSmoothing(train["High price"]).fit().fittedvalues
