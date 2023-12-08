import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from helper_functions import read_weather_data
from datetime import datetime
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm

# Loads Weather data from 2012 to 2022 and drops columns that are not needed
df = read_weather_data('./data/Papua New Guinea 2002-01-01 to 2022-12-31.csv')

# Creates column hoursofsunlight that stores the hours between sunrise and sunset rounded to the hour
hours_of_sunlight = []
date_format = '%Y-%m-%dT%H:%M:%S'

for sunrise, sunset in zip(df['sunrise'], df['sunset']):
    sr = datetime.strptime(sunrise, date_format)
    ss = datetime.strptime(sunset, date_format)
    hours = (ss -sr).total_seconds() / 3600
    hours_of_sunlight.append(hours)

df['hoursofsunlight'] = hours_of_sunlight

# Creates a column totalsolarradiation which stores the watts per square meter for each day of the year
df['totalsolarradiation'] = df['hoursofsunlight'] * df['solarradiation']

# Creates new column date that maps the strings from the column datetime to datetime objects and drops datetime column
df['date'] = pd.to_datetime(df['datetime'], format='%Y-%m-%d')

# Drops all columns but totalsolarradiation and date
df = df.drop(['solarradiation', 'sunrise', 'sunset', 'hoursofsunlight', 'datetime'], axis=1)

# Sets date column as index
df.set_index('date', inplace=True)

# Testing for stationary data
test_results = adfuller(df['totalsolarradiation'])

# Null Hypothesis: It is non stationary
# Alternative Hypothesis. It is stationary
# Significance Level: 0.05

def adfuller_test(totalsolarradiation):
    result = adfuller(totalsolarradiation)
    labels = ['ADF Test Statistic', 'p_value', '#Lags Used', 'Number of Observations Used']
    for value, label in zip(result, labels):
        print(label + ' : ' + str(value))
    if result[1] < 0.05:
        print("P value is less than 0.05 that means we can reject the null hypothesis(Ho). Therefore we can conclude that data has no unit root and is stationary")
    else:
        print("Weak evidence against null hypothesis that means time series has a unit root which indicates that it is non-stationary ")

adfuller_test(df['totalsolarradiation'])

# Fits Model
model = ARIMA(df['totalsolarradiation'], order=(1,1,1))
results = model.fit()

df['forecast'] = results.predict(start='2023-01-01',end='2023-12-31',dynamic=True)

print(results.summary())
print(df['forecast'])
