import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from helper_functions import read_weather_data
from datetime import datetime
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pmdarima as pm

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
df = df.drop(['solarradiation', 'sunrise', 'sunset', 'hoursofsunlight', 'datetime', 'date'], axis=1)

# Sets date column as index
# df.set_index('date', inplace=True)

# Plots dataframe for initial analysisdf.plot()
plt.show()
df.plot()
plt.show()

# Transforms the data
# df['totalsolarradiation'] = np.log(df['totalsolarradiation'])

# Plots dataframe for analysis
df.plot()
plt.show()

# Splits the dataframe into test and train
msk = (df.index < len(df)-365)

df_train = df[msk].copy()
df_test = df[~msk].copy()

# Checks if data is stantionary
adf_test = adfuller(df_train)
print(f'p-value: {adf_test[1]}')

# Selecting p and q parameters for the ARIMA model
acf_diff = plot_acf(df_train)
plt.show()
pacf_diff = plot_pacf(df_train)
plt.show()

# Model 1: (2,1,0) parameters
model = ARIMA(df_train, order=(2,1,0))
model_fit = model.fit()
print(model_fit.summary())

# Residual Analysis
residuals = model_fit.resid[1:]
fig, ax = plt.subplots(1,2)
residuals.plot(title='Residuals', ax=ax[0])
residuals.plot(title='Density', kind='kde', ax=ax[1])
plt.show()

acf_res = plot_acf(residuals)
pacf_rea = plot_pacf(residuals)
plt.show()

# Forecast with test data
forecast_test = model_fit.forecast(len(df_test))
df['forecast_manual'] = [None]*len(df_train) + list(forecast_test)
df.plot()
plt.show()

# Model 2: Auto-fit with pmdarima
auto_arima = pm.auto_arima(df_train, stepwise=False, seasonal=False)
print(auto_arima)
print(auto_arima.summary())

# Forecast with test data
forecast_test_auto = auto_arima.predict(n_periods=len(df_test))
df['forecast_auto'] = [None]*len(df_train) + list(forecast_test_auto)
df.plot()
plt.show()

# TODO: implement Sarimax
