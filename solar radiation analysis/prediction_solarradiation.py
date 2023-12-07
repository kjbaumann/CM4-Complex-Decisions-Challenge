import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from helper_functions import read_weather_data
from datetime import datetime

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

y = df['totalsolarradiation']

model = ARIMA(y, order=(1,1,1))
results = model.fit()

print(results.summary())
