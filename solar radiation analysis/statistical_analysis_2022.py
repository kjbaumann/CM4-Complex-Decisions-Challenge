# This file analyzes the weather data of 2022 in Papa New Guinnea.
# It provides graphs for total solar radiation each der per m² and waterproduction of a 100m² system

import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from helper_functions import read_weather_data

# Reads weather data from Papa New Guines for the year 2022
df = read_weather_data('./data/Papua New Guinea Weather Data 2022.csv')

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

# Creates a column waterproduction which stores the water produced by a 100 m² system for each day
df['waterproduction'] = df['totalsolarradiation'] / 10

# Plots histogram of totalsolarradiation
sns.set_style("whitegrid")
plt.hist(df['totalsolarradiation'], bins=12, color='#C9D4E7', align='mid', edgecolor='#4878CF', linewidth=2)
plt.title('Total Radiation per day (W/m²)', fontweight='bold')
plt.xlabel('Radiation per day (W/m²)')
plt.ylabel('Frequency')
plt.show()

# Plots a line graph of totalsolarradiation for each day
x = np.arange(365)

fig, ax = plt.subplots()
ax.plot(x, df['totalsolarradiation'], color='#4878CF', lw=2)
ax.fill_between(x, 0, df['totalsolarradiation'], alpha=.3)
plt.title('Total Radiation per day (W/m²)', fontweight='bold')
plt.xlabel('Days')
plt.ylabel('Total Radiation per day in (W/m²)')
plt.show()

# Plots a line graph for waterproduction each day in liters
x = np.arange(365)

fig, ax = plt.subplots()
ax.plot(x, df['waterproduction'], color='#4878CF', lw=2)
ax.fill_between(x, 0, df['waterproduction'], alpha=.3)
plt.title('Total Water Desalination per day (kg)', fontweight='bold')
plt.xlabel('Days')
plt.ylabel('Total Water Desalination per day in kg')
plt.show()

# Descriptive Analysis
# Calculates mean, mode, median of total_sunlight_radiation and waterproduction and stores it as a dataframe
columns = ['Name', 'Unit' 'Mean', 'Median', 'Mode', 'Min', 'Max']

descriptive_df = pd.DataFrame(columns=columns)

new_row_total_radiation = {
    'Name': 'Total Sunlight Radiation per Day',
    'Unit': 'W/m²',
    'Mean': df['totalsolarradiation'].mean(),
    'Median': df['totalsolarradiation'].median(),
    'Mode': df['totalsolarradiation'].mode()[0],
    'Min': df['totalsolarradiation'].min(),
    'Max': df['totalsolarradiation'].max()
    }

descriptive_df = descriptive_df._append(new_row_total_radiation, ignore_index=True)

new_row_total_water_production= {
    'Name': 'Total Water Production per Day',
    'Unit': 'kg',
    'Mean': df['waterproduction'].mean(),
    'Median': df['waterproduction'].median(),
    'Mode': df['waterproduction'].mode()[0],
    'Min': df['waterproduction'].min(),
    'Max': df['waterproduction'].max()
    }

descriptive_df = descriptive_df._append(new_row_total_water_production, ignore_index=True)
descriptive_df.to_csv('./exports/descripitve_analysis.csv')
