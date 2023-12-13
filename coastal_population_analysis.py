import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
import statsmodels.api as sm


# Imports DataFrame as CSV
df = pd.read_csv('./data/Coastal Polulation.csv')

# Keep only rows with 'PG: Papua New Guinea' in the 'GEO_PICT: Pacific Island Countries and territories' column
df = df[df['GEO_PICT: Pacific Island Countries and territories'] == 'PG: Papua New Guinea']

# Display the updated DataFrame
df.head(48)

# Assuming 'df' is your DataFrame
df = df.drop('GEO_PICT: Pacific Island Countries and territories', axis=1)

# Calculates mean for each range
mean_by_range = df.groupby('RANGE: Range')['OBS_VALUE'].mean().reset_index()
print(mean_by_range)

# Assuming 'df' is your DataFrame
selected_ranges = ["_T: Total population", "10KM: 10km from coasts", "5KM: 5km from coasts", "1KM: 1km from coasts"]

# Create separate histograms for each range
for range_value in selected_ranges:
    plt.figure(figsize=(10, 6))
    data_subset = df[df['RANGE: Range'] == range_value]
    plt.bar(data_subset['TIME_PERIOD: Time'], data_subset['OBS_VALUE'], label=f'Range: {range_value}', alpha=0.7)

    plt.xlabel('Year')
    plt.ylabel('Population')
    plt.title(f'Population Over Years for Range: {range_value}')
    plt.legend()
    plt.show()

# Count Number of rows
num_rows = len(df)
print(f"Number of rows: {num_rows}")

# Sample data (replace this with your actual data)
data = {
    'TIME_PERIOD: Time': ['2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021'] * 4,
    'RANGE: Range': ['Total', '5KM', '10KM', '1KM'] * 12,
    'OBS_VALUE': [7108000, 853, 7275326, 7454857, 7634976, 7816780, 8000322, 8185370, 8371613, 8558701, 8746363, 8934475, 9122994,
                  1500499, 1535821, 1573720, 1611743, 1650122, 1688868, 1727932, 1767248, 1806742, 1846357, 1886068, 1925864,
                  2121738, 2171685, 2225275, 2279040, 2333309, 2388096, 2443333, 2498926, 2554772, 2610789, 2666941, 2723214,
                  568640, 582026, 596389, 610798, 625342, 640026, 654830, 669729, 684696, 699709, 714758, 729840]
}


# Set up the Seaborn style
sns.set(style="whitegrid")

df['TIME_PERIOD: Time'] = pd.to_numeric(df['TIME_PERIOD: Time'])

# Create the plot using Seaborn
plt.figure(figsize=(12, 8))
sns.lineplot(x='TIME_PERIOD: Time', y='OBS_VALUE', hue='RANGE: Range', data=df, marker='o')

# Set plot labels and title
plt.xlabel('Year')
plt.ylabel('Population (in millions)')
plt.title('Population Growth Over Time for Different Ranges')

# Format y-axis ticks to display in millions
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x / 1e6:g}M'))

# Show the plot
plt.legend(title='Year', bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)
plt.show()
