import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
import statsmodels.api as sm
from matplotlib.ticker import ScalarFormatter



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

# Maps Ranges to ['total', '5km', '10km', '1km']
print(df['RANGE: Range'].unique())
range_map = { '1KM: 1km from coasts': '1km', '5KM: 5km from coasts': '5km', '10KM: 10km from coasts': '10km', '_T: Total population': 'total' }
df['RANGE: Range'] = df['RANGE: Range'].map(range_map)



# Assuming 'df' is your DataFrame
selected_ranges = ['1km', '5km', '10km']


# Create separate histograms for each range
for range_value in selected_ranges:
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 6))
    data_subset = df[df['RANGE: Range'] == range_value]
    plt.bar(data_subset['TIME_PERIOD: Time'], data_subset['OBS_VALUE'], alpha=0.7, color='#C9D4E7', edgecolor='#4878CF', linewidth=2)

    formatter = ScalarFormatter(useMathText=True)
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.ticklabel_format(axis='y', style='plain')
    plt.xlabel('Year')
    plt.ylabel('Population')
    plt.title(f'Population Growth living {range_value} of the Coast', fontweight='bold')
    plt.legend()
    plt.show()

# Count Number of rows
num_rows = len(df)
print(f"Number of rows: {num_rows}")

# Set up the Seaborn style
sns.set(style="whitegrid")

df['TIME_PERIOD: Time'] = pd.to_numeric(df['TIME_PERIOD: Time'])

# Create the plot using Seaborn
plt.figure(figsize=(12, 8))
sns.lineplot(x='TIME_PERIOD: Time', y='OBS_VALUE', hue='RANGE: Range', data=df, marker='o')

# Set plot labels and title
plt.xlabel('Year')
plt.ylabel('Population')
plt.title('Population Growth Over Time for Different Ranges')

plt.gca().yaxis.set_major_formatter(formatter)
plt.ticklabel_format(axis='y', style='plain')

# Show the plot
plt.legend(title='Year', bbox_to_anchor=(1.05, 1), ncol=2)
plt.show()
