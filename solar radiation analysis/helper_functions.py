import pandas as pd
from datetime import datetime

# This function reads Papa New Guinea waether data, stores it as a dataframe and removes columns that are not needed for analysis and drops rows that dont include solarradiation.
def read_weather_data(csv_fle_path):
    df = pd.read_csv(csv_fle_path)
    df = df.drop(columns=['name',
                      'stations',
                      'uvindex',
                      'visibility',
                      'sealevelpressure',
                      'winddir',
                      'windspeed',
                      'windgust',
                      'preciptype',
                      'snowdepth',
                      'snow',
                      'precipprob',
                      'precip',
                      'dew',
                      'humidity',
                      'feelslike',
                      'feelslikemax',
                      'feelslikemin',
                      'severerisk',
                      'tempmax',
                      'tempmax',
                      'tempmin',
                      'precipcover',
                      'cloudcover',
                      'solarenergy',
                      'moonphase',
                      'conditions',
                      'description',
                      'icon',
                      'temp'], axis=1)

    df = df.dropna(subset=['solarradiation'])
    df = df.reset_index(drop=True)
    return df
