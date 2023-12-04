import os

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder
import xgboost as xgb
from xgboost import XGBRegressor

# Lecture des fichiers
df_train = pd.read_parquet("../input/mdsb-2023/train.parquet")
df_test = pd.read_parquet("../input/mdsb-2023/final_test.parquet")

_target_column_name = "log_bike_count"

y_train = df_train[_target_column_name]
X_train = df_train.drop(columns=[_target_column_name])
X_test = df_test


def _encode_dates(X):
    X = X.copy()  # modify a copy of X
    # Encode the date information from the DateOfDeparture columns
    X.loc[:, "year"] = X["date"].dt.year
    X.loc[:, "month"] = X["date"].dt.month
    X.loc[:, "day"] = X["date"].dt.day
    X.loc[:, "weekday"] = X["date"].dt.weekday
    X.loc[:, "hour"] = X["date"].dt.hour

    # Finally we can drop the original columns from the dataframe
    return X.drop(columns=["date"])

X_train = X_train.drop(columns=[
    "counter_name", "site_name", "counter_technical_id", "coordinates", "counter_installation_date"
    , "bike_count"])
X_test = X_test.drop(columns=[
    "counter_name", "site_name", "counter_technical_id", "coordinates", "counter_installation_date"])

# Define the encoders we want to use
date_encoder = FunctionTransformer(_encode_dates)
# Encode the dates
date_encoder = FunctionTransformer(_encode_dates)


# Add the weather parameters
weather_data = pd.read_csv('external_data.csv')
colonnes_à_garder = ['date', 't']
weather_data = weather_data[colonnes_à_garder]
weather_data['date'] = pd.to_datetime(weather_data['date'], format='%Y-%m-%d %H:%M:%S')
columns_to_interpolate = ['t']
weather_data[columns_to_interpolate] = weather_data[columns_to_interpolate].interpolate(method='linear')

X_train = pd.merge(X_train, weather_data, how='left', on='date')
X_test = pd.merge(X_train, weather_data, how='left', on='date')

# Create a column for holidays 

vacances_scolaires = [
    ('2020-10-17', '2020-11-02'),  
    ('2020-12-19', '2021-01-04'),  
    ('2021-02-20', '2021-03-08'),  
    ('2021-04-10', '2021-04-26'), 
    ('2021-07-10', '2021-09-01'),  
    ('2021-10-23', '2021-11-08'),  
    ('2021-12-18', '2022-01-03'),  
]

for i, (debut, fin) in enumerate(vacances_scolaires):
    vacances_scolaires[i] = (pd.to_datetime(debut), pd.to_datetime(fin))

X_train['vacances'] = 0
X_test['vacances'] = 0

for debut, fin in vacances_scolaires:
    X_train.loc[(X_train['date'] >= debut) & (X_train['date'] <= fin), 'vacances'] = 1
    X_test.loc[(X_test['date'] >= debut) & (X_test['date'] <= fin), 'vacances'] = 1

# Ajoutons le paramètre COVID 
confinement_dates = pd.DataFrame({
    'debut': ['2020-03-17', '2020-10-30', '2021-04-03'],
    'fin': ['2020-05-11', '2020-12-15', '2021-05-03']
})

couvre_feu_dates = pd.DataFrame({
    'debut2': ['2020-10-17', '2020-12-15'],
    'fin2': ['2020-12-15', '2021-06-01']
})

confinement_dates['debut'] = pd.to_datetime(confinement_dates['debut'])
confinement_dates['fin'] = pd.to_datetime(confinement_dates['fin'])

couvre_feu_dates['debut2'] = pd.to_datetime(couvre_feu_dates['debut2'])
couvre_feu_dates['fin2'] = pd.to_datetime(couvre_feu_dates['fin2'])

def add_covid_features(data, confinement_dates, couvre_feu_dates):
    # Create a new column 'periode' initially set to 0
    data['periode'] = 0

    # Traverse the confinement periods
    for _, row in confinement_dates.iterrows():
        data.loc[
            (data['date'] >= row['debut']) & (data['date'] <= row['fin']),
            'periode'
        ] = 2

    # Traverse the curfew periods
    for _, row in couvre_feu_dates.iterrows():
        if row['fin2'] is not None:
            data.loc[
                (data['date'] >= row['debut2']) & (data['date'] <= row['fin2']) &
                (data['periode'] != 2), 
                'periode'
            ] = 1
        else:
            data.loc[
                (data['date'] >= row['debut2']) &
                (data['periode'] != 2),  
                'periode'
            ] = 1

    # Check if a date is both in confinement and curfew and assign 2
    data['periode'] = data.groupby('date')['periode'].transform('max')

add_covid_features(X_train, confinement_dates, couvre_feu_dates)
add_covid_features(X_test, confinement_dates, couvre_feu_dates)

X_train = date_encoder.fit_transform(X_train)
X_test = date_encoder.fit_transform(X_test)

# Columns to be used in the model
selected_columns = ['counter_id', 'year', 'month', 'day', 'weekday', 'hour', 'vacances', 'periode']

X_train_selected = X_train[selected_columns]
X_test_selected = X_test[selected_columns]


# Determine categorical columns
categorical_cols = ["counter_id", 'vacances', 'periode']

X_train[categorical_cols] = X_train[categorical_cols].astype('category')
X_test[categorical_cols] = X_test[categorical_cols].astype('category')

# Create our Pipeline
regressor = XGBRegressor(learning_rate=0.2, n_estimators=1000, enable_categorical=True)

regressor.fit(X_train_selected, y_train)

y_pred = regressor.predict(X_test_selected)
results = pd.DataFrame(
    dict(
        Id=np.arange(y_pred.shape[0]),
        log_bike_count=y_pred,
    )
)
results.to_csv("submission.csv", index=False)

