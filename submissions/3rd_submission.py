import os

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder
from catboost import CatBoostRegressor

# Lecture des fichiers
df_train = pd.read_parquet("../input/mdsb-2023/train.parquet")
df_test = pd.read_parquet("../input/mdsb-2023/final_test.parquet")

_target_column_name = "log_bike_count"

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
date_cols = _encode_dates(X_train[["date"]]).columns.tolist()

#Define the categorical columns we want CatBoost to take into account
categorical_cols = ["counter_id", "site_id"]

X_train = X_train[X_train['log_bike_count'] != 0]
y_train = df_train[_target_column_name]

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

# Créez une nouvelle colonne 'vacances' avec des valeurs par défaut à 0
X_train['vacances'] = 0

# Marquez les jours correspondant aux vacances scolaires avec 1
for debut, fin in vacances_scolaires:
    X_train.loc[(X_train['date'] >= debut) & (X_train['date'] <= fin), 'vacances'] = 1
    
    
#Create our Pipeline
regressor = CatBoostRegressor(iterations= 1000, learning_rate = 0.2, cat_features=categorical_cols)

pipeline = make_pipeline(date_encoder, regressor)
pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
results = pd.DataFrame(
    dict(
        Id=np.arange(y_pred.shape[0]),
        log_bike_count=y_pred,
    )
)
results.to_csv("submission.csv", index=False)

