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
X_train = date_encoder.fit_transform(X_train)
X_test = date_encoder.fit_transform(X_test)

# Columns to be used in the model
selected_columns = ['counter_id', 'site_id', 'latitude', 'longitude', 'year', 'month', 'day', 'weekday', 'hour']

X_train_selected = X_train[selected_columns]
X_test_selected = X_test[selected_columns]


# Determine categorical columns
categorical_cols = ["counter_id", "site_id"]

X_train[categorical_cols] = X_train[categorical_cols].astype('category')
X_test[categorical_cols] = X_test[categorical_cols].astype('category')

# Create our Pipeline
regressor = XGBRegressor(learning_rate=0.2, n_estimators=1000, enable_categorical=True)

regressor.fit(X_train_selected, y_train)

y_pred = pipeline.predict(X_test_selected)
results = pd.DataFrame(
    dict(
        Id=np.arange(y_pred.shape[0]),
        log_bike_count=y_pred,
    )
)
results.to_csv("submission.csv", index=False)

