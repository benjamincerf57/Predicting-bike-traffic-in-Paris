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


y_train = df_train[_target_column_name]
X_train = df_train.drop(columns=[_target_column_name])
X_test = df_test.drop(columns=[_target_column_name])


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




X_train = X_train.drop(columns=["counter_name", "site_name", "counter_technical_id"])
X_test = X_test.drop(columns=["counter_name", "site_name", "counter_technical_id"])

# Define the encoders we want to use
date_encoder = FunctionTransformer(_encode_dates)
date_cols = _encode_dates(X_train[["date"]]).columns.tolist()

#Define the categorical columns we want CatBoost to take into account
categorical_cols = ["counter_id", "site_id"]

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

