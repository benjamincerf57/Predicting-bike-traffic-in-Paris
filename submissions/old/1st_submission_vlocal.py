import os

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor

# Lecture des fichiers
train_path = os.path.join("data", "train.parquet")
test_path = os.path.join("data", "test.parquet")

df_train = pd.read_parquet(train_path)
df_test = pd.read_parquet(test_path)

_target_column_name = "log_bike_count"


y_train = df_train[_target_column_name]
y_test = df_test[_target_column_name]
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



date_encoder = FunctionTransformer(_encode_dates)
date_cols = _encode_dates(X_train[["date"]]).columns.tolist()

categorical_encoder = OneHotEncoder(handle_unknown="ignore")
categorical_cols = ["counter_name", "site_name"]

preprocessor = ColumnTransformer(
    [
        ("date", OneHotEncoder(handle_unknown="ignore"), date_cols),
        ("cat", categorical_encoder, categorical_cols),
    ]
)

regressor = CatBoostRegressor(iterations= 1000, learning_rate = 0.2)

pipeline = make_pipeline(date_encoder, preprocessor, regressor)
pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
results = pd.DataFrame(
    dict(
        Id=np.arange(y_pred.shape[0]),
        log_bike_count=y_pred,
    )
)

print(
    f"Train set, RMSE={mean_squared_error(y_train, pipeline.predict(X_train), squared=False):.2f}"
)
print(
    f"Test set, RMSE={mean_squared_error(y_test, pipeline.predict(X_test), squared=False):.2f}"
)

