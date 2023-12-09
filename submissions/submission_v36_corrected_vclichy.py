# %% [code]
"""
Submission Script.

This script preprocesses data and trains an XGBoost regressor for predicting
bike counts in Paris.

It includes functions to encode weather-related features, handle holidays, add
COVID-related features, and create a date encoder for date-related information.

The final XGBoost model is trained and predictions are saved to a CSV file.

Author: Charles De Cian, Benjamin Cerf
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer
from xgboost import XGBRegressor

# Import the train and test sets
df_train = pd.read_parquet("../input/mdsb-2023/train.parquet")
df_test = pd.read_parquet("../input/mdsb-2023/final_test.parquet")

# Import external weather data
weather_data = pd.read_csv("../input/external-data/external_data.csv")

# Dealing with the anomaly on boulevard de clichy
counters_of_interest = [
    "20 Avenue de Clichy NO-SE", "20 Avenue de Clichy SE-NO"
    ]

# Start of weeks that need to be changed in Clichy NO-SE
weeks_tbchanged_startv1 = [
    "2021-04-08 00:00:00",
    "2021-04-15 00:00:00",
    "2021-04-22 00:00:00",
    "2021-04-29 00:00:00",
    "2021-05-06 00:00:00",
    "2021-05-13 00:00:00",
    "2021-05-20 00:00:00",
    "2021-05-27 00:00:00",
    "2021-06-03 00:00:00",
    "2021-06-10 00:00:00",
    "2021-06-17 00:00:00",
    "2021-06-24 00:00:00",
    "2021-07-01 00:00:00",
    "2021-07-08 00:00:00",
    "2021-07-15 00:00:00"
    ]

# End of weeks that need to be changed in Clichy NO-SE
weeks_tbchanged_endv1 = [
    "2021-04-14 23:00:00",
    "2021-04-21 23:00:00",
    "2021-04-28 23:00:00",
    "2021-05-05 23:00:00",
    "2021-05-12 23:00:00",
    "2021-05-19 23:00:00",
    "2021-05-26 23:00:00",
    "2021-06-02 23:00:00",
    "2021-06-09 23:00:00",
    "2021-06-16 23:00:00",
    "2021-06-23 23:00:00",
    "2021-06-30 23:00:00",
    "2021-07-07 23:00:00",
    "2021-07-14 23:00:00",
    "2021-07-21 23:00:00"
]

# Values of the last "normal" week, that will be copied on detected
# abnormal weeks
mask_values_copiedv1 = (
    df_train["date"] >= "2021-04-01 00:00:00") & (
        df_train["date"] <= "2021-04-07 23:00:00") & (
            df_train["counter_name"] == counters_of_interest[0])

copied_valuesv1 = df_train.loc[mask_values_copiedv1, "log_bike_count"].tolist()

# Replace abnormal values
for start, end in zip(weeks_tbchanged_startv1, weeks_tbchanged_endv1):
    # Convert to tz-naive datetime
    start = pd.to_datetime(start).tz_localize(None)
    end = pd.to_datetime(end).tz_localize(None)

    mask_values_tbchanged = (df_train["date"] >= start) & (
        df_train["date"] <= end) & (
            df_train["counter_name"] == counters_of_interest[0])

    df_train.loc[mask_values_tbchanged, "log_bike_count"] = copied_valuesv1

# Start of weeks that need to be changed in Clichy SE-NO
weeks_tbchanged_startv2 = [
    "2021-03-18 00:00:00",
    "2021-03-25 00:00:00",
    "2021-04-01 00:00:00",
    "2021-04-08 00:00:00",
    "2021-04-15 00:00:00",
    "2021-04-22 00:00:00",
    "2021-04-29 00:00:00",
    "2021-05-06 00:00:00",
    "2021-05-13 00:00:00",
    "2021-05-20 00:00:00",
    "2021-05-27 00:00:00",
    "2021-06-03 00:00:00",
    "2021-06-10 00:00:00",
    "2021-06-17 00:00:00",
    "2021-06-24 00:00:00",
    "2021-07-01 00:00:00",
    "2021-07-08 00:00:00",
    "2021-07-15 00:00:00"
    ]

# End of weeks that need to be changed in Clichy SE-NO
weeks_tbchanged_endv2 = [
    "2021-03-24 23:00:00",
    "2021-03-31 23:00:00",
    "2021-04-07 23:00:00",
    "2021-04-14 23:00:00",
    "2021-04-21 23:00:00",
    "2021-04-28 23:00:00",
    "2021-05-05 23:00:00",
    "2021-05-12 23:00:00",
    "2021-05-19 23:00:00",
    "2021-05-26 23:00:00",
    "2021-06-02 23:00:00",
    "2021-06-09 23:00:00",
    "2021-06-16 23:00:00",
    "2021-06-23 23:00:00",
    "2021-06-30 23:00:00",
    "2021-07-07 23:00:00",
    "2021-07-14 23:00:00",
    "2021-07-21 23:00:00"
]

# Values of the last "normal" week, that will be copied on detected
# abnormal weeks
mask_values_copiedv2 = (
    df_train["date"] >= "2021-03-11 00:00:00") & (
        df_train["date"] <= "2021-03-17 23:00:00") & (
            df_train["counter_name"] == counters_of_interest[1])

copied_valuesv2 = df_train.loc[mask_values_copiedv2, "log_bike_count"].tolist()

# Replace abnormal values
for start, end in zip(weeks_tbchanged_startv2, weeks_tbchanged_endv2):
    # Convert to tz-naive datetime
    start = pd.to_datetime(start).tz_localize(None)
    end = pd.to_datetime(end).tz_localize(None)

    mask_values_tbchanged = (df_train["date"] >= start) & (
        df_train["date"] <= end) & (
            df_train["counter_name"] == counters_of_interest[1])

    df_train.loc[mask_values_tbchanged, "log_bike_count"] = copied_valuesv2

# Determine the target column name, implement X_train, y_train and X_test
_target_column_name = "log_bike_count"
y_train = df_train[_target_column_name]
X_train = df_train.drop(columns=[_target_column_name])
X_test = df_test

# Deal with the external weather data
weather_cols = ['date', 'rr3']
weather_data = weather_data[weather_cols]

# Reshape the data index dealing with the dates
weather_data['date'] = pd.to_datetime(
    weather_data['date'], format='%Y-%m-%d %H:%M:%S'
    )
weather_data = weather_data.drop_duplicates(subset='date')
new_index = pd.date_range(
    start=weather_data['date'].min(), end=weather_data['date'].max(), freq='H'
    )
weather_data = weather_data.set_index('date').reindex(new_index).reset_index()

# Interpolate the rr3 and t columns
columns_to_interpolate = ['rr3']
weather_data[columns_to_interpolate] = weather_data[
    columns_to_interpolate].interpolate(method='linear')
weather_data = weather_data.rename(columns={'index': 'date'})

# Merge the external data on the train and test sets
X_train = pd.merge(X_train, weather_data, how='left', on='date')
X_test = pd.merge(X_test, weather_data, how='left', on='date')


# Deal with the rain by encoding it
def encode_precipitation(value):
    """
    Encode precipitation values.

    Parameters:
    - value (float): The precipitation value.

    Returns:
    - int: 0 if value < 2.5, 1 if value >= 2.5.
    """
    if value < 2.5:
        return 0
    elif value >= 2.5:
        return 1


X_train['precipitations'] = X_train['rr3'].apply(encode_precipitation)
X_test['precipitations'] = X_test['rr3'].apply(encode_precipitation)

# Create a column for holidays
school_holidays = [
    ('2020-10-17', '2020-11-02'),
    ('2020-12-19', '2021-01-04'),
    ('2021-02-20', '2021-03-08'),
    ('2021-04-10', '2021-04-26'),
    ('2021-07-10', '2021-09-01'),
    ('2021-10-23', '2021-11-08'),
    ('2021-12-18', '2022-01-03'),
]

for i, (start, end) in enumerate(school_holidays):
    school_holidays[i] = (pd.to_datetime(start), pd.to_datetime(end))

X_train['holidays'] = 0
X_test['holidays'] = 0

for start, end in school_holidays:
    X_train.loc[
        (X_train['date'] >= start) & (X_train['date'] <= end), 'holidays'] = 1
    X_test.loc[
        (X_test['date'] >= start) & (X_test['date'] <= end), 'holidays'] = 1

# Add a parameter based on COVID measures that were taken on that period
confinement_dates = pd.DataFrame({
    'start': ['2020-03-17', '2020-10-30', '2021-04-03'],
    'end': ['2020-05-11', '2020-12-15', '2021-05-03']
})

curfew_dates = pd.DataFrame({
    'start2': ['2020-10-17', '2020-12-15'],
    'end2': ['2020-12-15', '2021-06-01']
})

confinement_dates['start'] = pd.to_datetime(confinement_dates['start'])
confinement_dates['end'] = pd.to_datetime(confinement_dates['end'])

curfew_dates['start2'] = pd.to_datetime(curfew_dates['start2'])
curfew_dates['end2'] = pd.to_datetime(curfew_dates['end2'])


def add_covid_features(data, confinement_dates, curfew_dates):
    """
    Add COVID-related features to the dataset.

    Parameters:
    - data (DataFrame): The input dataset.
    - confinement_dates (DataFrame): DataFrame containing confinement
    period information.
    - curfew_dates (DataFrame): DataFrame containing curfew period
    information.

    Returns:
    - None: The function modifies the input dataset in place.
    """
    # Create a new column 'periode' initially set to 0
    data['periode'] = 0

    # Go through the confinement periods
    for _, row in confinement_dates.iterrows():
        data.loc[
            (data['date'] >= row['start']) & (data['date'] <= row['end']),
            'periode'
        ] = 2

    # Go through the curfew periods
    for _, row in curfew_dates.iterrows():
        if row['end2'] is not None:
            data.loc[
                (data['date'] >= row['start2']) &
                (data['date'] <= row['end2']) &
                (data['periode'] != 2),
                'periode'
            ] = 1
        else:
            data.loc[
                (data['date'] >= row['start2']) &
                (data['periode'] != 2),
                'periode'
            ] = 1

    # Check if a date is both in confinement and curfew and assign 2
    data['periode'] = data.groupby('date')['periode'].transform('max')


add_covid_features(X_train, confinement_dates, curfew_dates)
add_covid_features(X_test, confinement_dates, curfew_dates)


# Define the date encoder we want to use
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

# Converte site_id to categoridal
X_train['site_id'] = X_train['site_id'].astype('category')
X_test['site_id'] = X_test['site_id'].astype('category')

# Encode the dates
X_train = date_encoder.fit_transform(X_train)
X_test = date_encoder.fit_transform(X_test)

# Columns to be used in the model
selected_columns = [
    'counter_id', 'site_id', 'year', 'month', 'day', 'weekday',
    'hour', 'holidays', 'periode', 'precipitations'
    ]

X_train_selected = X_train[selected_columns]
X_test_selected = X_test[selected_columns]

# Set the categorical columns as type category
categorical_cols = [
    'site_id', 'holidays', 'periode', 'precipitations'
    ]
X_train_selected.loc[:, categorical_cols] = X_train_selected[
    categorical_cols].astype('category')
X_test_selected.loc[:, categorical_cols] = X_test_selected[
    categorical_cols].astype('category')

# Create our regressor
regressor = XGBRegressor(
    learning_rate=0.2, n_estimators=900, enable_categorical=True)

regressor.fit(X_train_selected, y_train)

# Compute the predictions and shaping it into the good format
y_pred = regressor.predict(X_test_selected)
results = pd.DataFrame(
    dict(
        Id=np.arange(y_pred.shape[0]),
        log_bike_count=y_pred,
    )
)
results.to_csv("submission.csv", index=False)
