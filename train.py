# -*- coding: utf-8 -*-
"""capstone_project.ipynb

## 1. Data preparation

### Load the dataset
"""

# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
import joblib

"""### Load data"""
data = pd.read_csv("car_price_prediction_.csv")
data.columns = data.columns.str.lower().str.replace(' ', '_')
target_label = 'price'
numerical_cols = data._get_numeric_data().columns
categorical_cols = data.columns[data.dtypes == 'object']
numerical_cols = numerical_cols.drop([target_label, 'car_id'])
y = data[[target_label]]


def generate_data(data):
    df_full_train, df_test = train_test_split(
        data, test_size=0.2, random_state=1)
    df_train = df_full_train.reset_index(drop=True)
    df_val = df_test.reset_index(drop=True)
    y_train = df_train[target_label].values
    y_val = df_val[target_label].values
    del df_train['car_id']
    del df_val['car_id']
    del df_train[target_label]
    del df_val[target_label]
    return df_train, df_val, y_train, y_val


X_train, X_test, y_train, y_test = generate_data(data)

"""### Build the Preprocessing Pipeline"""
# Preprocessing for numerical data: Standard scaling
numerical_transformer = StandardScaler()
# Preprocessing for categorical data: One-hot encoding
categorical_transformer = OneHotEncoder(
    drop='first')  # Avoid dummy variable trap
# Combine preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)

"""### Save the pipline"""
params = {'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': 50,
          "random_state": 42, "n_jobs": -1, "objective": "reg:squarederror"}
pipeline = Pipeline(
    steps=[('preprocessor', preprocessor), ('model', XGBRegressor(**params))])
pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
rmse = mean_squared_error(y_test, predictions, squared=False)
r2 = r2_score(y_test, predictions)
print({'RMSE': rmse, 'R²': r2})
joblib.dump(pipeline, "pipeline.joblib")
