# modelling_tuning.py

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow
import mlflow.sklearn
import joblib

# Load dataset hasil preprocessing
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

X_train = train_data.drop('price', axis=1)
y_train = train_data['price']
X_test = test_data.drop('price', axis=1)
y_test = test_data['price']

# Hyperparameter tuning grid
n_estimators_options = [50, 100, 150]
max_depth_options = [5, 10, 15]

# Jalankan MLflow manual logging
for n in n_estimators_options:
    for d in max_depth_options:
        with mlflow.start_run():
            model = RandomForestRegressor(n_estimators=n, max_depth=d, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            # Manual Logging
            mlflow.log_param("n_estimators", n)
            mlflow.log_param("max_depth", d)
            mlflow.log_metric("MAE", mae)
            mlflow.log_metric("MSE", mse)
            mlflow.log_metric("R2", r2)

            # Simpan model
            joblib.dump(model, f'model_rf_{n}_{d}.joblib')
            mlflow.log_artifact(f'model_rf_{n}_{d}.joblib')
