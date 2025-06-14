# modelling.py

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow
import mlflow.sklearn
import joblib

# Aktifkan autolog dari MLflow
mlflow.sklearn.autolog()

# Load dataset yang sudah dipisah sebelumnya
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Pisahkan fitur dan target
X_train = train_data.drop('price', axis=1)
y_train = train_data['price']
X_test = test_data.drop('price', axis=1)
y_test = test_data['price']

# Jalankan eksperimen MLflow
with mlflow.start_run():
    # Inisialisasi dan latih model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Prediksi
    y_pred = model.predict(X_test)

    # Evaluasi
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("MAE:", mae)
    print("MSE:", mse)
    print("R2:", r2)

    # Simpan model
    joblib.dump(model, 'model.joblib')