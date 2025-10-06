# Admission Chance Prediction Project
# -------------------------------------------------
# Predicts student's chance of admission using machine learning models

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib, os

# Step 1: Load dataset
url = 'https://github.com/ybifoundation/Dataset/raw/main/Admission%20Chance.csv'
admission = pd.read_csv(url)
admission.columns = admission.columns.str.strip()

# Step 2: Define target and features
X = admission.drop(['Serial No', 'Chance of Admit'], axis=1)
y = admission['Chance of Admit']

# Step 3: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=2529)

# Step 4: Initialize models
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(random_state=2529),
    'Gradient Boosting': GradientBoostingRegressor(random_state=2529)
}

# Step 5: Train and evaluate models
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    results[name] = {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2}

results_df = pd.DataFrame(results).T
print("\nModel Evaluation Results:")
print(results_df)

# Step 6: Save best model
best_model_name = results_df['R2'].idxmax()
best_model = models[best_model_name]
joblib.dump(best_model, 'admission_best_model.joblib')
print(f"\n✅ Best model saved successfully: {best_model_name}")
