import pandas as pd
import numpy as np
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

print("Starting model training and evaluation.")

# Load clean data
CSV_PATH = "dataset/clean_food_prices_som.csv"
df = pd.read_csv(CSV_PATH)

# Setup data for training
X = df.drop(columns=["price", "log_price"])
y = df["price"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train all three models
lr = LinearRegression().fit(X_train, y_train)
rf = RandomForestRegressor(
    n_estimators=100, random_state=42, n_jobs=-1).fit(X_train, y_train)
gbr = GradientBoostingRegressor(
    n_estimators=100, random_state=42).fit(X_train, y_train)

# Predictions
lr_pred = lr.predict(X_test)
rf_pred = rf.predict(X_test)
gbr_pred = gbr.predict(X_test)

# Evaluation function
model_metrics = {}


def get_metrics(y_true, y_pred):
    metrics = {
        "R2": r2_score(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred))
    }
    return {k: round(v, 4) for k, v in metrics.items()}


# Store metrics
model_metrics["linear_regression"] = get_metrics(y_test, lr_pred)
model_metrics["random_forest"] = get_metrics(y_test, rf_pred)
model_metrics["gradient_boosting"] = get_metrics(y_test, gbr_pred)

# Save metrics for the API
with open("models/model_metrics.json", "w") as f:
    json.dump(model_metrics, f, indent=4)
print("\n✅ Metrics saved.")

# Sanity checks (3 minimum required)
test_indices = [5, 100, 200]
print("\nSanity Checks (3 samples):")
for i in test_indices:
    x_one_df = X_test.iloc[[i]]
    y_true = y_test.iloc[i]
    p_lr = float(lr.predict(x_one_df)[0])
    p_rf = float(rf.predict(x_one_df)[0])
    p_gbr = float(gbr.predict(x_one_df)[0])
    print(
        f"Row {i} - Actual: {y_true:,.0f} | LR: {p_lr:,.0f} | RF: {p_rf:,.0f} | GBR: {p_gbr:,.0f}")

# Save models
joblib.dump(lr, "models/lr_food_model.joblib")
joblib.dump(rf, "models/rf_food_model.joblib")
joblib.dump(gbr, "models/gbr_food_model.joblib")
print("\n✅ All models saved and ready.")
