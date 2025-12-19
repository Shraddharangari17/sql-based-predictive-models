# src/train_finance_project_model.py

import os
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from .preprocess_finance_project import preprocess_finance_project_data
from .config import FINANCE_PROJECT_MODEL_PATH

def train_finance_project_model():
    os.makedirs(os.path.dirname(FINANCE_PROJECT_MODEL_PATH), exist_ok=True)

    X_train, X_test, y_train, y_test = preprocess_finance_project_data()

    print("🚀 Training RandomForestRegressor for Finance (Amount)...")
    model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    mse = mean_squared_error(y_test, y_test_pred)

    print(f"📈 Training R² Score: {train_r2:.3f}")
    print(f"⭐ Validation R² Score: {test_r2:.3f}")
    print(f"📉 Validation MSE: {mse:.3f}")

    joblib.dump({
        "model": model,
        "columns": X_train.columns.tolist()
    }, FINANCE_PROJECT_MODEL_PATH)

    print(f"✔ Finance project model saved at: {FINANCE_PROJECT_MODEL_PATH}")

if __name__ == "__main__":
    train_finance_project_model()
