# src/train_accounting_model.py

import os
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from .preprocess_accounting import preprocess_accounting_data
from .config import ACCOUNTING_MODEL_PATH

def train_accounting_model():
    os.makedirs(os.path.dirname(ACCOUNTING_MODEL_PATH), exist_ok=True)

    X_train, X_test, y_train, y_test = preprocess_accounting_data()

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )

    print("🚀 Training Accounting Model...")
    model.fit(X_train, y_train)

    # Training metrics
    y_train_pred = model.predict(X_train)
    train_r2 = r2_score(y_train, y_train_pred)

    # Validation metrics
    y_test_pred = model.predict(X_test)
    test_r2 = r2_score(y_test, y_test_pred)
    mse = mean_squared_error(y_test, y_test_pred)

    print(f"📈 Training R² Score   : {train_r2:.2f}")
    print(f"⭐ Validation R² Score : {test_r2:.2f}")
    print(f"📉 Validation MSE      : {mse:.2f}")

    joblib.dump(model, ACCOUNTING_MODEL_PATH)
    print(f"✔ Accounting model saved at: {ACCOUNTING_MODEL_PATH}")

if __name__ == "__main__":
    train_accounting_model()
