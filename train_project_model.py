# src/train_project_model.py

import os
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from .preprocess_project import preprocess_project_data
from .config import PROJECT_MODEL_PATH

def train_project_model():
    os.makedirs(os.path.dirname(PROJECT_MODEL_PATH), exist_ok=True)

    X_train, X_test, y_train, y_test, target_norm = preprocess_project_data()

    print("🚀 Training Project Completion % Model...")
    model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    print(f"📈 Training R² Score: {r2_score(y_train, y_train_pred):.3f}")
    print(f"⭐ Validation R² Score: {r2_score(y_test, y_test_pred):.3f}")
    print(f"📉 Validation MSE: {mean_squared_error(y_test, y_test_pred):.3f}")

    joblib.dump({
        "model": model,
        "columns": X_train.columns.tolist(),
        "target_col": target_norm
    }, PROJECT_MODEL_PATH)

    print(f"✔ Model saved at: {PROJECT_MODEL_PATH}")

if __name__ == "__main__":
    train_project_model()
