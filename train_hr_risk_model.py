import os, joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from .preprocess_hr import preprocess_hr_data
from .config import MODEL_PATH

def train():
    print("🚀 Training HR Salary Regression Model (Random Forest)", flush=True)

    os.makedirs("models", exist_ok=True)

    X_train, X_test, y_train, y_test = preprocess_hr_data()

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_split=8,
        min_samples_leaf=4,
        random_state=42,
        n_jobs=-1
    )

    print("⏳ Fitting model...", flush=True)
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)
    test_mse = mean_squared_error(y_test, test_pred)

    print(f"📈 Training R²   : {train_r2:.3f}")
    print(f"⭐ Validation R² : {test_r2:.3f}")
    print(f"📉 Validation MSE: {test_mse:.2f}")

    joblib.dump(
        {"model": model, "columns": X_train.columns.tolist()},
        MODEL_PATH
    )

    print(f"✔ Model saved at: {MODEL_PATH}")

if __name__ == "__main__":
    train()
