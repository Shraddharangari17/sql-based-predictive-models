from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import joblib
from .preprocess_warehouse import preprocess_warehouse_data
from .config import WAREHOUSE_MODEL_PATH


def train_warehouse_model():
    print("🚀 Training Warehouse Stock Prediction Model")

    preprocessor, X_train, X_test, y_train, y_test = preprocess_warehouse_data()

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=18,
        min_samples_split=4,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    print(f"📈 Training R²   : {r2_score(y_train, train_pred):.3f}")
    print(f"⭐ Validation R² : {r2_score(y_test, test_pred):.3f}")
    print(f"📉 Validation MSE: {mean_squared_error(y_test, test_pred):.2f}")

    joblib.dump(
        {"model": model, "preprocessor": preprocessor},
        WAREHOUSE_MODEL_PATH
    )

    print(f"✔ Model saved at: {WAREHOUSE_MODEL_PATH}")


if __name__ == "__main__":
    train_warehouse_model()
