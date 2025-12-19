from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import joblib

from .preprocess_distribution import preprocess_distribution_data
from .config import MODEL_PATH

def train():
    print("🚀 Training Distribution Delay Model (Random Forest)")

    preprocessor, X_train, X_test, y_train, y_test = preprocess_distribution_data()

    model = RandomForestRegressor(
        n_estimators=400,
        max_depth=15,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1
    )

    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipe.fit(X_train, y_train)

    train_pred = pipe.predict(X_train)
    test_pred = pipe.predict(X_test)

    print(f"📈 Training R²   : {r2_score(y_train, train_pred):.3f}")
    print(f"⭐ Validation R² : {r2_score(y_test, test_pred):.3f}")
    print(f"📉 Validation MSE: {mean_squared_error(y_test, test_pred):.2f}")

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)

    print(f"✅ Model saved at: {MODEL_PATH}")

if __name__ == "__main__":
    train()
