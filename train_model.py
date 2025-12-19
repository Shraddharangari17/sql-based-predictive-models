# src/train_model.py

import os
import joblib
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score
from .preprocess import preprocess_data
from .config import MODEL_PATH

def train_model():
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    X_train, X_test, y_train, y_test = preprocess_data()

    cat_features = [i for i, col in enumerate(X_train.columns) if X_train[col].dtype == 'object']

    print("🚀 Training CatBoost Model...")
    model = CatBoostRegressor(
        iterations=1500,
        learning_rate=0.05,
        depth=10,
        loss_function='RMSE',
        eval_metric='R2',
        verbose=False
    )

    model.fit(X_train, y_train, cat_features=cat_features, eval_set=(X_test, y_test))

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)

    print(f"📈 Training R² Score: {train_r2:.2f}")
    print(f"⭐ Validation R² Score: {test_r2:.2f}")

    joblib.dump(model, MODEL_PATH)
    print(f"✔ Model saved at: {MODEL_PATH}")

if __name__ == "__main__":
    train_model()
