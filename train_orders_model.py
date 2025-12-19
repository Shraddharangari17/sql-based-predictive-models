# src/train_orders_model.py

import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from .preprocess_orders import preprocess_orders_data
from .config import ORDERS_MODEL_PATH

def train_orders_model():
    os.makedirs(os.path.dirname(ORDERS_MODEL_PATH), exist_ok=True)

    X_train, X_test, y_train, y_test, le_target = preprocess_orders_data()

    print("🚀 Training Random Forest Classifier...")
    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_split=2,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    print(f"📈 Training Accuracy   : {train_acc:.3f}")
    print(f"⭐ Validation Accuracy : {test_acc:.3f}")
    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_test_pred, target_names=le_target.classes_))

    joblib.dump({
        "model": model,
        "target_encoder": le_target,
        "columns": X_train.columns.tolist()
    }, ORDERS_MODEL_PATH)

    print("✔ Model saved at:", ORDERS_MODEL_PATH)

if __name__ == "__main__":
    train_orders_model()
