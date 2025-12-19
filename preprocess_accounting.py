# src/preprocess_accounting.py

import pandas as pd
from sklearn.model_selection import train_test_split
from .db_utils import get_connection
from .config import (
    ACCOUNTING_TABLE,
    ACCOUNTING_TARGET_COLUMN,
    ACCOUNTING_TEST_SIZE,
    ACCOUNTING_RANDOM_STATE,
)

def load_accounting_data():
    conn = get_connection()
    df = pd.read_sql(f"SELECT * FROM {ACCOUNTING_TABLE}", conn)
    conn.close()
    return df

def preprocess_accounting_data():
    df = load_accounting_data()
    print("📌 Accounting data loaded from SQL:", df.shape)

    # Ensure column names are consistent with DB (spaces replaced by _)
    df.columns = [c.replace(" ", "_") for c in df.columns]

    if ACCOUNTING_TARGET_COLUMN not in df.columns:
        raise ValueError(f"TARGET column `{ACCOUNTING_TARGET_COLUMN}` not found in accounting table columns: {df.columns.tolist()}")

    # Try to convert numeric-looking columns
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")

    # Drop rows with missing target
    df = df.dropna(subset=[ACCOUNTING_TARGET_COLUMN])

    # Separate features and target
    X = df.drop(ACCOUNTING_TARGET_COLUMN, axis=1)
    y = df[ACCOUNTING_TARGET_COLUMN]

    # Identify numeric and categorical
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in numeric_cols]

    # Fill missing numeric with median
    for col in numeric_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce")
        X[col] = X[col].fillna(X[col].median())

    # Fill missing categorical with mode
    for col in cat_cols:
        X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else "Unknown")

    # One-hot encode categoricals
    X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=ACCOUNTING_TEST_SIZE,
        random_state=ACCOUNTING_RANDOM_STATE
    )

    print("✔ Accounting preprocessing done.")
    print("   Train:", X_train.shape, "| Test:", X_test.shape)
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    preprocess_accounting_data()
