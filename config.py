# src/config.py

DB_CONFIG = {
    "host": "localhost",
    "port": 3306,
    "user": "root",
    "password": "Shraddha@1234",  # 👍 Your MySQL password is here
    "database": "erp_system"
}

# ---------- COMPLIANCE MODEL CONFIG ----------
TRAIN_TABLE = "compliance_risk"
TARGET_COLUMN = "Overall_Risk_Score"
MODEL_PATH = "models/compliance_risk_model.pkl"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ---------- ACCOUNTING MODEL CONFIG ----------
ACCOUNTING_TABLE = "accounting"
ACCOUNTING_TARGET_COLUMN = "net_amount"  # Target column you selected
ACCOUNTING_MODEL_PATH = "models/accounting_model.pkl"
ACCOUNTING_TEST_SIZE = 0.2
ACCOUNTING_RANDOM_STATE = 42
