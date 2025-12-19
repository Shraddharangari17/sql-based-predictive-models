# src/load_accounting_csv_to_mysql.py

import pandas as pd
from .db_utils import get_connection
from .config import ACCOUNTING_TABLE

CSV_PATH = "data/accounting_data.csv"  # make sure file is here

def create_accounting_table_from_csv():
    df = pd.read_csv(CSV_PATH)

    conn = get_connection()
    cursor = conn.cursor()

    # Drop if already exists
    cursor.execute(f"DROP TABLE IF EXISTS {ACCOUNTING_TABLE}")

    # Create table with all TEXT columns (simple generic schema)
    columns_sql_parts = []
    for col in df.columns:
        col_clean = col.replace(" ", "_")  # remove spaces in column names
        columns_sql_parts.append(f"`{col_clean}` TEXT")
    columns_sql = ", ".join(columns_sql_parts)

    cursor.execute(f"CREATE TABLE {ACCOUNTING_TABLE} ({columns_sql});")
    conn.commit()
    cursor.close()
    conn.close()
    print(f"✔ Table `{ACCOUNTING_TABLE}` created from CSV columns.")


def import_accounting_csv():
    df = pd.read_csv(CSV_PATH)

    # Make column names SQL-safe (no spaces)
    df.columns = [c.replace(" ", "_") for c in df.columns]

    conn = get_connection()
    cursor = conn.cursor()

    cols = ", ".join([f"`{col}`" for col in df.columns])

    for _, row in df.iterrows():
        values = []
        for v in row:
            if pd.isna(v):
                values.append("NULL")
            else:
                v_str = str(v).replace("'", "''")
                values.append(f"'{v_str}'")
        values_sql = ", ".join(values)

        cursor.execute(f"INSERT INTO {ACCOUNTING_TABLE} ({cols}) VALUES ({values_sql});")

    conn.commit()
    cursor.close()
    conn.close()
    print(f"✔ Data imported into `{ACCOUNTING_TABLE}` successfully!")


if __name__ == "__main__":
    create_accounting_table_from_csv()
    import_accounting_csv()
