"""
data_pipeline.py  –  End-to-End SQL Data Pipeline
===================================================
1. Ingests raw CSV into a SQLite database (tables: sales, products, customers)
2. Runs analytical queries and returns cleaned DataFrames
3. Can be used standalone or imported by app.py / notebooks

Usage:
    python data_pipeline.py --csv "Diwali Sales Data.csv" --db diwali.db
"""

import argparse
import sqlite3
from pathlib import Path

import pandas as pd

# ─── Schema DDL ─────────────────────────────────────────────────────────────

DDL = """
CREATE TABLE IF NOT EXISTS sales (
    row_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id         TEXT,
    cust_name       TEXT,
    product_id      TEXT,
    gender          TEXT,
    age_group       TEXT,
    age             INTEGER,
    marital_status  INTEGER,
    state           TEXT,
    zone            TEXT,
    occupation      TEXT,
    product_category TEXT,
    orders          INTEGER,
    amount          INTEGER
);

CREATE TABLE IF NOT EXISTS dim_customers AS
SELECT DISTINCT
    user_id, cust_name, gender, age_group, age,
    marital_status, state, zone, occupation
FROM sales
WHERE 1=0;

CREATE TABLE IF NOT EXISTS dim_products AS
SELECT DISTINCT product_id, product_category
FROM sales
WHERE 1=0;
"""


# ─── Ingest ─────────────────────────────────────────────────────────────────

def ingest_csv(csv_path: str, db_path: str = "diwali.db") -> None:
    """Clean the raw CSV and load it into SQLite."""
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path, encoding="unicode_escape")
    df = df.drop(columns=["Status", "unnamed1"], errors="ignore").dropna().copy()
    df["Amount"] = df["Amount"].astype(int)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    rename = {
        "user_id": "user_id",
        "cust_name": "cust_name",
        "product_id": "product_id",
        "gender": "gender",
        "age_group": "age_group",
        "age": "age",
        "marital_status": "marital_status",
        "state": "state",
        "zone": "zone",
        "occupation": "occupation",
        "product_category": "product_category",
        "orders": "orders",
        "amount": "amount",
    }

    cols_present = {c for c in rename if c in df.columns}
    df = df[[c for c in rename if c in cols_present]]

    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.executescript(DDL)
    df.to_sql("sales", con, if_exists="replace", index=False)

    # Populate dimension tables
    cur.execute("DELETE FROM dim_customers")
    cur.execute("""
        INSERT INTO dim_customers
        SELECT DISTINCT user_id, cust_name, gender, age_group, age,
               marital_status, state, zone, occupation
        FROM sales
    """)

    cur.execute("DELETE FROM dim_products")
    cur.execute("""
        INSERT INTO dim_products
        SELECT DISTINCT product_id, product_category FROM sales
    """)

    con.commit()
    con.close()
    print(f"✅ Ingested {len(df):,} rows into '{db_path}'  "
          f"(tables: sales, dim_customers, dim_products)")


# ─── Query helpers ───────────────────────────────────────────────────────────

def get_connection(db_path: str = "diwali.db") -> sqlite3.Connection:
    return sqlite3.connect(db_path)


def query(sql: str, db_path: str = "diwali.db") -> pd.DataFrame:
    con = get_connection(db_path)
    df = pd.read_sql(sql, con)
    con.close()
    return df


def load_clean(db_path: str = "diwali.db") -> pd.DataFrame:
    """Return the full sales table as a DataFrame (mirrors original EDA input)."""
    df = query("SELECT * FROM sales", db_path)
    col_map = {
        "user_id": "User_ID", "cust_name": "Cust_name", "product_id": "Product_ID",
        "gender": "Gender", "age_group": "Age Group", "age": "Age",
        "marital_status": "Marital_Status", "state": "State", "zone": "Zone",
        "occupation": "Occupation", "product_category": "Product_Category",
        "orders": "Orders", "amount": "Amount",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    return df


# ─── Analytical queries ──────────────────────────────────────────────────────

QUERIES = {
    "revenue_by_state": """
        SELECT state AS State, SUM(amount) AS Total_Revenue, COUNT(*) AS Transactions
        FROM sales
        GROUP BY state
        ORDER BY Total_Revenue DESC
        LIMIT 10
    """,
    "revenue_by_category": """
        SELECT product_category AS Category, SUM(amount) AS Revenue, SUM(orders) AS Orders
        FROM sales
        GROUP BY product_category
        ORDER BY Revenue DESC
    """,
    "revenue_by_segment": """
        SELECT gender || ' | ' || age_group || ' | ' || occupation AS Segment,
               SUM(amount) AS Revenue, COUNT(DISTINCT user_id) AS Customers
        FROM sales
        GROUP BY gender, age_group, occupation
        ORDER BY Revenue DESC
        LIMIT 10
    """,
    "top_customers": """
        SELECT user_id AS User_ID, cust_name AS Name,
               SUM(amount) AS Total_Spend, SUM(orders) AS Orders
        FROM sales
        GROUP BY user_id, cust_name
        ORDER BY Total_Spend DESC
        LIMIT 10
    """,
    "zone_category_pivot_raw": """
        SELECT zone AS Zone, product_category AS Category, SUM(amount) AS Revenue
        FROM sales
        GROUP BY zone, product_category
    """,
}


def run_all_queries(db_path: str = "diwali.db") -> dict[str, pd.DataFrame]:
    results = {}
    for name, sql in QUERIES.items():
        results[name] = query(sql, db_path)
        print(f"  📋 {name}: {len(results[name])} rows")
    return results


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diwali Sales SQL Pipeline")
    parser.add_argument("--csv", default="Diwali Sales Data.csv")
    parser.add_argument("--db", default="diwali.db")
    args = parser.parse_args()

    ingest_csv(args.csv, args.db)

    print("\n📊 Running analytical queries...")
    results = run_all_queries(args.db)

    print("\nTop 5 states by revenue:")
    print(results["revenue_by_state"].head().to_string(index=False))

    print("\nTop 10 customers:")
    print(results["top_customers"].to_string(index=False))
