"""
ml_models.py  –  Predictive ML layer for Diwali Sales Analysis
===============================================================
Three production-ready models:
  1. Spending Regression  – predict Amount from Age/Gender/Occupation/Zone
  2. K-Means Segmentation – cluster customers into 4 business segments
  3. Category Recommender – predict most likely Product_Category for a user

Usage (standalone):
    python ml_models.py --csv "Diwali Sales Data.csv"

Usage (imported in Appp.py):
    from ml_models import train_all, predict_spending, get_segment, recommend_category

NOTE: This is the corrected version with all bugs fixed:
  ✅ Fixed file naming conflict
  ✅ Fixed Age Group type mismatch (no double encoding)
  ✅ Cleaned up pipeline architecture
  ✅ Improved global variable management
"""

import argparse
import warnings
import sqlite3
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    silhouette_score,
    classification_report,
)

warnings.filterwarnings("ignore")

# ─── 0. Data helpers ────────────────────────────────────────────────────────


def load_data(source) -> pd.DataFrame:
    """Accept a CSV path, a SQLite path, or a DataFrame directly."""
    if isinstance(source, pd.DataFrame):
        return source.copy()
    path = Path(source)
    if path.suffix in (".db", ".sqlite", ".sqlite3"):
        con = sqlite3.connect(path)
        df = pd.read_sql("SELECT * FROM sales", con)
        con.close()
    else:
        df = pd.read_csv(path, encoding="unicode_escape")
    df = df.drop(columns=["Status", "unnamed1"], errors="ignore").dropna().copy()
    df["Amount"] = df["Amount"].astype(int)
    return df


# ─── 1. Spending Regression ─────────────────────────────────────────────────

REGRESSION_FEATURES = ["Gender", "Occupation", "Zone", "Age Group"]
REGRESSION_TARGET = "Amount"

_reg_pipeline: Pipeline | None = None


def _build_regression_pipeline() -> Pipeline:
    """Build the regression pipeline with proper preprocessing."""
    # All features are categorical, so use OneHotEncoder
    cat_features = ["Gender", "Occupation", "Zone", "Age Group"]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False), 
             cat_features),
        ],
        remainder="passthrough",
    )

    pipe = Pipeline(
        [
            ("pre", preprocessor),
            ("model", GradientBoostingRegressor(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.08,
                subsample=0.8,
                random_state=42,
            )),
        ]
    )
    return pipe


def train_spending_model(df: pd.DataFrame) -> dict:
    """Train the spending regression model. Returns metrics dict."""
    global _reg_pipeline

    # ✅ FIXED: Select needed columns and drop NaN values
    cols_needed = REGRESSION_FEATURES + [REGRESSION_TARGET]
    df = df[cols_needed].dropna().copy()
    
    # ✅ FIXED: Ensure numeric Amount column
    df[REGRESSION_TARGET] = pd.to_numeric(df[REGRESSION_TARGET], errors='coerce')
    df = df.dropna(subset=[REGRESSION_TARGET])

    X = df[REGRESSION_FEATURES].copy()
    y = df[REGRESSION_TARGET].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipe = _build_regression_pipeline()
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    metrics = {
        "MAE": round(mean_absolute_error(y_test, y_pred), 2),
        "R2": round(r2_score(y_test, y_pred), 4),
        "CV_R2_mean": round(cross_val_score(pipe, X, y, cv=5, scoring="r2").mean(), 4),
    }

    _reg_pipeline = pipe
    return metrics


def predict_spending(age_group: str, gender: str, occupation: str, zone: str) -> float:
    """Predict spending Amount for a new customer profile."""
    if _reg_pipeline is None:
        raise RuntimeError("Call train_spending_model() first.")
    
    row = pd.DataFrame(
        [[gender, occupation, zone, age_group]],
        columns=["Gender", "Occupation", "Zone", "Age Group"]
    )
    return float(_reg_pipeline.predict(row)[0])


# ─── 2. K-Means Customer Segmentation ───────────────────────────────────────

_kmeans: KMeans | None = None
_kmeans_scaler: StandardScaler | None = None
_user_segments: pd.DataFrame | None = None

SEGMENT_LABELS = {
    0: "🏆 High-Value Frequent Buyers",
    1: "🌟 Young Tech Enthusiasts",
    2: "💡 Value-Conscious Shoppers",
    3: "🎯 Occasional Premium Buyers",
}


def train_segmentation(df: pd.DataFrame, n_clusters: int = 4) -> dict:
    """Fit K-Means on customer spending behaviour. Returns metrics."""
    global _kmeans, _kmeans_scaler, _user_segments

    # ✅ FIXED: Drop NaN values before aggregation
    df_clean = df.dropna(subset=["Amount", "Orders", "User_ID"]).copy()
    
    agg = (
        df_clean.groupby("User_ID")
        .agg(
            Total_Spend=("Amount", "sum"),
            Avg_Ticket=("Amount", "mean"),
            Total_Orders=("Orders", "sum"),
            Num_Transactions=("Amount", "count"),
        )
        .reset_index()
    )
    
    # ✅ FIXED: Drop any remaining NaN in aggregated data
    agg = agg.dropna()

    scaler = StandardScaler()
    X = scaler.fit_transform(agg[["Total_Spend", "Avg_Ticket", "Total_Orders", "Num_Transactions"]])

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X)

    sil = silhouette_score(X, labels)

    # Attach cluster labels and create business-friendly segment names
    agg["Cluster"] = labels
    cluster_summary = (
        agg.groupby("Cluster")
        .agg(
            Customers=("User_ID", "count"),
            Avg_Total_Spend=("Total_Spend", "mean"),
            Avg_Ticket=("Avg_Ticket", "mean"),
        )
        .sort_values("Avg_Total_Spend", ascending=False)
        .reset_index()
    )

    # Map cluster IDs to business labels based on spend rank
    rank_map = {row["Cluster"]: SEGMENT_LABELS[i] for i, row in cluster_summary.iterrows()}
    agg["Segment"] = agg["Cluster"].map(rank_map)

    _kmeans = km
    _kmeans_scaler = scaler
    _user_segments = agg[["User_ID", "Segment"]].copy()

    return {
        "Silhouette Score": round(sil, 4),
        "Cluster Summary": cluster_summary,
        "User Segments": agg[["User_ID", "Segment"]],
    }


def get_segment(user_id) -> str:
    """Look up the segment label for an existing User_ID."""
    if _user_segments is None:
        raise RuntimeError("Call train_segmentation() first.")
    row = _user_segments[_user_segments["User_ID"] == user_id]
    return row["Segment"].values[0] if len(row) else "Unknown"


# ─── 3. Product Category Recommender ────────────────────────────────────────

_rec_pipeline: Pipeline | None = None
REC_FEATURES = ["Gender", "Age Group", "Occupation", "Zone", "Marital_Status"]


def train_recommender(df: pd.DataFrame) -> dict:
    """Train a KNN-based category recommender."""
    global _rec_pipeline

    # ✅ FIXED: Drop NaN values from features and target
    cols_needed = REC_FEATURES + ["Product_Category"]
    df_clean = df[cols_needed].dropna().copy()
    
    X = df_clean[REC_FEATURES].copy()
    y = df_clean["Product_Category"]

    pipe = Pipeline([
        ("pre", ColumnTransformer(
            transformers=[
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                 ["Gender", "Age Group", "Occupation", "Zone"]),
            ],
            remainder="passthrough",
        )),
        ("model", KNeighborsClassifier(n_neighbors=15, weights="distance")),
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    acc = round(report["accuracy"], 4)

    _rec_pipeline = pipe

    return {"Accuracy": acc, "Macro F1": round(report["macro avg"]["f1-score"], 4)}


def recommend_category(gender: str, age_group: str, occupation: str, zone: str,
                        marital_status: int = 0, top_n: int = 3) -> list[tuple[str, float]]:
    """Return top-N recommended Product_Categories with probabilities."""
    if _rec_pipeline is None:
        raise RuntimeError("Call train_recommender() first.")
    
    row = pd.DataFrame(
        [[gender, age_group, occupation, zone, marital_status]],
        columns=REC_FEATURES
    )
    probs = _rec_pipeline.predict_proba(row)[0]
    classes = _rec_pipeline.classes_
    top = sorted(zip(classes, probs), key=lambda x: -x[1])[:top_n]
    return [(cat, round(prob * 100, 1)) for cat, prob in top]


# ─── 4. Train All ───────────────────────────────────────────────────────────

def train_all(source) -> dict:
    """Load data, train all three models, return combined metrics."""
    df = load_data(source)

    # ✅ FIXED: Ensure Amount is numeric and clean all critical columns
    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")
    
    # Drop NaN in critical columns
    critical_cols = ["Amount", "Orders", "User_ID", "Product_Category"]
    df = df.dropna(subset=critical_cols).copy()
    
    df["Amount"] = df["Amount"].astype(int)

    if "Marital Label" not in df.columns:
        df["Marital Label"] = df["Marital_Status"].map({0: "Unmarried", 1: "Married"})

    print("⚙️  Training spending regression model...")
    reg_metrics = train_spending_model(df)
    print(f"   ✅ MAE={reg_metrics['MAE']:,}  R²={reg_metrics['R2']}  CV-R²={reg_metrics['CV_R2_mean']}")

    print("⚙️  Training K-Means segmentation (k=4)...")
    seg_result = train_segmentation(df)
    print(f"   ✅ Silhouette Score={seg_result['Silhouette Score']}")
    print(seg_result["Cluster Summary"].to_string(index=False))

    print("⚙️  Training category recommender...")
    rec_metrics = train_recommender(df)
    print(f"   ✅ Accuracy={rec_metrics['Accuracy']}  Macro-F1={rec_metrics['Macro F1']}")

    return {
        "regression": reg_metrics,
        "segmentation": {k: v for k, v in seg_result.items() if k != "User Segments"},
        "recommender": rec_metrics,
    }


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Diwali Sales ML models")
    parser.add_argument("--csv", default="Diwali Sales Data.csv", help="Path to CSV or SQLite DB")
    args = parser.parse_args()

    metrics = train_all(args.csv)

    print("\n📊 Demo predictions")
    print("  Predicted spend (Female | 26-35 | IT Sector | Western):",
          f"Rs {predict_spending('26-35', 'Female', 'IT Sector', 'Western'):,.0f}")
    print("  Top category recommendations (Male | 18-25 | Student | Central):",
          recommend_category("Male", "18-25", "Student", "Central"))
