"""
App.py  –  Universal Sales Analytics Dashboard (Streamlit)
===========================================================
Works with ANY CSV file by auto-detecting column types.
Also supports the original Diwali Sales Data with full ML features.

Run with:   python -m streamlit run App.py
"""

import sqlite3
import tempfile
import warnings
import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from matplotlib.ticker import FuncFormatter

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted")

# ─── Page config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Universal Sales Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    [data-testid="stMetricValue"] { font-size: 1.4rem !important; }
    h1 { color: #1a1a2e; }
    .stTabs [data-baseweb="tab"] { font-size: 15px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ─── Helpers ─────────────────────────────────────────────────────────────────

def fmt_num(v):
    if pd.isna(v): return "N/A"
    if abs(v) >= 1_000_000: return f"{v/1_000_000:.1f}M"
    if abs(v) >= 1_000: return f"{v/1_000:.1f}K"
    return f"{v:,.1f}"

money_fmt = FuncFormatter(lambda x, _: fmt_num(x))

def clean_axis(ax, title="", xlabel="", ylabel=""):
    ax.set_title(title, fontsize=12, weight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# ─── Column auto-detection ───────────────────────────────────────────────────

def detect_columns(df: pd.DataFrame) -> dict:
    """
    Auto-detect which columns serve which role.
    Returns a dict with keys: numeric, categorical, id, target
    Works with ANY CSV structure.
    """
    cols = df.columns.tolist()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # Guess the main numeric target (largest mean value = likely revenue/sales/amount)
    target = None
    if numeric_cols:
        means = df[numeric_cols].mean()
        target = means.idxmax()

    # Guess ID column (high cardinality string or int col named with 'id'/'ID')
    id_col = None
    for c in cols:
        if "id" in c.lower() and df[c].nunique() > len(df) * 0.3:
            id_col = c
            break

    return {
        "numeric": numeric_cols,
        "categorical": cat_cols,
        "target": target,
        "id": id_col,
        "all": cols,
    }


def is_diwali_dataset(df: pd.DataFrame) -> bool:
    required = {"Amount", "Gender", "Age Group", "Occupation", "Zone", "Product_Category"}
    return required.issubset(set(df.columns))


def load_and_clean(uploaded_file) -> pd.DataFrame:
    """Load any CSV and do basic cleaning."""
    suffix = Path(uploaded_file.name).suffix.lower()

    if suffix == ".csv":
        # Try multiple encodings
        for enc in ["utf-8", "unicode_escape", "latin-1", "cp1252"]:
            try:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding=enc)
                break
            except Exception:
                continue
    elif suffix in (".xlsx", ".xls"):
        df = pd.read_excel(uploaded_file)
    else:
        st.error(f"Unsupported file type: {suffix}. Please upload a CSV or Excel file.")
        st.stop()

    # Basic cleaning
    df = df.dropna(how="all")                          # drop fully empty rows
    df.columns = df.columns.str.strip()                # strip whitespace from headers
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]  # drop unnamed index cols

    # Try converting numeric-looking columns
    for col in df.columns:
        if df[col].dtype == object:
            converted = pd.to_numeric(df[col], errors="coerce")
            if converted.notna().mean() > 0.8:         # >80% parseable → numeric
                df[col] = converted

    return df

# ─── Universal EDA ───────────────────────────────────────────────────────────

def show_universal_eda(df: pd.DataFrame, detected: dict, filters: dict):
    """Works for any CSV. Shows distributions, top values, correlations."""

    numeric_cols = detected["numeric"]
    cat_cols = detected["categorical"]
    target = detected["target"]

    # ── KPI row ──────────────────────────────────────────────────────────────
    kpi_cols = st.columns(min(len(numeric_cols), 4))
    for i, col in enumerate(numeric_cols[:4]):
        kpi_cols[i].metric(col, fmt_num(df[col].sum()), f"avg {fmt_num(df[col].mean())}")

    st.markdown(f"**{len(df):,} rows · {len(df.columns)} columns** after filters")
    st.divider()

    # ── Numeric distributions ─────────────────────────────────────────────────
    if numeric_cols:
        st.subheader("📈 Numeric Distributions")
        n = min(len(numeric_cols), 4)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
        if n == 1: axes = [axes]
        for ax, col in zip(axes, numeric_cols[:n]):
            ax.hist(df[col].dropna(), bins=30, color="#4a90d9", edgecolor="white", alpha=0.85)
            clean_axis(ax, col)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    # ── Target breakdown by each categorical column ───────────────────────────
    if target and cat_cols:
        st.subheader(f"📊 {target} Breakdown by Category")
        n = min(len(cat_cols), 4)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
        if n == 1: axes = [axes]
        for ax, col in zip(axes, cat_cols[:n]):
            top = (df.groupby(col)[target].sum()
                     .sort_values(ascending=False).head(10).reset_index())
            sns.barplot(data=top, x=target, y=col, ax=ax, palette="rocket")
            ax.xaxis.set_major_formatter(money_fmt)
            clean_axis(ax, f"{target} by {col}")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    # ── Correlation heatmap ───────────────────────────────────────────────────
    if len(numeric_cols) >= 2:
        st.subheader("🔥 Correlation Heatmap")
        corr = df[numeric_cols].corr()
        fig, ax = plt.subplots(figsize=(min(len(numeric_cols) * 1.5, 12), 5))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax,
                    center=0, square=True, linewidths=0.5)
        ax.set_title("Numeric Feature Correlations", fontsize=12, weight="bold")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    # ── Value counts for categorical ──────────────────────────────────────────
    if cat_cols:
        st.subheader("📋 Category Value Counts")
        chosen_cat = st.selectbox("Select a column to inspect", cat_cols)
        vc = df[chosen_cat].value_counts().head(20).reset_index()
        vc.columns = [chosen_cat, "Count"]
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.barplot(data=vc, x="Count", y=chosen_cat, palette="YlOrBr", ax=ax)
        clean_axis(ax, f"Top values in '{chosen_cat}'")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    # ── Raw data preview ──────────────────────────────────────────────────────
    with st.expander("🔍 View Raw Data"):
        st.dataframe(df.head(100), use_container_width=True)


# ─── Universal ML ────────────────────────────────────────────────────────────

def show_universal_ml(df: pd.DataFrame, detected: dict):
    """Generic ML: regression on any numeric target + clustering."""
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.cluster import KMeans
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, r2_score, silhouette_score

    numeric_cols = detected["numeric"]
    cat_cols = detected["categorical"]
    target = detected["target"]

    if not target:
        st.warning("No numeric target column detected in your CSV.")
        return

    # Let user pick target and features
    st.markdown("### ⚙️ Configure Models")
    c1, c2 = st.columns(2)
    sel_target = c1.selectbox("Target column (to predict)", numeric_cols,
                               index=numeric_cols.index(target) if target in numeric_cols else 0)
    available_features = [c for c in df.columns if c != sel_target]
    sel_features = c2.multiselect("Feature columns", available_features,
                                   default=available_features[:min(6, len(available_features))])

    if not sel_features:
        st.info("Select at least one feature column.")
        return

    if st.button("🚀 Train Models"):
        with st.spinner("Training… please wait"):
            X = df[sel_features].copy()
            y = df[sel_target].dropna()
            X = X.loc[y.index]

            # Separate numeric and cat features from selection
            sel_num = [c for c in sel_features if c in numeric_cols and c != sel_target]
            sel_cat = [c for c in sel_features if c in cat_cols]

            transformers = []
            if sel_cat:
                transformers.append(("ohe", OneHotEncoder(handle_unknown="ignore",
                                                           sparse_output=False), sel_cat))
            if sel_num:
                transformers.append(("sc", StandardScaler(), sel_num))

            if not transformers:
                st.error("Need at least one usable feature column.")
                return

            pre = ColumnTransformer(transformers, remainder="drop")
            pipe = Pipeline([
                ("pre", pre),
                ("model", GradientBoostingRegressor(n_estimators=150, max_depth=4,
                                                     learning_rate=0.1, random_state=42))
            ])

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)

            mae = mean_absolute_error(y_test, y_pred)
            r2  = r2_score(y_test, y_pred)

        st.success("✅ Regression model trained!")
        m1, m2 = st.columns(2)
        m1.metric("Mean Absolute Error", fmt_num(mae))
        m2.metric("R² Score", f"{r2:.4f}")

        # Actual vs Predicted chart
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.scatter(y_test, y_pred, alpha=0.4, color="#4a90d9", s=20)
        lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
        ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
        clean_axis(ax, "Actual vs Predicted", f"Actual {sel_target}", f"Predicted {sel_target}")
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig); plt.close()

        # Clustering
        st.divider()
        st.markdown("### 🧩 Customer / Row Clustering (K-Means)")
        if sel_num:
            scaler = StandardScaler()
            X_clust = scaler.fit_transform(df[sel_num].dropna())
            k = st.slider("Number of clusters (k)", 2, 8, 4)
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X_clust)
            sil = silhouette_score(X_clust, labels)
            st.metric("Silhouette Score", f"{sil:.4f}",
                      help="Closer to 1.0 = well-separated clusters")

            df_clust = df[sel_num].dropna().copy()
            df_clust["Cluster"] = labels.astype(str)

            fig, ax = plt.subplots(figsize=(8, 5))
            for cl, grp in df_clust.groupby("Cluster"):
                ax.scatter(grp[sel_num[0]],
                           grp[sel_num[1]] if len(sel_num) > 1 else grp[sel_num[0]],
                           label=f"Cluster {cl}", alpha=0.5, s=30)
            ax.set_xlabel(sel_num[0])
            ax.set_ylabel(sel_num[1] if len(sel_num) > 1 else sel_num[0])
            ax.set_title("Cluster Scatter Plot", fontsize=12, weight="bold")
            ax.legend()
            ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig); plt.close()

            with st.expander("📋 Cluster Summary"):
                summary = df_clust.groupby("Cluster")[sel_num].mean().round(2)
                st.dataframe(summary, use_container_width=True)
        else:
            st.info("Select numeric feature columns to enable clustering.")

        # Live prediction
        st.divider()
        st.markdown("### 🔮 Live Prediction")
        st.caption("Enter values to predict the target column")
        input_data = {}
        pcols = st.columns(min(len(sel_features), 4))
        for i, feat in enumerate(sel_features):
            col = pcols[i % len(pcols)]
            if feat in cat_cols:
                opts = df[feat].dropna().unique().tolist()
                input_data[feat] = col.selectbox(feat, sorted(opts))
            else:
                mn = float(df[feat].min())
                mx = float(df[feat].max())
                med = float(df[feat].median())
                input_data[feat] = col.number_input(feat, min_value=mn, max_value=mx, value=med)

        input_df = pd.DataFrame([input_data])
        prediction = pipe.predict(input_df)[0]
        st.info(f"📈 Predicted **{sel_target}**: **{fmt_num(prediction)}**")


# ─── Diwali-specific ML ───────────────────────────────────────────────────────

def show_diwali_ml(df: pd.DataFrame):
    """Full ML suite specifically for the Diwali Sales dataset."""
    try:
        from ml_models import (train_all, predict_spending,
                               train_segmentation, get_segment,
                               recommend_category)
        diwali_ml_available = True
    except ImportError:
        diwali_ml_available = False

    if not diwali_ml_available:
        st.warning("ml_models.py not found in the same folder. Falling back to universal ML.")
        detected = detect_columns(df)
        show_universal_ml(df, detected)
        return

    if st.button("🚀 Train All Models on Filtered Data"):
        with st.spinner("Training… this may take 15–20 seconds"):
            metrics = train_all(df)
            seg_result = train_segmentation(df)
        st.session_state["diwali_trained"] = True
        st.session_state["diwali_seg"] = seg_result
        st.session_state["diwali_metrics"] = metrics
        st.success("All models trained!")

        c1, c2, c3 = st.columns(3)
        c1.metric("Regression MAE", f"{fmt_num(metrics['regression']['MAE'])}")
        c1.metric("Regression R²", metrics["regression"]["R2"])
        c2.metric("Segmentation Silhouette", metrics["segmentation"]["Silhouette Score"])
        c3.metric("Recommender Accuracy", metrics["recommender"]["Accuracy"])
        c3.metric("Recommender Macro-F1", metrics["recommender"]["Macro F1"])

    if st.session_state.get("diwali_trained"):
        st.divider()
        age_opts = ["0-17", "18-25", "26-35", "36-45", "46-50", "51-55", "55+"]

        st.markdown("### 💰 Spending Predictor")
        col_a, col_b, col_c, col_d = st.columns(4)
        p_age  = col_a.selectbox("Age Group", age_opts, index=2)
        p_gen  = col_b.selectbox("Gender", sorted(df["Gender"].unique()))
        p_occ  = col_c.selectbox("Occupation", sorted(df["Occupation"].unique()))
        p_zone = col_d.selectbox("Zone", sorted(df["Zone"].unique()))
        pred = predict_spending(p_age, p_gen, p_occ, p_zone)
        st.info(f"📈 Predicted spend: **Rs {pred:,.0f}**")

        st.divider()
        st.markdown("### 🧩 Customer Segments")
        seg_result = st.session_state["diwali_seg"]
        st.dataframe(seg_result["Cluster Summary"]
                     .rename(columns={"Avg_Total_Spend":"Avg Spend","Avg_Ticket":"Avg Ticket"}),
                     use_container_width=True)
        seg_counts = seg_result["User Segments"]["Segment"].value_counts().reset_index()
        seg_counts.columns = ["Segment","Customers"]
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.barplot(data=seg_counts, x="Customers", y="Segment", palette="rocket", ax=ax)
        clean_axis(ax, "Customers per Segment")
        plt.tight_layout(); st.pyplot(fig); plt.close()

        lookup = st.selectbox("Look up a User_ID", sorted(df["User_ID"].unique().tolist()))
        try:
            st.success(f"User **{lookup}** → **{get_segment(lookup)}**")
        except Exception as e:
            st.error(str(e))

        st.divider()
        st.markdown("### 🛍️ Category Recommender")
        rc1, rc2, rc3, rc4 = st.columns(4)
        r_gen  = rc1.selectbox("Gender ", sorted(df["Gender"].unique()))
        r_age  = rc2.selectbox("Age Group ", age_opts, index=2)
        r_occ  = rc3.selectbox("Occupation ", sorted(df["Occupation"].unique()))
        r_zone = rc4.selectbox("Zone ", sorted(df["Zone"].unique()))
        r_mar  = st.radio("Marital Status", ["Unmarried (0)","Married (1)"], horizontal=True)
        m_val  = 0 if "Unmarried" in r_mar else 1
        recs = recommend_category(r_gen, r_age, r_occ, r_zone, m_val, top_n=5)
        for rank, (cat, prob) in enumerate(recs, 1):
            st.progress(int(prob), text=f"{rank}. **{cat}** — {prob:.1f}%")
    else:
        st.info("Click **Train All Models** above to activate predictions.")


# ─── SQL Tab ─────────────────────────────────────────────────────────────────

def show_sql_tab(df: pd.DataFrame):
    st.subheader("🗄️ Live SQL Query")
    st.caption("Your data is loaded into an in-memory SQLite table called `data`")

    con = sqlite3.connect(":memory:")
    df.to_sql("data", con, if_exists="replace", index=False)

    # Show schema
    with st.expander("📋 Table Schema"):
        schema = pd.DataFrame({"Column": df.columns, "Type": df.dtypes.astype(str).values})
        st.dataframe(schema, use_container_width=True)

    cols_str = ", ".join(df.columns[:3])
    default_sql = f"SELECT {cols_str}\nFROM data\nLIMIT 10;"

    sql_input = st.text_area("SQL Query", value=default_sql, height=150)

    if st.button("▶️ Run Query"):
        try:
            result = pd.read_sql(sql_input, con)
            st.dataframe(result, use_container_width=True)
            st.download_button("⬇️ Download CSV", result.to_csv(index=False).encode(),
                               "query_result.csv", "text/csv")
        except Exception as e:
            st.error(f"SQL Error: {e}")
    con.close()


# ═══════════════════════════════ MAIN APP ════════════════════════════════════

st.title("📊 Universal Sales Analytics Dashboard")
st.caption("Upload any CSV or Excel file — charts and ML models adapt automatically.")

# ─── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("📂 Upload Data")
    uploaded = st.file_uploader("CSV or Excel file", type=["csv", "xlsx", "xls"])

    if not uploaded:
        st.info("Upload any CSV to begin.")
        st.markdown("**Works with:**\n- Sales data\n- Customer records\n- Survey results\n- Any tabular data")
        st.stop()

    with st.spinner("Loading..."):
        df_raw = load_and_clean(uploaded)

    st.success(f"✅ {len(df_raw):,} rows · {len(df_raw.columns)} cols")
    detected = detect_columns(df_raw)

    st.divider()

    # Dynamic filters — show dropdowns for all categorical columns
    st.header("🔍 Filters")
    active_filters = {}
    cat_cols = detected["categorical"]

    # Show up to 4 filter dropdowns
    for col in cat_cols[:4]:
        opts = ["All"] + sorted(df_raw[col].dropna().unique().tolist())
        sel = st.selectbox(col, opts)
        if sel != "All":
            active_filters[col] = sel

    # Numeric range filter for target
    target = detected["target"]
    if target:
        mn = float(df_raw[target].min())
        mx = float(df_raw[target].max())
        rng = st.slider(f"{target} range", mn, mx, (mn, mx))
        active_filters[f"_range_{target}"] = rng

# ─── Apply filters ────────────────────────────────────────────────────────────

df = df_raw.copy()
for col, val in active_filters.items():
    if col.startswith("_range_"):
        tcol = col.replace("_range_", "")
        df = df[(df[tcol] >= val[0]) & (df[tcol] <= val[1])]
    else:
        df = df[df[col] == val]

if df.empty:
    st.warning("No data matches the current filters. Adjust the sidebar.")
    st.stop()

# ─── Tabs ─────────────────────────────────────────────────────────────────────

is_diwali = is_diwali_dataset(df)

if is_diwali:
    st.info("🪔 Diwali Sales Dataset detected — full ML suite enabled!")
    tab_eda, tab_ml, tab_sql = st.tabs(["📊 EDA", "🤖 ML Models", "🗄️ SQL Query"])
else:
    st.info(f"📁 Custom dataset detected — universal analytics mode")
    tab_eda, tab_ml, tab_sql = st.tabs(["📊 EDA", "🤖 ML Models", "🗄️ SQL Query"])

with tab_eda:
    show_universal_eda(df, detected, active_filters)

with tab_ml:
    if is_diwali:
        show_diwali_ml(df)
    else:
        show_universal_ml(df, detected)

with tab_sql:
    show_sql_tab(df)
