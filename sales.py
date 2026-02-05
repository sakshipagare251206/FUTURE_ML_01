import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────
# THEME / STYLE
# ─────────────────────────────────────────────────────────────────
PALETTE = {
    "navy":   "#1B2845",
    "teal":   "#00B4D8",
    "gold":   "#F4A261",
    "coral":  "#E76F51",
    "slate":  "#457B9D",
    "cream":  "#F1FAEE",
    "green":  "#2A9D8F",
    "purple": "#7B68EE",
}

CSS = f"""
<style>
/* ── global ── */
body {{ font-family: 'Segoe UI', sans-serif; background: {PALETTE['cream']}; }}
.stApp {{ background: {PALETTE['cream']}; }}

/* ── sidebar ── */
[data-testid="stSidebar"] {{
    background: {PALETTE['navy']} !important;
    color: #fff;
}}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3,
[data-testid="stSidebar"] .stMarkdown p {{
    color: #fff !important;
}}
[data-testid="stSidebar"] label {{
    color: #e0e0e0 !important;
}}

/* ── metric cards ── */
.metric-card {{
    background: #fff;
    border-radius: 14px;
    padding: 18px 22px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.08);
    text-align: center;
    border-top: 4px solid {PALETTE['teal']};
}}
.metric-card .label {{
    font-size: 13px;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 6px;
}}
.metric-card .value {{
    font-size: 28px;
    font-weight: 700;
    color: {PALETTE['navy']};
}}
.metric-card .sub {{
    font-size: 12px;
    color: #9ca3af;
    margin-top: 4px;
}}
.metric-card.gold {{ border-top-color: {PALETTE['gold']}; }}
.metric-card.coral {{ border-top-color: {PALETTE['coral']}; }}
.metric-card.green {{ border-top-color: {PALETTE['green']}; }}

/* ── tables ── */
.stDataframe {{ border-radius: 10px; overflow: hidden; }}
table {{ border-radius: 10px; }}

/* ── section headers ── */
.section-hdr {{
    font-size: 20px;
    font-weight: 700;
    color: {PALETTE['navy']};
    border-bottom: 3px solid {PALETTE['teal']};
    padding-bottom: 6px;
    margin-bottom: 14px;
}}

/* ── info box ── */
.info-box {{
    background: #e0f2fe;
    border-left: 4px solid {PALETTE['teal']};
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 16px;
}}
.info-box p {{ margin: 4px 0; color: {PALETTE['navy']}; font-size: 14px; }}

/* ── button overrides ── */
.stButton button {{
    background: {PALETTE['navy']} !important;
    color: #fff !important;
    border-radius: 8px;
    font-weight: 600;
}}
.stButton button:hover {{ background: {PALETTE['slate']} !important; }}

/* hide default streamlit footer */
.reportview-container .main .block-container {{ padding-top: 0.6rem; }}
footer {{ visibility: hidden; }}
</style>
"""


# ─────────────────────────────────────────────────────────────────
# DATA GENERATORS
# ─────────────────────────────────────────────────────────────────
def generate_superstore_data(n_days: int = 730, seed: int = 42) -> pd.DataFrame:
    """Realistic synthetic Superstore-style dataset."""
    np.random.seed(seed)
    dates      = pd.date_range("2022-01-01", periods=n_days, freq="D")
    categories = ["Furniture", "Office Supplies", "Technology"]
    sub_cats   = {
        "Furniture":        ["Chairs", "Tables", "Bookcases", "Shelving"],
        "Office Supplies":  ["Paper", "Labels", "Binders", "Scissors"],
        "Technology":       ["Phones", "Laptops", "Tablets", "Monitors"],
    }
    regions    = ["East", "West", "North", "South"]
    base_sales = {"Furniture": 180, "Office Supplies": 60, "Technology": 250}

    rows = []
    for d in dates:
        cat   = np.random.choice(categories)
        sub   = np.random.choice(sub_cats[cat])
        reg   = np.random.choice(regions)
        base  = base_sales[cat]
        seasonal = 30 * np.sin(2 * np.pi * d.dayofyear / 365)
        trend    = 0.05 * (d - dates[0]).days
        weekend  = 20  if d.weekday() >= 5 else 0
        holiday  = 40  if d.month in [11, 12] else 0
        noise    = np.random.normal(0, 25)
        sales    = max(5, base + seasonal + trend + weekend + holiday + noise)
        quantity = max(1, int(sales / (base * 0.12) + np.random.normal(0, 1)))
        discount = np.random.choice([0, 0, 0, 0.05, 0.1, 0.15, 0.2])
        rows.append({
            "Order Date":    d,
            "Category":      cat,
            "Sub-Category":  sub,
            "Region":        reg,
            "Sales":         round(sales, 2),
            "Quantity":      quantity,
            "Discount":      discount,
        })
    return pd.DataFrame(rows)


def generate_retail_data(n_days: int = 548, seed: int = 7) -> pd.DataFrame:
    """Synthetic Online-Retail-style dataset."""
    np.random.seed(seed)
    dates      = pd.date_range("2022-06-01", periods=n_days, freq="D")
    countries  = ["United Kingdom", "Germany", "France", "Netherlands", "Ireland"]
    descs      = ["WHITE HANGING HEART", "SPOTTED EGG POUCH",
                  "LUNCH BAG STARS", "3D SILVER BALLOON",
                  "TOTE BAG", "CERAMIC MUG", "LANTERN", "CANDLE HOLDER"]
    rows = []
    for d in dates:
        n_orders = np.random.poisson(4)
        for _ in range(max(1, n_orders)):
            price    = np.random.choice([1.25, 2.55, 3.75, 4.50, 6.00, 8.25, 12.00])
            qty      = np.random.randint(1, 20)
            revenue  = round(price * qty + np.random.normal(0, 2), 2)
            revenue  = max(0.5, revenue)
            rows.append({
                "Order Date": d,
                "Description": np.random.choice(descs),
                "Country":     np.random.choice(countries),
                "Sales":       revenue,
                "Quantity":    qty,
                "UnitPrice":   price,
            })
    return pd.DataFrame(rows)


SAMPLE_DATASETS = {
    "Superstore Sales (2 years)": generate_superstore_data,
    "Online Retail (18 months)": generate_retail_data,
}


# ─────────────────────────────────────────────────────────────────
# DATA CLEANING
# ─────────────────────────────────────────────────────────────────
def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Auto-detect date column, coerce types, drop bad rows."""
    df = df.copy()

    # ── find date column ──
    date_col = None
    for c in df.columns:
        if "date" in c.lower() or "time" in c.lower():
            try:
                df[c] = pd.to_datetime(df[c])
                date_col = c
                break
            except Exception:
                continue
    if date_col is None:
        # try first column
        try:
            df[df.columns[0]] = pd.to_datetime(df[df.columns[0]])
            date_col = df.columns[0]
        except Exception:
            pass

    if date_col and date_col != "Order Date":
        df = df.rename(columns={date_col: "Order Date"})

    # ── find / ensure Sales column ──
    sales_col = None
    for c in df.columns:
        if c.lower() in ("sales", "revenue", "amount", "total", "total_sales", "sales_amount"):
            sales_col = c
            break
    if sales_col is None:
        # pick first numeric
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if num_cols:
            sales_col = num_cols[0]
    if sales_col and sales_col != "Sales":
        df = df.rename(columns={sales_col: "Sales"})

    # ── coerce Sales to numeric, drop negatives ──
    df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce")
    df = df.dropna(subset=["Sales"])
    df = df[df["Sales"] > 0]

    # ── Quantity ──
    if "Quantity" in df.columns:
        df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce").fillna(1).astype(int)
    else:
        df["Quantity"] = 1

    # ── Discount ──
    if "Discount" in df.columns:
        df["Discount"] = pd.to_numeric(df["Discount"], errors="coerce").fillna(0)
    else:
        df["Discount"] = 0.0

    # ── Category / Region (optional) ──
    if "Category" not in df.columns:
        df["Category"] = "General"
    if "Region" not in df.columns and "Country" in df.columns:
        df["Region"] = df["Country"]
    elif "Region" not in df.columns:
        df["Region"] = "All"

    df["Order Date"] = pd.to_datetime(df["Order Date"])
    df = df.sort_values("Order Date").reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING  (daily aggregation + lags + calendar)
# ─────────────────────────────────────────────────────────────────
def build_daily_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to daily, then add rich time-series features."""
    agg_dict = {
        "Sales":       ("Sales",    "sum"),
        "Quantity":    ("Quantity", "sum"),
        "AvgDiscount": ("Discount", "mean"),
        "NumOrders":   ("Sales",    "count"),
    }

    # category dummies if multiple categories exist
    cats = df["Category"].unique()
    if len(cats) > 1:
        for c in cats:
            agg_dict[f"Cat_{c}"] = ("Category", lambda x, _c=c: (x == _c).sum())

    daily = df.groupby("Order Date").agg(**agg_dict).reset_index().sort_values("Order Date").reset_index(drop=True)

    # ── calendar features ──
    daily["Month"]      = daily["Order Date"].dt.month
    daily["DayOfWeek"]  = daily["Order Date"].dt.dayofweek
    daily["DayOfYear"]  = daily["Order Date"].dt.dayofyear
    daily["Quarter"]    = daily["Order Date"].dt.quarter
    daily["IsWeekend"]  = (daily["DayOfWeek"] >= 5).astype(int)
    daily["IsHoliday"]  = daily["Month"].isin([11, 12]).astype(int)
    daily["MonthSin"]   = np.sin(2 * np.pi * daily["Month"] / 12)
    daily["MonthCos"]   = np.cos(2 * np.pi * daily["Month"] / 12)
    daily["DaySin"]     = np.sin(2 * np.pi * daily["DayOfYear"] / 365)
    daily["DayCos"]     = np.cos(2 * np.pi * daily["DayOfYear"] / 365)
    # Important: DayNum must be relative to the absolute start of the dataset
    daily["DayNum"]     = (daily["Order Date"] - daily["Order Date"].min()).dt.days

    # ── lags ──
    for lag in [1, 3, 7, 14, 30]:
        daily[f"Lag_{lag}"] = daily["Sales"].shift(lag)

    # ── rolling means ──
    for w in [3, 7, 14, 30]:
        daily[f"Roll_{w}"] = daily["Sales"].shift(1).rolling(w, min_periods=1).mean()

    # ── diffs ──
    daily["Diff_1"] = daily["Sales"].diff(1)
    daily["Diff_7"] = daily["Sales"].diff(7)

    return daily


def get_feature_cols(daily: pd.DataFrame) -> list:
    """Return ordered feature column names present in daily."""
    base = ["Quantity", "AvgDiscount", "NumOrders"]
    cat_cols = sorted([c for c in daily.columns if c.startswith("Cat_")])
    calendar = ["Month", "DayOfWeek", "DayOfYear", "Quarter",
                "IsWeekend", "IsHoliday", "MonthSin", "MonthCos",
                "DaySin", "DayCos", "DayNum"]
    lags   = ["Lag_1", "Lag_3", "Lag_7", "Lag_14", "Lag_30"]
    rolls  = ["Roll_3", "Roll_7", "Roll_14", "Roll_30"]
    diffs  = ["Diff_1", "Diff_7"]
    all_cols = base + cat_cols + calendar + lags + rolls + diffs
    return [c for c in all_cols if c in daily.columns]


# ─────────────────────────────────────────────────────────────────
# MODEL TRAINING + EVALUATION
# ─────────────────────────────────────────────────────────────────
MODEL_ZOO = {
    "Ridge Regression": lambda: Ridge(alpha=1.0),
    "Random Forest":    lambda: RandomForestRegressor(n_estimators=200, max_depth=12,
                                                      min_samples_leaf=3, random_state=42),
    "Gradient Boosting": lambda: GradientBoostingRegressor(n_estimators=200, max_depth=5,
                                                           learning_rate=0.05, subsample=0.8,
                                                           random_state=42),
}


def train_and_evaluate(daily: pd.DataFrame, feature_cols: list, test_frac: float = 0.2):
    """
    Returns dict of results keyed by model name.
    """
    clean = daily.dropna(subset=feature_cols).copy()
    if len(clean) < 10:
        return {}, None, None

    X = clean[feature_cols].values
    y = clean["Sales"].values
    dates = clean["Order Date"].values

    split = int(len(X) * (1 - test_frac))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    test_dates      = dates[split:]

    results = {}
    for name, factory in MODEL_ZOO.items():
        try:
            model = factory()
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            results[name] = {
                "model":      model,
                "mae":        mean_absolute_error(y_test, preds),
                "rmse":       np.sqrt(mean_squared_error(y_test, preds)),
                "r2":         r2_score(y_test, preds),
                "y_test":     y_test,
                "y_pred":     preds,
                "test_dates": test_dates,
                "feature_cols": feature_cols,
            }
        except Exception:
            continue
            
    return results, X_train, y_train


# ─────────────────────────────────────────────────────────────────
# FUTURE FORECAST  (recursive / direct)
# ─────────────────────────────────────────────────────────────────
def recursive_forecast(model, daily: pd.DataFrame, feature_cols: list, n_days: int = 30, start_date_ref=None) -> pd.DataFrame:
    """
    Walk forward n_days using the last known / previously predicted Sales.
    """
    history = daily["Sales"].tolist()
    last_date = daily["Order Date"].max()
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=n_days, freq="D")
    
    # If explicit reference start date not provided, use daily's min
    if start_date_ref is None:
        start_date_ref = daily["Order Date"].min()

    # detect category columns
    cat_cols = [c for c in feature_cols if c.startswith("Cat_")]
    # average category mix from last 30 days
    cat_means = {c: daily[c].tail(30).mean() if c in daily.columns else 0 for c in cat_cols}
    avg_qty      = daily["Quantity"].tail(30).mean()
    avg_disc     = daily["AvgDiscount"].tail(30).mean()
    avg_orders   = daily["NumOrders"].tail(30).mean()

    preds = []
    for fd in future_dates:
        row = {}
        row["Quantity"]    = avg_qty
        row["AvgDiscount"] = avg_disc
        row["NumOrders"]   = avg_orders
        for cc in cat_cols:
            row[cc] = cat_means.get(cc, 0)

        row["Month"]     = fd.month
        row["DayOfWeek"] = fd.dayofweek
        row["DayOfYear"] = fd.dayofyear
        row["Quarter"]   = (fd.month - 1) // 3 + 1
        row["IsWeekend"] = int(fd.dayofweek >= 5)
        row["IsHoliday"] = int(fd.month in [11, 12])
        row["MonthSin"]  = np.sin(2 * np.pi * fd.month / 12)
        row["MonthCos"]  = np.cos(2 * np.pi * fd.month / 12)
        row["DaySin"]    = np.sin(2 * np.pi * fd.dayofyear / 365)
        row["DayCos"]    = np.cos(2 * np.pi * fd.dayofyear / 365)
        # Fix: Calculate DayNum relative to the original start date reference
        row["DayNum"]    = (fd - start_date_ref).days

        # Fill lags from history
        for lag in [1, 3, 7, 14, 30]:
            if f"Lag_{lag}" in feature_cols:
                row[f"Lag_{lag}"] = history[-lag] if len(history) >= lag else history[0]
        
        # Fill rolls from history
        for w in [3, 7, 14, 30]:
            if f"Roll_{w}" in feature_cols:
                row[f"Roll_{w}"] = np.mean(history[-w:]) if len(history) >= w else np.mean(history)
        
        # Fill diffs
        if "Diff_1" in feature_cols:
            row["Diff_1"] = history[-1] - history[-2] if len(history) >= 2 else 0
        if "Diff_7" in feature_cols:
            row["Diff_7"] = history[-1] - history[-8] if len(history) >= 8 else 0

        # Create vector matching the training feature columns exactly
        try:
            vec = np.array([[row[c] for c in feature_cols]])
            p   = max(1.0, float(model.predict(vec)[0]))
        except KeyError:
            # Fallback if a column is missing in row construction
            p = history[-1]
        
        preds.append(p)
        history.append(p)

    # confidence band (±1.5 × last-30 std)
    recent_std = np.std(daily["Sales"].tail(30).values)
    if np.isnan(recent_std) or recent_std == 0:
        recent_std = np.mean(history) * 0.1
    
    ci_half    = 1.5 * recent_std

    return pd.DataFrame({
        "Date":               future_dates,
        "Forecasted Sales":   [round(v, 2) for v in preds],
        "Lower Bound":        [max(0, round(v - ci_half, 2)) for v in preds],
        "Upper Bound":        [round(v + ci_half, 2) for v in preds],
    })


# ─────────────────────────────────────────────────────────────────
# PER-CATEGORY FORECAST
# ─────────────────────────────────────────────────────────────────
def category_forecast(df: pd.DataFrame, n_days: int = 30):
    """Train one GB model per category, return {cat: forecast_df}."""
    out = {}
    cats = df["Category"].unique()
    if len(cats) <= 1:
        return out
    
    # Store global min date for DayNum consistency
    global_start = df["Order Date"].min()

    for cat in cats:
        try:
            sub = df[df["Category"] == cat].copy()
            if len(sub) < 60:
                continue
            sub_daily = build_daily_features(sub)
            fcols     = get_feature_cols(sub_daily)
            sub_clean = sub_daily.dropna(subset=fcols)
            if len(sub_clean) < 40:
                continue
            
            X = sub_clean[fcols].values
            y = sub_clean["Sales"].values
            
            model = GradientBoostingRegressor(n_estimators=150, max_depth=4,
                                            learning_rate=0.05, random_state=42)
            model.fit(X, y)
            out[cat] = recursive_forecast(model, sub_daily, fcols, n_days, start_date_ref=sub["Order Date"].min())
        except Exception:
            continue
            
    return out


# ─────────────────────────────────────────────────────────────────
# PLOTLY CHART HELPERS
# ─────────────────────────────────────────────────────────────────
def chart_actual_vs_predicted(results: dict, best_name: str, test_dates):
    if not results or best_name not in results:
        return go.Figure()
        
    r = results[best_name]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=test_dates, y=r["y_test"],
        mode="lines", name="Actual Sales",
        line=dict(color=PALETTE["navy"], width=2.5)
    ))
    fig.add_trace(go.Scatter(
        x=test_dates, y=r["y_pred"],
        mode="lines", name="Predicted Sales",
        line=dict(color=PALETTE["coral"], width=2.5, dash="dash")
    ))
    fig.update_layout(
        title="<b>Actual vs Predicted Sales</b> (Test Period)",
        xaxis_title="Date", yaxis_title="Sales ($)",
        template="plotly_white", height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=50, r=30, t=60, b=40),
    )
    return fig


def chart_forecast(forecast_df: pd.DataFrame, daily: pd.DataFrame, n_hist: int = 60):
    hist = daily.tail(n_hist)
    fig = go.Figure()
    # history
    fig.add_trace(go.Scatter(
        x=hist["Order Date"], y=hist["Sales"],
        mode="lines", name="Historical Sales",
        line=dict(color=PALETTE["navy"], width=2.2)
    ))
    # forecast
    fig.add_trace(go.Scatter(
        x=forecast_df["Date"], y=forecast_df["Forecasted Sales"],
        mode="lines", name="Forecast",
        line=dict(color=PALETTE["teal"], width=2.8)
    ))
    # confidence band
    fig.add_trace(go.Scatter(
        x=list(forecast_df["Date"]) + list(forecast_df["Date"][::-1]),
        y=list(forecast_df["Upper Bound"]) + list(forecast_df["Lower Bound"][::-1]),
        fill="toself", fillcolor="rgba(0,180,216,0.12)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Confidence Band", showlegend=True
    ))
    
    # ── FIX: Convert Timestamp to numeric (epoch) or python datetime to avoid pandas/plotly conflict ──
    # The error "Addition/subtraction of integers ... with Timestamp" happens here if we pass raw Timestamp.
    if not daily.empty:
        max_date = daily["Order Date"].max()
        # Convert to milliseconds for Plotly date axis
        x_loc = max_date.timestamp() * 1000 
        
        fig.add_vline(x=x_loc, line_dash="dot",
                    line_color=PALETTE["gold"], line_width=2,
                    annotation_text="Forecast Start", annotation_position="top left")
    
    fig.update_layout(
        title="<b>Sales Forecast — Next 30 Days</b>",
        xaxis_title="Date", yaxis_title="Sales ($)",
        template="plotly_white", height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=50, r=30, t=60, b=40),
    )
    return fig


def chart_model_comparison(results: dict):
    if not results:
        return go.Figure()
    names  = list(results.keys())
    maes   = [results[n]["mae"]  for n in names]
    r2s    = [results[n]["r2"]   for n in names]
    colors = [PALETTE["teal"], PALETTE["coral"], PALETTE["gold"]]

    fig = make_subplots(rows=1, cols=2, subplot_titles=("MAE (lower = better)", "R² Score (higher = better)"))
    fig.add_trace(go.Bar(x=names, y=maes, marker_color=colors, showlegend=False), row=1, col=1)
    fig.add_trace(go.Bar(x=names, y=r2s,  marker_color=colors, showlegend=False), row=1, col=2)
    fig.update_layout(template="plotly_white", height=320,
                      margin=dict(l=40, r=30, t=60, b=40))
    fig.update_yaxes(title_text="MAE ($)", row=1, col=1)
    fig.update_yaxes(title_text="R² Score", row=1, col=2)
    return fig


def chart_category_breakdown(forecast_cats: dict):
    if not forecast_cats:
        return None
    fig = go.Figure()
    colors = [PALETTE["teal"], PALETTE["coral"], PALETTE["gold"], PALETTE["green"]]
    for i, (cat, fdf) in enumerate(forecast_cats.items()):
        fig.add_trace(go.Scatter(
            x=fdf["Date"], y=fdf["Forecasted Sales"],
            mode="lines", name=cat,
            line=dict(color=colors[i % len(colors)], width=2.2)
        ))
    fig.update_layout(
        title="<b>Category-Level Sales Forecast</b>",
        xaxis_title="Date", yaxis_title="Sales ($)",
        template="plotly_white", height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=50, r=30, t=60, b=40),
    )
    return fig


def chart_monthly_trend(df: pd.DataFrame):
    df2 = df.copy()
    df2["YearMonth"] = df2["Order Date"].dt.to_period("M")
    monthly = df2.groupby("YearMonth")["Sales"].sum().reset_index()
    monthly["YearMonth"] = monthly["YearMonth"].astype(str)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=monthly["YearMonth"], y=monthly["Sales"],
        marker_color=PALETTE["slate"],
        marker_line_color=PALETTE["navy"], marker_line_width=0.5
    ))
    fig.update_layout(
        title="<b>Monthly Sales Trend</b>",
        xaxis_title="Month", yaxis_title="Total Sales ($)",
        template="plotly_white", height=340,
        xaxis_tickangle=-45,
        margin=dict(l=50, r=30, t=60, b=60),
    )
    return fig


def chart_feature_importance(model, feature_cols: list):
    if not hasattr(model, "feature_importances_"):
        return None
    imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=True)
    top = imp.tail(12)
    fig = go.Figure(go.Bar(
        y=top.index.tolist(), x=top.values.tolist(),
        orientation="h",
        marker_color=PALETTE["teal"],
        marker_line_color=PALETTE["navy"], marker_line_width=0.8
    ))
    fig.update_layout(
        title="<b>Top Feature Importances</b>",
        xaxis_title="Importance", template="plotly_white",
        height=340, margin=dict(l=120, r=30, t=60, b=40),
    )
    return fig


def chart_weekday_pattern(df: pd.DataFrame):
    df2 = df.copy()
    df2["DayName"] = df2["Order Date"].dt.day_name()
    order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    wk = df2.groupby("DayName")["Sales"].mean().reindex(order)
    fig = go.Figure(go.Bar(
        x=wk.index.tolist(), y=wk.values.tolist(),
        marker_color=[PALETTE["slate"] if i < 5 else PALETTE["gold"] for i in range(7)],
    ))
    fig.update_layout(
        title="<b>Average Sales by Day of Week</b>",
        yaxis_title="Avg Sales ($)", template="plotly_white",
        height=300, margin=dict(l=50, r=30, t=55, b=40),
    )
    return fig


# ─────────────────────────────────────────────────────────────────
# CSV DOWNLOAD HELPER
# ─────────────────────────────────────────────────────────────────
def csv_download(df: pd.DataFrame, label: str, filename: str):
    csv_bytes = df.to_csv(index=False).encode()
    st.download_button(label=label, data=csv_bytes,
                       file_name=filename, mime="text/csv")


# ─────────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="Sales & Demand Forecasting",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CSS, unsafe_allow_html=True)

    # ── SIDEBAR ────────────────────────────────────────────────
    st.sidebar.markdown("## 📈 Sales Forecasting", unsafe_allow_html=False)
    st.sidebar.markdown("---")

    # Data source selector
    st.sidebar.markdown("### 📂 Data Source")
    data_choice = st.sidebar.selectbox(
        "Choose a dataset",
        ["— Upload your CSV —"] + list(SAMPLE_DATASETS.keys())
    )

    uploaded_file = None
    if data_choice == "— Upload your CSV —":
        uploaded_file = st.sidebar.file_uploader(
            "Upload a sales CSV file",
            type=["csv"],
            help="Must contain a date column and a numeric sales / revenue column."
        )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ Settings")

    forecast_days = st.sidebar.slider("Forecast Horizon (days)", min_value=7, max_value=90, value=30, step=7)
    test_pct      = st.sidebar.slider("Test Split %", min_value=10, max_value=40, value=20, step=5)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📌 Quick Nav")
    nav_options = ["📊 Overview", "🔍 EDA", "🤖 Model Results", "🚀 Forecast", "📁 Category Drill-Down", "💡 Business Summary"]
    active_page = st.sidebar.radio("Go to", nav_options)

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "<div style='font-size:11px;color:#aaa;text-align:center;'>"
        "Sales Forecasting Dashboard v1.1<br>"
        "scikit-learn · Plotly · Streamlit"
        "</div>", unsafe_allow_html=True
    )

    # ── LOAD DATA ──────────────────────────────────────────────
    df_raw = None
    if data_choice != "— Upload your CSV —":
        df_raw = SAMPLE_DATASETS[data_choice]()
    elif uploaded_file is not None:
        try:
            df_raw = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            st.stop()

    if df_raw is None:
        # ── LANDING PAGE ──
        col1, col2 = st.columns([1.2, 1])
        with col1:
            st.markdown("""
            <div style="padding: 40px 0;">
              <h1 style="font-size:38px; color:#1B2845; margin-bottom:6px;">📈 Sales & Demand<br>Forecasting Dashboard</h1>
              <p style="color:#457B9D; font-size:18px; margin-bottom:20px;">
                Predict your future sales with ML-powered models and present results your stakeholders will understand.
              </p>
              <div class="info-box">
                <p><b>👉 To get started:</b></p>
                <p>• Pick a <b>sample dataset</b> from the sidebar, or</p>
                <p>• <b>Upload your own CSV</b> with a date + sales column</p>
              </div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div style="background:#fff;border-radius:16px;padding:28px;box-shadow:0 3px 18px rgba(0,0,0,0.08); margin-top:40px;">
              <h3 style="color:#1B2845; margin-top:0;">✨ What This Dashboard Does</h3>
              <p style="color:#374151;line-height:1.7;">
                <b>1.</b> Cleans & validates your data automatically<br>
                <b>2.</b> Engineers 28+ time-series features<br>
                <b>3.</b> Trains & compares 3 ML models<br>
                <b>4.</b> Generates a <b>30-day recursive forecast</b> with confidence bands<br>
                <b>5.</b> Breaks down forecasts by product category<br>
                <b>6.</b> Produces a <b>business-ready summary</b>
              </p>
            </div>
            """, unsafe_allow_html=True)
        st.stop()

    # ── CLEAN ──────────────────────────────────────────────────
    df = clean_dataframe(df_raw)
    if df.empty:
        st.error("❌ No valid data found. Please check your CSV file (Date + Sales columns required).")
        st.stop()

    # ── FEATURE ENGINEERING ────────────────────────────────────
    daily       = build_daily_features(df)
    feature_cols = get_feature_cols(daily)
    daily_clean = daily.dropna(subset=feature_cols).copy()

    # ── TRAIN MODELS ───────────────────────────────────────────
    if len(daily_clean) < 20:
        st.warning("⚠️ Dataset is too small for meaningful machine learning. Please upload at least 2-3 months of data.")
        st.stop()

    results, X_train, y_train = train_and_evaluate(daily_clean, feature_cols, test_pct / 100)
    
    if not results:
        st.error("❌ Model training failed. Ensure your data has sufficient variability.")
        st.stop()

    # pick best by R²
    best_name = max(results, key=lambda k: results[k]["r2"])
    best_res  = results[best_name]

    # ── FORECAST ───────────────────────────────────────────────
    best_model   = best_res["model"]
    # Pass original daily dataframe to get correct global min date for DayNum
    forecast_df  = recursive_forecast(best_model, daily_clean, feature_cols, forecast_days, start_date_ref=daily["Order Date"].min())
    cat_forecasts = category_forecast(df, forecast_days)

    # ── TEST DATES ─────────────────────────────────────────────
    test_dates = best_res["test_dates"]

    # ═══════════════════════════════════════════════════════════
    # PAGE: OVERVIEW
    # ═══════════════════════════════════════════════════════════
    if active_page == "📊 Overview":
        st.markdown('<div class="section-hdr">📊 Dataset Overview</div>', unsafe_allow_html=True)

        # ── KPI cards ──
        total_sales   = df["Sales"].sum()
        avg_daily     = daily["Sales"].mean()
        date_range    = f"{df['Order Date'].min().strftime('%b %d, %Y')} → {df['Order Date'].max().strftime('%b %d, %Y')}"
        n_categories  = df["Category"].nunique()

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">Total Sales</div>
                <div class="value">${total_sales:,.0f}</div>
                <div class="sub">{date_range}</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="metric-card gold">
                <div class="label">Avg Daily Sales</div>
                <div class="value">${avg_daily:,.2f}</div>
                <div class="sub">per day</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="metric-card coral">
                <div class="label">Total Orders</div>
                <div class="value">{len(df):,}</div>
                <div class="sub">transactions</div>
            </div>""", unsafe_allow_html=True)
        with c4:
            st.markdown(f"""
            <div class="metric-card green">
                <div class="label">Categories</div>
                <div class="value">{n_categories}</div>
                <div class="sub">product groups</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── monthly trend + weekday pattern ──
        col_a, col_b = st.columns([1.6, 1])
        with col_a:
            st.plotly_chart(chart_monthly_trend(df), use_container_width=True)
        with col_b:
            st.plotly_chart(chart_weekday_pattern(df), use_container_width=True)

        # ── raw data preview ──
        st.markdown('<div class="section-hdr">📋 Raw Data Preview</div>', unsafe_allow_html=True)
        st.dataframe(df.head(200).style.format({"Sales": "${:,.2f}"}), use_container_width=True)

        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            csv_download(df, "⬇️ Download Cleaned Data", "cleaned_sales_data.csv")
        with col_dl2:
            csv_download(daily_clean, "⬇️ Download Feature-Engineered Data", "daily_features.csv")

    # ═══════════════════════════════════════════════════════════
    # PAGE: EDA
    # ═══════════════════════════════════════════════════════════
    elif active_page == "🔍 EDA":
        st.markdown('<div class="section-hdr">🔍 Exploratory Data Analysis</div>', unsafe_allow_html=True)

        # ── sales by category ──
        cats = df["Category"].unique()
        if len(cats) > 1:
            cat_stats = df.groupby("Category")["Sales"].agg(["sum", "mean", "count"]).reset_index()
            cat_stats.columns = ["Category", "Total Sales", "Avg Sale", "Count"]
            fig_cat = px.bar(cat_stats, x="Category", y="Total Sales",
                             color="Category", template="plotly_white",
                             color_discrete_sequence=[PALETTE["teal"], PALETTE["coral"], PALETTE["gold"]])
            fig_cat.update_layout(title="<b>Total Sales by Category</b>", height=300,
                                  showlegend=False, margin=dict(t=55, b=40))
            st.plotly_chart(fig_cat, use_container_width=True)

            # region if meaningful
            if df["Region"].nunique() > 1:
                reg_stats = df.groupby("Region")["Sales"].sum().sort_values(ascending=False)
                fig_reg = px.pie(values=reg_stats.values, names=reg_stats.index,
                                 color_discrete_sequence=[PALETTE["navy"], PALETTE["teal"],
                                                          PALETTE["coral"], PALETTE["gold"]])
                fig_reg.update_layout(title="<b>Sales Distribution by Region</b>", height=300, margin=dict(t=55))
                col_pie, col_stats = st.columns([1, 1])
                with col_pie:
                    st.plotly_chart(fig_reg, use_container_width=True)
                with col_stats:
                    st.markdown('<div class="section-hdr" style="font-size:16px;">📊 Category Summary</div>', unsafe_allow_html=True)
                    st.dataframe(cat_stats.style.format({"Total Sales": "${:,.2f}", "Avg Sale": "${:,.2f}"}),
                                 use_container_width=True)

        # ── sales distribution ──
        fig_dist = px.histogram(df, x="Sales", nbins=60, template="plotly_white",
                                color_discrete_sequence=[PALETTE["slate"]])
        fig_dist.update_layout(title="<b>Sales Distribution</b>", height=300,
                               xaxis_title="Sale Amount ($)", yaxis_title="Frequency",
                               margin=dict(t=55, b=40))
        st.plotly_chart(fig_dist, use_container_width=True)

        # ── feature correlations (top lag/roll vs Sales) ──
        corr_cols = ["Sales"] + [c for c in feature_cols if "Lag" in c or "Roll" in c]
        corr_cols = [c for c in corr_cols if c in daily_clean.columns]
        if len(corr_cols) > 2:
            corr_matrix = daily_clean[corr_cols].corr()
            fig_corr = px.imshow(corr_matrix, text_auto=".2f", color_continuous_scale="RdYlGn",
                                 aspect="auto", template="plotly_white")
            fig_corr.update_layout(title="<b>Correlation: Sales vs Lag / Rolling Features</b>",
                                   height=380, margin=dict(t=60, b=40))
            st.plotly_chart(fig_corr, use_container_width=True)

        # ── data quality summary ──
        st.markdown('<div class="section-hdr">🧹 Data Quality Summary</div>', unsafe_allow_html=True)
        quality_info = f"""
        <div class="info-box">
            <p>✅ <b>Rows after cleaning:</b> {len(df):,} &nbsp; | &nbsp;
            <b>Columns:</b> {len(df.columns)} &nbsp; | &nbsp;
            <b>Date range:</b> {df['Order Date'].min().date()} → {df['Order Date'].max().date()}</p>
            <p>✅ <b>Negative / zero sales removed</b> &nbsp; | &nbsp;
            Missing values imputed or dropped &nbsp; | &nbsp;
            Types coerced automatically</p>
        </div>
        """
        st.markdown(quality_info, unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════
    # PAGE: MODEL RESULTS
    # ═══════════════════════════════════════════════════════════
    elif active_page == "🤖 Model Results":
        st.markdown('<div class="section-hdr">🤖 Model Training & Evaluation</div>', unsafe_allow_html=True)

        # ── metric cards ──
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">Best Model</div>
                <div class="value" style="font-size:22px;">{best_name}</div>
                <div class="sub">selected by R² score</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="metric-card gold">
                <div class="label">R² Score</div>
                <div class="value">{best_res['r2']:.3f}</div>
                <div class="sub">explains {best_res['r2']*100:.1f}% of variance</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="metric-card coral">
                <div class="label">MAE</div>
                <div class="value">${best_res['mae']:.2f}</div>
                <div class="sub">avg daily error</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── comparison bar charts ──
        st.plotly_chart(chart_model_comparison(results), use_container_width=True)

        # ── actual vs predicted ──
        st.plotly_chart(chart_actual_vs_predicted(results, best_name, test_dates), use_container_width=True)

        # ── results table ──
        st.markdown('<div class="section-hdr">📊 Model Comparison Table</div>', unsafe_allow_html=True)
        table_data = pd.DataFrame([
            {"Model": n, "MAE ($)": round(r["mae"], 2), "RMSE ($)": round(r["rmse"], 2), "R²": round(r["r2"], 3)}
            for n, r in results.items()
        ]).sort_values("R²", ascending=False).reset_index(drop=True)
        table_data.index = table_data.index + 1
        st.dataframe(table_data, use_container_width=True)

        # ── feature importance ──
        fig_imp = chart_feature_importance(best_model, feature_cols)
        if fig_imp:
            st.markdown('<div class="section-hdr">🏆 Feature Importances (Best Model)</div>', unsafe_allow_html=True)
            st.plotly_chart(fig_imp, use_container_width=True)

    # ═══════════════════════════════════════════════════════════
    # PAGE: FORECAST
    # ═══════════════════════════════════════════════════════════
    elif active_page == "🚀 Forecast":
        st.markdown(f'<div class="section-hdr">🚀 {forecast_days}-Day Sales Forecast</div>', unsafe_allow_html=True)

        if forecast_df.empty:
            st.error("Forecast generation failed. Please check if model trained correctly.")
        else:
            # ── summary cards ──
            total_fcst   = forecast_df["Forecasted Sales"].sum()
            avg_fcst     = forecast_df["Forecasted Sales"].mean()
            
            # Safe access to peak/low day
            try:
                peak_idx = forecast_df["Forecasted Sales"].idxmax()
                low_idx  = forecast_df["Forecasted Sales"].idxmin()
                peak_day = forecast_df.loc[peak_idx]
                low_day  = forecast_df.loc[low_idx]
                peak_val = peak_day['Forecasted Sales']
                peak_date = peak_day['Date'].strftime('%b %d, %Y')
                low_val  = low_day['Forecasted Sales']
                low_date  = low_day['Date'].strftime('%b %d, %Y')
            except Exception:
                peak_val, peak_date = 0, "N/A"
                low_val, low_date = 0, "N/A"

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="label">Total Forecasted Sales</div>
                    <div class="value">${total_fcst:,.0f}</div>
                    <div class="sub">next {forecast_days} days</div>
                </div>""", unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div class="metric-card gold">
                    <div class="label">Avg Daily Forecast</div>
                    <div class="value">${avg_fcst:,.2f}</div>
                    <div class="sub">per day</div>
                </div>""", unsafe_allow_html=True)
            with c3:
                st.markdown(f"""
                <div class="metric-card green">
                    <div class="label">Peak Day</div>
                    <div class="value">${peak_val:,.2f}</div>
                    <div class="sub">{peak_date}</div>
                </div>""", unsafe_allow_html=True)
            with c4:
                st.markdown(f"""
                <div class="metric-card coral">
                    <div class="label">Lowest Day</div>
                    <div class="value">${low_val:,.2f}</div>
                    <div class="sub">{low_date}</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ── main forecast chart ──
            st.plotly_chart(chart_forecast(forecast_df, daily_clean), use_container_width=True)

            # ── forecast table ──
            st.markdown('<div class="section-hdr">📋 Daily Forecast Table</div>', unsafe_allow_html=True)
            disp_forecast = forecast_df.copy()
            disp_forecast["Date"] = disp_forecast["Date"].dt.strftime("%Y-%m-%d (%A)")
            st.dataframe(
                disp_forecast.style.format({
                    "Forecasted Sales": "${:,.2f}",
                    "Lower Bound":      "${:,.2f}",
                    "Upper Bound":      "${:,.2f}",
                }),
                use_container_width=True
            )

            csv_download(forecast_df, "⬇️ Download Forecast CSV", "sales_forecast.csv")

    # ═══════════════════════════════════════════════════════════
    # PAGE: CATEGORY DRILL-DOWN
    # ═══════════════════════════════════════════════════════════
    elif active_page == "📁 Category Drill-Down":
        st.markdown('<div class="section-hdr">📁 Category-Level Forecast</div>', unsafe_allow_html=True)

        if not cat_forecasts:
            st.info("Only one category detected or insufficient data for split — category drill-down is not available.")
        else:
            # ── combined chart ──
            fig_cats = chart_category_breakdown(cat_forecasts)
            if fig_cats:
                st.plotly_chart(fig_cats, use_container_width=True)

            # ── per-category cards + tables ──
            n_cats = len(cat_forecasts)
            cols   = st.columns(min(n_cats, 3))
            for i, (cat, fdf) in enumerate(cat_forecasts.items()):
                with cols[i % len(cols)]:
                    total = fdf["Forecasted Sales"].sum()
                    avg   = fdf["Forecasted Sales"].mean()
                    st.markdown(f"""
                    <div class="metric-card" style="margin-bottom:12px;">
                        <div class="label">{cat}</div>
                        <div class="value">${total:,.0f}</div>
                        <div class="sub">Avg/day: ${avg:,.2f}</div>
                    </div>""", unsafe_allow_html=True)

            # ── expandable tables ──
            for cat, fdf in cat_forecasts.items():
                with st.expander(f"📊 {cat} — Detailed Forecast"):
                    disp = fdf.copy()
                    disp["Date"] = disp["Date"].dt.strftime("%Y-%m-%d")
                    st.dataframe(disp.style.format({
                        "Forecasted Sales": "${:,.2f}",
                        "Lower Bound":      "${:,.2f}",
                        "Upper Bound":      "${:,.2f}",
                    }), use_container_width=True)
                    csv_download(fdf, f"⬇️ Download {cat}", f"forecast_{cat.replace(' ','_')}.csv")

    # ═══════════════════════════════════════════════════════════
    # PAGE: BUSINESS SUMMARY
    # ═══════════════════════════════════════════════════════════
    elif active_page == "💡 Business Summary":
        st.markdown('<div class="section-hdr">💡 Business Planning Summary</div>', unsafe_allow_html=True)

        total_fcst = forecast_df["Forecasted Sales"].sum()
        avg_hist   = daily_clean["Sales"].tail(30).mean()
        avg_fcst   = forecast_df["Forecasted Sales"].mean()
        change_pct = ((avg_fcst - avg_hist) / avg_hist) * 100 if avg_hist != 0 else 0

        # ── headline card ──
        trend_arrow = "📈" if change_pct > 0 else "📉"
        trend_color = "#2A9D8F" if change_pct > 0 else "#E76F51"

        st.markdown(f"""
        <div style="background:#fff; border-radius:16px; padding:28px 32px; box-shadow:0 3px 18px rgba(0,0,0,0.08); margin-bottom:20px;">
            <h2 style="color:#1B2845; margin-top:0;">{trend_arrow} Sales Outlook — Next {forecast_days} Days</h2>
            <div style="display:flex; gap:40px; flex-wrap:wrap;">
                <div>
                    <div style="font-size:13px; color:#6b7280; text-transform:uppercase;">Forecasted Revenue</div>
                    <div style="font-size:36px; font-weight:700; color:#1B2845;">${total_fcst:,.0f}</div>
                </div>
                <div>
                    <div style="font-size:13px; color:#6b7280; text-transform:uppercase;">vs Last 30 Days</div>
                    <div style="font-size:36px; font-weight:700; color:{trend_color};">{change_pct:+.1f}%</div>
                </div>
                <div>
                    <div style="font-size:13px; color:#6b7280; text-transform:uppercase;">Model Accuracy (R²)</div>
                    <div style="font-size:36px; font-weight:700; color:#457B9D;">{best_res['r2']:.3f}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── actionable insights ──
        st.markdown('<div class="section-hdr">🎯 Actionable Insights for Your Business</div>', unsafe_allow_html=True)

        insights = []
        # inventory
        try:
            peak_date_str = forecast_df.loc[forecast_df['Forecasted Sales'].idxmax(), 'Date'].strftime('%b %d')
        except:
            peak_date_str = "N/A"

        insights.append(f"""
        <div style="background:#fff; border-radius:12px; padding:20px 24px; box-shadow:0 2px 8px rgba(0,0,0,0.06); margin-bottom:12px; border-left:4px solid #00B4D8;">
            <h4 style="color:#1B2845; margin:0 0 6px;">📦 Inventory Planning</h4>
            <p style="color:#374151; margin:0; line-height:1.6;">
                Expect <b>${total_fcst:,.0f}</b> in total sales over the next {forecast_days} days.
                Stock up to cover <b>${forecast_df['Upper Bound'].sum():,.0f}</b> (upper confidence bound)
                to avoid stockouts — especially around <b>{peak_date_str}</b> (predicted peak).
            </p>
        </div>""")

        # cash flow
        insights.append(f"""
        <div style="background:#fff; border-radius:12px; padding:20px 24px; box-shadow:0 2px 8px rgba(0,0,0,0.06); margin-bottom:12px; border-left:4px solid #F4A261;">
            <h4 style="color:#1B2845; margin:0 0 6px;">💰 Cash Flow Management</h4>
            <p style="color:#374151; margin:0; line-height:1.6;">
                Average daily revenue is expected to be <b>${avg_fcst:,.2f}</b>.
                {"Sales are trending upward — consider reinvesting in marketing." if change_pct > 0 else "Sales show a slight dip — review pricing or promotions to stimulate demand."}
                Budget conservatively using the <b>lower bound</b> (${forecast_df['Lower Bound'].sum():,.0f}) for break-even analysis.
            </p>
        </div>""")

        # staffing
        peak_weekdays = forecast_df.copy()
        peak_weekdays["dow"] = peak_weekdays["Date"].dt.dayofweek
        weekend_sales = peak_weekdays[peak_weekdays["dow"] >= 5]["Forecasted Sales"].mean()
        weekday_sales = peak_weekdays[peak_weekdays["dow"] < 5]["Forecasted Sales"].mean()
        higher = "weekends" if weekend_sales > weekday_sales else "weekdays"
        
        try:
            peak_day_full = forecast_df.loc[forecast_df['Forecasted Sales'].idxmax(), 'Date'].strftime('%A, %b %d')
            peak_amt_full = forecast_df['Forecasted Sales'].max()
        except:
            peak_day_full, peak_amt_full = "N/A", 0

        insights.append(f"""
        <div style="background:#fff; border-radius:12px; padding:20px 24px; box-shadow:0 2px 8px rgba(0,0,0,0.06); margin-bottom:12px; border-left:4px solid #2A9D8F;">
            <h4 style="color:#1B2845; margin:0 0 6px;">👥 Staffing & Operations</h4>
            <p style="color:#374151; margin:0; line-height:1.6;">
                Sales are higher on <b>{higher}</b> — schedule more staff accordingly.
                Peak predicted day is <b>{peak_day_full}</b>
                with expected sales of <b>${peak_amt_full:,.2f}</b>.
            </p>
        </div>""")

        # category
        if cat_forecasts:
            top_cat = max(cat_forecasts, key=lambda c: cat_forecasts[c]["Forecasted Sales"].sum())
            top_amt = cat_forecasts[top_cat]["Forecasted Sales"].sum()
            insights.append(f"""
            <div style="background:#fff; border-radius:12px; padding:20px 24px; box-shadow:0 2px 8px rgba(0,0,0,0.06); margin-bottom:12px; border-left:4px solid #7B68EE;">
                <h4 style="color:#1B2845; margin:0 0 6px;">🏷️ Category Focus</h4>
                <p style="color:#374151; margin:0; line-height:1.6;">
                    <b>{top_cat}</b> is projected to be your top revenue category with <b>${top_amt:,.0f}</b> 
                    in forecasted sales. Prioritise this category for promotions and inventory.
                </p>
            </div>""")

        for ins in insights:
            st.markdown(ins, unsafe_allow_html=True)

        # ── model confidence note ──
        st.markdown(f"""
        <div class="info-box" style="margin-top:20px;">
            <p><b>📌 About the Forecast</b></p>
            <p>This forecast was generated using <b>{best_name}</b> (R² = {best_res['r2']:.3f}, MAE = ${best_res['mae']:.2f}/day).</p>
            <p>The <b>confidence band</b> represents ±1.5 standard deviations of recent daily sales — 
            actual results may vary due to external factors (promotions, holidays, market shifts).</p>
            <p>Re-run the model regularly as new sales data comes in for the most accurate forecasts.</p>
        </div>
        """, unsafe_allow_html=True)

        # ── download everything ──
        st.markdown('<div class="section-hdr">📥 Downloads</div>', unsafe_allow_html=True)
        dc1, dc2, dc3 = st.columns(3)
        with dc1:
            csv_download(forecast_df, "⬇️ Main Forecast", "sales_forecast.csv")
        with dc2:
            csv_download(daily_clean, "⬇️ Feature Data", "daily_features.csv")
        with dc3:
            # combined category forecasts
            if cat_forecasts:
                combined = pd.concat([fdf.assign(Category=cat) for cat, fdf in cat_forecasts.items()])
                csv_download(combined, "⬇️ Category Forecasts", "category_forecasts.csv")


# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()