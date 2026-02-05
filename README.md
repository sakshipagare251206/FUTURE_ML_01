# 📈 Sales & Demand Forecasting Dashboard

A fully self-contained **Streamlit** application that trains ML models on historical sales data, generates multi-day forecasts with confidence bands, and presents everything in a business-ready dashboard — no coding required after setup.

---

## ⚡ Quick Start (< 3 minutes)

```bash
# 1. Clone / copy the two files into a folder
#    app.py  ←  the dashboard
#    requirements.txt

# 2. Install dependencies (one-time)
pip install -r requirements.txt

# 3. Launch
streamlit run app.py
# Opens automatically at  http://localhost:8501
```

> **No API keys, no cloud setup, no database.** Everything runs locally.

---

## 📦 What's Inside

| File | Purpose |
|---|---|
| `app.py` | Complete Streamlit dashboard (single file) |
| `requirements.txt` | Python dependencies |
| `README.md` | This file |

---

## 🎯 Features at a Glance

### 📂 Data Flexibility
- **Two built-in sample datasets** (Superstore-style & Online-Retail-style) — click and go.
- **Upload any CSV** — the app auto-detects date, sales, category, and region columns.
- Automatic cleaning: negative values removed, types coerced, missing values handled.

### 🧪 Feature Engineering (28 features)
| Group | Features |
|---|---|
| Calendar | Month, Day of Week, Day of Year, Quarter, Weekend flag, Holiday flag |
| Cyclical | Sin/Cos encodings of Month & Day-of-Year |
| Trend | Linear day number |
| Lags | 1-, 3-, 7-, 14-, 30-day lagged sales |
| Rolling | 3-, 7-, 14-, 30-day rolling-mean sales |
| Diffs | 1-day & 7-day sales differences |
| Category | Proportion of each product category per day |
| Operational | Quantity, Avg Discount, Number of Orders |

### 🤖 Three ML Models (auto-compared)
| Model | Strength |
|---|---|
| **Ridge Regression** | Fast baseline, good with linear trends |
| **Random Forest** | Handles non-linear patterns robustly |
| **Gradient Boosting** | Usually the most accurate; best feature-importance support |

The dashboard automatically picks the **best model by R²** and uses it for forecasting.

### 🚀 Forecasting
- **Recursive walk-forward forecast** — each predicted day feeds back as a lag for the next.
- **Confidence bands** (±1.5σ of recent daily variance) shown on the chart.
- Configurable horizon: **7 – 90 days**.

### 📁 Category Drill-Down
- Separate Gradient Boosting model trained **per product category**.
- Side-by-side category forecasts with expandable detail tables.

### 💡 Business Summary Page
- Actionable cards for **inventory, cash flow, staffing, and category focus**.
- Automatic trend detection (↑ or ↓ vs last 30 days).
- One-click CSV exports for every output.

---

## 🖥️ Dashboard Pages

| Page | What You'll See |
|---|---|
| **📊 Overview** | KPI cards, monthly trend, weekday pattern, raw data preview |
| **🔍 EDA** | Category & region breakdowns, sales distribution, correlation heatmap, data-quality summary |
| **🤖 Model Results** | MAE / RMSE / R² comparison, Actual-vs-Predicted overlay, feature importances |
| **🚀 Forecast** | 30-day (configurable) forecast chart with confidence band, daily forecast table |
| **📁 Category Drill-Down** | Per-category forecast charts and tables |
| **💡 Business Summary** | Inventory, cash-flow, staffing & category insights; all downloads |

---

## 📋 CSV Format Guide

If you upload your **own** data, the app expects (at minimum):

| Required | Name examples | Notes |
|---|---|---|
| Date column | `Order Date`, `Date`, `Timestamp` | Any parseable date format |
| Sales column | `Sales`, `Revenue`, `Amount`, `Total` | Numeric |

Optional columns that enhance the dashboard:

| Column | Examples |
|---|---|
| Category | `Category`, `Product Category` |
| Region | `Region`, `Country` |
| Quantity | `Quantity`, `Qty` |
| Discount | `Discount` (0–1 or 0–100) |

---

## 🔧 Customisation Tips

| What | How |
|---|---|
| Change forecast horizon | Use the **Forecast Horizon** slider in the sidebar (7–90 days) |
| Adjust train/test split | Use the **Test Split %** slider (10–40 %) |
| Add a new model | Add an entry to the `MODEL_ZOO` dict in `app.py` |
| Change colours | Edit the `PALETTE` dict at the top of `app.py` |
| Generate more sample data | Call `generate_superstore_data(n_days=1460)` for 4 years |

---

## 📐 Architecture

```
app.py
 ├── Data Layer
 │    ├── generate_superstore_data()   — synthetic Superstore CSV
 │    ├── generate_retail_data()       — synthetic Online-Retail CSV
 │    └── clean_dataframe()            — auto-detect & coerce columns
 ├── Feature Layer
 │    ├── build_daily_features()       — aggregate + 28 engineered features
 │    └── get_feature_cols()           — ordered feature list
 ├── Model Layer
 │    ├── MODEL_ZOO                    — Ridge / RF / GB factories
 │    ├── train_and_evaluate()         — temporal train/test + metrics
 │    └── recursive_forecast()         — walk-forward prediction loop
 ├── Category Layer
 │    └── category_forecast()          — per-category GB + forecast
 ├── Visualisation Layer
 │    └── chart_*()                    — 7 Plotly chart builders
 └── UI Layer (main)
      └── main()                       — 6-page Streamlit app
```

---

## 🛑 Troubleshooting

| Problem | Solution |
|---|---|
| `ModuleNotFoundError: streamlit` | Run `pip install -r requirements.txt` |
| Upload doesn't work | Make sure the file is `.csv` and has a date + numeric column |
| R² is very low | Try uploading more data (≥ 1 year). Also check for outliers in your CSV |
| Forecast looks flat | If your data has very little seasonality the model will produce a near-constant forecast — this is expected |

---

*Built with ❤️ using Streamlit, scikit-learn, Plotly & pandas.*
