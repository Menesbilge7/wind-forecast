# Wind Speed Forecast

A modular machine learning framework for wind speed forecasting. Supports multiple models, any CSV dataset, and interactive comparison — built as a professional tool extending an ITU graduation project.

**Live demo:** [wind-forecast-enes.streamlit.app](https://wind-forecast-enes.streamlit.app)

---

## Features

- **Any CSV dataset** — upload your own meteorological data, select target and feature columns interactively
- **4 models** — LSTM, GRU, XGBoost, Linear Regression
- **Model comparison** — train all models on the same data, side-by-side leaderboard
- **Lag feature engineering** — automatic lag/rolling/diff features for XGBoost
- **Uncertainty estimation** — Monte Carlo Dropout confidence bands
- **Hyperparameter optimization** — Bayesian search via Optuna
- **Real-time inference** — load a saved model, upload recent data, get predictions
- **ERA5 integration** — download reanalysis data for any global coordinate via Copernicus CDS

---

## Benchmark Results

*Skill Score = 1 − RMSE_model / RMSE_persistence*

### Terkos (summer, days 109–277, n=4039)
10 m wind speed · 1-hour ahead · test set (10 % holdout)

| Model | R² | RMSE (m/s) | MAE (m/s) | Skill Score |
|---|---|---|---|---|
| **Linear Regression** | **0.880** | **1.032** | **0.698** | **+0.031** |
| XGBoost + lag features | 0.877 | 1.039 | 0.738 | +0.025 |
| GRU | 0.878 | 1.039 | 0.725 | +0.024 |
| LSTM | 0.873 | 1.061 | 0.739 | +0.003 |
| Persistence baseline | 0.872 | 1.065 | 0.690 | 0.000 |

### Osmangazi (full year, days 1–365, n=8637)
10 m wind speed · 1-hour ahead · test set (10 % holdout)

| Model | R² | RMSE (m/s) | MAE (m/s) | Skill Score |
|---|---|---|---|---|
| **LSTM** | **0.8941** | **0.9337** | — | **+0.010** |
| XGBoost + lag features | 0.8939 | 0.9356 | 0.678 | +0.008 |
| GRU | 0.8907 | 0.9484 | 0.702 | -0.006 |
| Persistence baseline | 0.8920 | 0.9428 | 0.678 | 0.000 |
| Linear Regression | 0.8341 | 1.1684 | 0.744 | -0.239 |

> **Insight:** Summer-only data (Terkos) favours simpler models; full-year data with seasonal variability (Osmangazi) gives the edge to LSTM and XGBoost with lag features.

---

## Quick Start

```bash
git clone https://github.com/Menesbilge7/wind-forecast.git
cd wind-forecast
python -m venv .venv && .venv\Scripts\activate   # Windows
pip install -r requirements.txt
streamlit run app.py
```

---

## Project Structure

```
wind-forecast/
├── app.py                  # 6-page Streamlit application
├── main.py                 # CLI pipeline  (python main.py configs/x.yaml)
├── configs/                # YAML experiment configs
├── src/
│   ├── data/
│   │   ├── loader.py
│   │   ├── preprocessor.py   # cyclic features, lag features, scaling
│   │   ├── sequencer.py      # sliding window sequences
│   │   └── era5_fetcher.py
│   ├── models/
│   │   ├── lstm.py / gru.py / linear.py / xgboost_model.py
│   │   └── base.py
│   ├── training/
│   │   ├── trainer.py
│   │   └── optimizer.py      # Optuna Bayesian search
│   └── evaluation/
│       └── metrics.py        # RMSE, MAE, R², Skill Score
└── outputs/
    ├── models/   # saved weights + scaler + metadata
    ├── plots/
    └── reports/
```

---

## Data Format

Any CSV with meteorological columns is supported. The app auto-detects numeric columns and lets you select the target and features interactively.

Example (Terkos station):
```
gunler;saat;otuz_metre;yirmi_metre;on_metre;sicaklik;nem;basinc
109;1;4.38;4.11;4.06;14.7;78;1018
```

ERA5 reanalysis data can be downloaded directly from the app for any global coordinate.

---

## ERA5 Setup

Add your Copernicus CDS token to `~/.cdsapirc`:
```
url: https://cds.climate.copernicus.eu/api
key: YOUR_TOKEN_HERE
```

For Streamlit Cloud, add it under **App Settings → Secrets**:
```toml
[cds]
url = "https://cds.climate.copernicus.eu/api"
key = "YOUR_TOKEN_HERE"
```

---

*ITU MTO494 graduation project — extended into a general-purpose forecasting framework.*
