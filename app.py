from __future__ import annotations
import json
import os
import warnings
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.data.loader import load_csv
from src.data.preprocessor import add_cyclic_features, add_lag_features, split_data, fit_scaler, scale, inverse_scale_column
from src.data.sequencer import make_sequences, make_persistence_sequences
from src.evaluation.metrics import compute_metrics, compute_skill_score

st.set_page_config(
    page_title="Rüzgar Hızı Tahmin Sistemi",
    page_icon="💨",
    layout="wide",
)

# ── Yardımcı fonksiyonlar ─────────────────────────────────────────────────────

def build_model(model_type: str, cfg: dict):
    if model_type == "lstm":
        from src.models.lstm import LSTMModel
        return LSTMModel(cfg)
    elif model_type == "gru":
        from src.models.gru import GRUModel
        return GRUModel(cfg)
    elif model_type == "xgboost":
        from src.models.xgboost_model import XGBoostModel
        return XGBoostModel(cfg)
    else:
        from src.models.linear import LinearModel
        return LinearModel(cfg)


def train_dl_model(model, X_train, y_train, X_val, y_val, cfg, progress_slot, info_slot):
    from tensorflow import keras

    class _Cb(keras.callbacks.Callback):
        def __init__(self, total):
            super().__init__()
            self.total = total
            self.history = {"loss": [], "val_loss": []}

        def on_epoch_end(self, epoch, logs=None):
            self.history["loss"].append(logs.get("loss", 0))
            self.history["val_loss"].append(logs.get("val_loss", 0))
            progress_slot.progress(min((epoch + 1) / self.total, 1.0))
            info_slot.info(
                f"Epoch {epoch+1}/{self.total} — "
                f"loss: {logs.get('loss', 0):.4f}  val_loss: {logs.get('val_loss', 0):.4f}"
            )

    cb = _Cb(cfg["epochs"])
    early = keras.callbacks.EarlyStopping(patience=cfg["patience"], restore_best_weights=True)
    model.model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        callbacks=[cb, early],
        verbose=0,
    )
    return cb.history


def get_predictions(model, X_test, y_test, scaler, all_columns, target_col):
    y_pred_s = model.predict(X_test)
    y_pred = inverse_scale_column(y_pred_s, scaler, all_columns, target_col)
    y_true = inverse_scale_column(y_test,   scaler, all_columns, target_col)
    return y_true, y_pred


def render_prediction_plot(y_true, y_pred, title: str):
    multistep = y_true.ndim == 2 and y_true.shape[1] > 1
    yt = y_true[:, 0] if multistep else y_true
    yp = y_pred[:, 0] if multistep else y_pred
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    axes[0].plot(yt, label="Gerçek", alpha=0.8, linewidth=0.9)
    axes[0].plot(yp, label="Tahmin", alpha=0.8, linewidth=0.9)
    axes[0].set_title(title)
    axes[0].set_xlabel("Zaman adımı")
    axes[0].set_ylabel("Rüzgar hızı (m/s)")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    lims = [min(yt.min(), yp.min()), max(yt.max(), yp.max())]
    axes[1].scatter(yt, yp, alpha=0.4, s=8)
    axes[1].plot(lims, lims, "r--", linewidth=1)
    axes[1].set_xlabel("Gerçek"); axes[1].set_ylabel("Tahmin")
    axes[1].set_title("Scatter"); axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def render_loss_plot(history: dict):
    if not history.get("loss"):
        st.info("Doğrusal regresyon için loss grafiği oluşturulmaz.")
        return
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(history["loss"], label="Train")
    ax.plot(history.get("val_loss", []), label="Val")
    ax.set_title("Eğitim Kaybı"); ax.set_xlabel("Epoch"); ax.set_ylabel("MSE")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); st.pyplot(fig); plt.close(fig)


def render_horizon_profile(metrics: dict):
    if "per_horizon" not in metrics:
        st.info("Tek adımlı modelde horizon profili gösterilmez.")
        return
    ph = metrics["per_horizon"]
    steps = list(range(1, len(ph) + 1))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, key, color, ylabel in zip(
        axes,
        ["rmse", "mae", "r2"],
        ["steelblue", "orange", "green"],
        ["RMSE (m/s)", "MAE (m/s)", "R²"],
    ):
        ax.plot(steps, [m[key] for m in ph], marker="o", color=color)
        ax.set_title(f"{ylabel} — horizon başına")
        ax.set_xlabel("Tahmin adımı (saat)"); ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
    plt.tight_layout(); st.pyplot(fig); plt.close(fig)
    st.dataframe(
        pd.DataFrame(ph, index=[f"+{i}h" for i in steps])[["rmse", "mae", "r2"]].round(4),
        use_container_width=True,
    )


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("💨 Rüzgar Tahmin")
    page = st.radio("Sayfa", ["Tek Model", "Model Karşılaştırması", "Hiperparametre Optimizasyonu", "Belirsizlik Tahmini", "Gerçek Zamanlı Tahmin", "ERA5 Veri İndir"], label_visibility="collapsed")
    st.divider()

    st.subheader("📂 Veri")
    uploaded = st.file_uploader("CSV dosyası yükle", type=["csv"])
    separator = st.selectbox("Separator", [";", ",", "\\t"], index=0)
    decimal   = st.selectbox("Ondalık ayırıcı", [",", "."], index=0)

    _sidebar_cols: list[str] = []
    target_col: str = ""
    feature_selection: list[str] = []
    _needs_cols = page not in ("ERA5 Veri İndir", "Gerçek Zamanlı Tahmin")
    if uploaded is not None and _needs_cols:
        try:
            _sep_char = "\t" if separator == "\\t" else separator
            uploaded.seek(0)
            _prev = load_csv(uploaded, separator=_sep_char, decimal=decimal)
            _prev = add_cyclic_features(_prev)
            _sidebar_cols = _prev.select_dtypes(include="number").columns.tolist()
            if _sidebar_cols:
                _wind_keywords = ["wind", "hiz", "speed", "metre", "knot", "kts", "ws"]
                _default_target = next(
                    (c for c in _sidebar_cols if any(k in c.lower() for k in _wind_keywords)),
                    _sidebar_cols[-1],
                )
                target_col = st.selectbox(
                    "Hedef sütun (tahmin edilecek)",
                    _sidebar_cols,
                    index=_sidebar_cols.index(_default_target),
                )
                feature_selection = st.multiselect(
                    "Özellikler (feature)",
                    _sidebar_cols,
                    default=[c for c in _sidebar_cols if c != target_col],
                )
        except Exception as _e:
            st.warning(f"Sütunlar okunamadı — separator/ondalık ayarını kontrol et. ({_e})")
    elif _needs_cols:
        st.caption("CSV yüklenince sütun seçimi aktif olur.")

    autoregressive = st.checkbox("Autoregressive (hedef geçmişi özellik olarak ekle)", value=True)
    use_lag_features = st.checkbox("Lag özellikler ekle (XGBoost için önerilir)", value=False)
    train_ratio = st.slider("Train oranı (%)", 60, 90, 80) / 100
    val_ratio   = st.slider("Validation oranı (%)", 5, 20, 10) / 100
    st.divider()

    st.subheader("⏱ Zaman Serisi")
    lookback = st.slider("Lookback (saat)", 6, 72, 24)
    horizon  = st.slider("Horizon (saat)", 1, 24, 1)
    st.divider()

    st.subheader("🧠 Model Hiperparametreleri")
    if page == "Tek Model":
        model_choice = st.selectbox("Model tipi", ["lstm", "gru", "xgboost", "linear"])
    elif page == "Hiperparametre Optimizasyonu":
        model_choice = st.selectbox("Model tipi", ["lstm", "gru", "xgboost"])
    elif page == "Belirsizlik Tahmini":
        model_choice = st.selectbox("Model tipi", ["lstm", "gru"])
    else:
        model_choice = None  # karşılaştırmada hepsi çalışır

    _is_xgb = model_choice == "xgboost"
    _is_dl  = model_choice in ("lstm", "gru")

    # Derin öğrenme parametreleri (LSTM/GRU)
    if not _is_xgb:
        epochs     = st.slider("Max epoch", 10, 200, 100)
        batch_size = st.selectbox("Batch size", [16, 32, 64], index=1)
        units_1    = st.slider("1. katman nöron", 16, 256, 64, step=16)
        units_2    = st.slider("2. katman nöron", 8, 128, 32, step=8)
        dropout    = st.slider("Dropout", 0.0, 0.5, 0.2, step=0.05)
        patience   = st.slider("Early stopping patience", 3, 30, 10)
        lr         = st.select_slider("Learning rate", [0.01, 0.001, 0.0005, 0.0001], value=0.001)
    else:
        epochs = batch_size = units_1 = units_2 = 0
        dropout = 0.0; patience = 0; lr = 0.0

    # XGBoost parametreleri
    if _is_xgb:
        xgb_n_estimators = st.slider("Ağaç sayısı (n_estimators)", 100, 2000, 500, step=100)
        xgb_max_depth    = st.slider("Maks derinlik (max_depth)", 3, 10, 6)
        xgb_lr           = st.select_slider("Learning rate (XGB)", [0.3, 0.1, 0.05, 0.01], value=0.05)
        xgb_patience     = st.slider("Early stopping patience", 10, 50, 20)
    else:
        xgb_n_estimators = 500; xgb_max_depth = 6; xgb_lr = 0.05; xgb_patience = 20
    st.divider()

    run_btn = st.button("🚀 Eğit", type="primary", use_container_width=True)

# ── Başlık ────────────────────────────────────────────────────────────────────
st.title("💨 Rüzgar Hızı Tahmin Sistemi")
st.caption("LSTM · GRU · Doğrusal Regresyon — karşılaştırmalı zaman serisi tahmini")

if page == "ERA5 Veri İndir":
    uploaded = None  # ERA5 sayfasında CSV gerekmez

if uploaded is None and page not in ("ERA5 Veri İndir", "Gerçek Zamanlı Tahmin"):
    st.info("Sol panelden bir CSV dosyası yükleyin.")
    st.stop()

# ── Ortak veri hazırlığı (ERA5 ve RT sayfalarında atlanır) ───────────────────
_skip_common = page in ("ERA5 Veri İndir", "Gerçek Zamanlı Tahmin")
if not _skip_common:
    sep = "\t" if separator == "\\t" else separator
    uploaded.seek(0)
    df  = load_csv(uploaded, separator=sep, decimal=decimal)
    df = add_cyclic_features(df)
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    dropped = [c for c in df.columns if c not in numeric_cols]
    if dropped:
        df = df[numeric_cols]
        st.caption(f"Sayısal olmayan sütunlar çıkarıldı: {dropped}")

    if not target_col or target_col not in df.columns:
        st.error(f"Hedef sütun seçilmedi veya bulunamadı. Mevcut: {list(df.columns)}")
        st.stop()

    if use_lag_features:
        df = add_lag_features(df, target_col)
        st.caption(f"Lag özellikler eklendi — yeni boyut: {df.shape[1]} sütun, {len(df)} satır")

    # Sidebar'dan gelen feature seçimini kullan; hiçbir şey seçilmemişse fallback
    _base_features = [c for c in feature_selection if c in df.columns]
    # Lag özellikleri otomatik ekle (sidebar multiselect yeniden yüklenmeden önce)
    if use_lag_features:
        _lag_cols = [c for c in df.columns if c.startswith(f"{target_col}_")]
        _base_features = list(dict.fromkeys(_base_features + _lag_cols))
    if autoregressive and target_col not in _base_features:
        _base_features.append(target_col)
    feature_cols = _base_features if _base_features else [c for c in df.columns if c != target_col]
    all_columns  = list(dict.fromkeys(feature_cols + [target_col]))

    with st.expander("📋 Veri Önizleme", expanded=False):
        c1, c2, c3 = st.columns(3)
        c1.metric("Satır", len(df))
        c2.metric("Sütun", len(df.columns))
        c3.metric("Eksik değer", int(df.isnull().sum().sum()))
        st.dataframe(df.head(10), use_container_width=True)
        fig, ax = plt.subplots(figsize=(12, 2))
        ax.plot(df[target_col].values, linewidth=0.7)
        ax.set_title(f"{target_col}"); ax.set_ylabel("m/s"); ax.grid(True, alpha=0.3)
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    train_df, val_df, test_df = split_data(df, train_ratio, val_ratio)
    scaler   = fit_scaler(train_df, all_columns)
    train_s  = scale(train_df, scaler, all_columns)
    val_s    = scale(val_df,   scaler, all_columns)
    test_s   = scale(test_df,  scaler, all_columns)

    X_train, y_train = make_sequences(train_s, feature_cols, target_col, lookback, horizon)
    X_val,   y_val   = make_sequences(val_s,   feature_cols, target_col, lookback, horizon)
    X_test,  y_test  = make_sequences(test_s,  feature_cols, target_col, lookback, horizon)
    y_pers_test      = make_persistence_sequences(test_s, target_col, lookback, horizon)

    dl_cfg = {
        "units": [units_1, units_2],
        "dropout": dropout,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": xgb_lr if _is_xgb else lr,
        "patience": xgb_patience if _is_xgb else patience,
        "horizon": horizon,
        "n_estimators": xgb_n_estimators,
        "max_depth": xgb_max_depth,
        "colsample_bytree": 0.8,
        "subsample": 0.8,
    }

# ═══════════════════════════════════════════════════════════════════════════════
# SAYFA 5 — ERA5 VERİ İNDİR  (run_btn'den bağımsız, önce işlenir)
# ═══════════════════════════════════════════════════════════════════════════════
if page == "ERA5 Veri İndir":
    st.subheader("🌍 ERA5 Veri İndirici")
    st.caption("Copernicus Climate Data Store — herhangi bir koordinat için saatlik meteorolojik veri")

    _cds_url, _cds_key = None, None
    try:
        _cds_url = st.secrets["cds"]["url"]
        _cds_key = st.secrets["cds"]["key"]
        st.success("CDS API anahtarı bulundu (Streamlit Secrets).")
    except Exception:
        if (Path.home() / ".cdsapirc").exists():
            st.success("CDS API anahtarı bulundu (~/.cdsapirc).")
        else:
            st.error("**CDS API anahtarı bulunamadı.**")
            st.info(
                "**Streamlit Cloud kullanıyorsan** → Uygulama sayfasında "
                "**⋮ → Settings → Secrets** bölümüne şunu ekle:\n"
                "```toml\n[cds]\nurl = \"https://cds.climate.copernicus.eu/api\"\n"
                "key = \"BURAYA_TOKEN\"\n```\n\n"
                "**Yerel çalışıyorsan** → Terminale yaz:\n"
                "```\necho url: https://cds.climate.copernicus.eu/api > "
                "%USERPROFILE%\\.cdsapirc\necho key: BURAYA_TOKEN >> "
                "%USERPROFILE%\\.cdsapirc\n```\n\n"
                "Token: [cds.climate.copernicus.eu](https://cds.climate.copernicus.eu) "
                "→ Profil → Personal Access Token"
            )
            st.stop()

    from src.data.era5_fetcher import download_era5, netcdf_to_dataframe, ERA5_VARIABLES
    import datetime

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📍 Koordinat")
        lat = st.number_input("Enlem (Latitude)",  value=41.25, min_value=-90.0,  max_value=90.0,  step=0.01, format="%.2f")
        lon = st.number_input("Boylam (Longitude)", value=28.75, min_value=-180.0, max_value=180.0, step=0.01, format="%.2f")
        st.caption("Terkos: 41.38°N, 28.58°E  |  İstanbul: 41.01°N, 28.97°E")

    with col2:
        st.subheader("📅 Tarih Aralığı")
        start = st.date_input("Başlangıç", value=datetime.date(2023, 1, 1))
        end   = st.date_input("Bitiş",     value=datetime.date(2023, 12, 31))

    st.subheader("📊 Değişken Seçimi")
    all_var_names = list(ERA5_VARIABLES.keys())
    default_vars  = [
        "10m_u_component_of_wind", "10m_v_component_of_wind",
        "2m_temperature", "surface_pressure", "2m_dewpoint_temperature",
    ]
    selected_vars = st.multiselect("ERA5 değişkenleri", all_var_names, default=default_vars)

    nc_path  = Path("data/external/era5_raw.nc")
    csv_path = Path("data/external/era5_processed.csv")

    dl_btn = st.button("⬇️ ERA5 Verisi İndir", type="primary")

    if dl_btn:
        if not selected_vars:
            st.error("En az bir değişken seçin.")
        else:
            with st.spinner("ERA5 verisi indiriliyor… (birkaç dakika sürebilir)"):
                try:
                    download_era5(lat=lat, lon=lon, start_date=str(start), end_date=str(end),
                                  variables=selected_vars, output_path=str(nc_path),
                                  cds_url=_cds_url, cds_key=_cds_key)
                    st.success(f"İndirildi: {nc_path}")
                except Exception as e:
                    st.error(f"İndirme hatası: {e}")
                    st.stop()
            with st.spinner("NetCDF → DataFrame dönüştürülüyor…"):
                try:
                    era5_df = netcdf_to_dataframe(nc_path)
                    era5_df.to_csv(csv_path, index=False)
                    st.success(f"CSV kaydedildi: {csv_path}")
                except ImportError:
                    st.error("xarray paketi gerekli. Terminale yaz: `.venv\\Scripts\\pip install xarray netcdf4`")
                    st.stop()
            st.subheader("📋 Veri Önizleme")
            st.dataframe(era5_df.head(24), use_container_width=True)
            fig, axes = plt.subplots(2, 1, figsize=(12, 5))
            if "wind_speed_10m" in era5_df.columns:
                axes[0].plot(era5_df["wind_speed_10m"].values[:720], linewidth=0.8)
                axes[0].set_title("10m Rüzgar Hızı (ilk 30 gün)"); axes[0].set_ylabel("m/s"); axes[0].grid(True, alpha=0.3)
            if "t2m" in era5_df.columns:
                axes[1].plot(era5_df["t2m"].values[:720], color="orange", linewidth=0.8)
                axes[1].set_title("2m Sıcaklık (ilk 30 gün)"); axes[1].set_ylabel("°C"); axes[1].grid(True, alpha=0.3)
            plt.tight_layout(); st.pyplot(fig); plt.close(fig)
            st.download_button("📊 ERA5 CSV İndir", data=era5_df.to_csv(index=False),
                               file_name="era5_processed.csv", mime="text/csv")
            st.info("**Sonraki adım:** Bu CSV'yi 'Tek Model' sayfasına yükle, hedef sütun: `wind_speed_10m`")

    elif csv_path.exists():
        st.info(f"Önceden indirilmiş veri mevcut: `{csv_path}`")
        era5_df = pd.read_csv(csv_path, parse_dates=["datetime"])
        st.dataframe(era5_df.head(10), use_container_width=True)
        st.download_button("📊 Mevcut ERA5 CSV İndir", data=era5_df.to_csv(index=False),
                           file_name="era5_processed.csv", mime="text/csv")

    st.stop()  # diğer sayfa bloklarına geçmesin

# ═══════════════════════════════════════════════════════════════════════════════
# SAYFA 6 — GERÇEK ZAMANLI TAHMİN  (run_btn'den bağımsız)
# ═══════════════════════════════════════════════════════════════════════════════
if page == "Gerçek Zamanlı Tahmin":
    import joblib as _jl

    st.subheader("📡 Gerçek Zamanlı Tahmin")
    st.caption("Eğitilmiş modeli yükle → son dönem verisini gir → sonraki saatleri tahmin et")

    _mdir = Path("outputs/models")
    available_models = [
        m for m in ("lstm", "gru", "linear")
        if (_mdir / f"{m}_scaler.pkl").exists() and (_mdir / f"{m}_meta.json").exists()
    ]

    if not available_models:
        st.warning("Henüz kaydedilmiş model bulunamadı.")
        st.info(
            "Önce **Tek Model** sayfasına git, bir CSV yükle ve modeli eğit. "
            "Eğitim bitince model otomatik olarak kaydedilir."
        )
        st.stop()

    col_cfg, col_data = st.columns([1, 2])

    with col_cfg:
        sel_model = st.selectbox("Model", available_models)
        with open(_mdir / f"{sel_model}_meta.json", encoding="utf-8") as _f:
            _meta = json.load(_f)
        saved_cols    = _meta["all_columns"]
        saved_target  = _meta["target_column"]
        saved_lookback = _meta["lookback"]
        saved_horizon  = _meta["horizon"]
        saved_features = [c for c in saved_cols if c != saved_target]

        st.caption(f"Lookback: {saved_lookback}h · Horizon: {saved_horizon}h · {len(saved_features)} özellik")
        if saved_horizon > 1:
            show_steps = st.slider("Gösterilecek adım sayısı", 1, saved_horizon, min(saved_horizon, 6))
        else:
            show_steps = 1

    with col_data:
        st.subheader("📂 Son Dönem Verisi")
        rt_sep = st.selectbox("Separator", [";", ",", "\\t"], key="rt_sep")
        rt_dec = st.selectbox("Ondalık", [",", "."], key="rt_dec")
        rt_upload = st.file_uploader(
            f"CSV yükle — en az {saved_lookback} satır, eğitim verileriyle aynı format",
            type=["csv"], key="rt_upload",
        )

    if rt_upload is None:
        st.info(f"Eğitim verisiyle aynı formatta bir CSV dosyası yükleyin (en az {saved_lookback} satır).")
        st.stop()

    _rt_sep_char = "\t" if rt_sep == "\\t" else rt_sep
    rt_df = load_csv(rt_upload, separator=_rt_sep_char, decimal=rt_dec)
    rt_df = add_cyclic_features(rt_df)
    rt_df = rt_df.select_dtypes(include="number")

    missing_cols = [c for c in saved_cols if c not in rt_df.columns]
    if missing_cols:
        st.error(f"Eksik sütunlar: {missing_cols}")
        st.caption(f"Mevcut sütunlar: {list(rt_df.columns)}")
        st.stop()

    rt_df = rt_df[saved_cols]

    if len(rt_df) < saved_lookback:
        st.error(f"En az {saved_lookback} satır gerekli, yüklenen: {len(rt_df)} satır.")
        st.stop()

    with col_data:
        st.success(f"{len(rt_df)} satır yüklendi — son {saved_lookback} satır giriş penceresi olarak kullanılacak.")
        with st.expander("Veri önizleme (son 10 satır)"):
            st.dataframe(rt_df.tail(10), use_container_width=True)

    predict_btn = st.button("🔮 Tahmin Et", type="primary", use_container_width=True)
    if not predict_btn:
        st.stop()

    with st.spinner("Model yükleniyor ve tahmin yapılıyor…"):
        _saved_scaler = _jl.load(_mdir / f"{sel_model}_scaler.pkl")

        window = rt_df.iloc[-saved_lookback:].copy()
        window_scaled = _saved_scaler.transform(window.values)

        feat_idx = [saved_cols.index(c) for c in saved_features]
        X_rt = window_scaled[:, feat_idx].reshape(1, saved_lookback, len(saved_features))

        if sel_model in ("lstm", "gru"):
            from tensorflow import keras as _keras
            _rt_model = _keras.models.load_model(_mdir / f"{sel_model}.keras")
            y_scaled_rt = _rt_model.predict(X_rt, verbose=0)
        else:
            _rt_lin = _jl.load(_mdir / f"{sel_model}.keras")
            y_scaled_rt = _rt_lin.predict(X_rt.reshape(1, -1))

        y_pred_rt = inverse_scale_column(y_scaled_rt, _saved_scaler, saved_cols, saved_target)

    st.subheader("🎯 Tahmin Sonuçları")

    if y_pred_rt.ndim == 2 and y_pred_rt.shape[1] > 1:
        preds = y_pred_rt[0, :show_steps]
    else:
        preds = y_pred_rt.ravel()[:1]
        show_steps = 1

    steps = list(range(1, show_steps + 1))

    # Metrik kartları
    _cols_m = st.columns(min(show_steps, 6))
    for _i, (_s, _v) in enumerate(zip(steps, preds)):
        _cols_m[_i % len(_cols_m)].metric(f"+{_s}h", f"{_v:.2f} m/s")

    # Geçmiş + tahmin grafiği
    fig, ax = plt.subplots(figsize=(12, 4))
    window_target = window[saved_target].values
    x_past   = np.arange(-saved_lookback, 0)
    x_future = np.arange(1, show_steps + 1)
    ax.plot(x_past,   window_target, color="gray",      linewidth=0.9, label=f"Son {saved_lookback}h (gerçek)")
    ax.plot(x_future, preds,         color="steelblue", linewidth=1.5, marker="o", label="Tahmin")
    ax.axvline(x=0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Saat (0 = şimdiki an)")
    ax.set_ylabel("Rüzgar hızı (m/s)")
    ax.set_title(f"{sel_model.upper()} — Sonraki {show_steps} Saat Tahmini")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    # Tahmin tablosu + indirme
    pred_table = pd.DataFrame({"Adım": [f"+{s}h" for s in steps], "Tahmin (m/s)": preds.round(3)})
    st.dataframe(pred_table.set_index("Adım"), use_container_width=True)
    st.download_button(
        "📊 Tahmin CSV İndir",
        pred_table.to_csv(index=False),
        f"{sel_model}_rt_tahmin.csv",
        "text/csv",
    )

    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
if not run_btn:
    st.stop()

# ═══════════════════════════════════════════════════════════════════════════════
# SAYFA 1 — TEK MODEL
# ═══════════════════════════════════════════════════════════════════════════════
if page == "Tek Model":
    st.subheader(f"🤖 {model_choice.upper()} Eğitimi")
    info = st.empty()
    bar  = st.progress(0)
    info.info(f"{model_choice.upper()} eğitiliyor… ({len(X_train)} eğitim örneği)")

    model = build_model(model_choice, {**dl_cfg, "horizon": horizon})
    model.build((X_train.shape[1], X_train.shape[2]))

    if model_choice in ("lstm", "gru"):
        history = train_dl_model(model, X_train, y_train, X_val, y_val, dl_cfg, bar, info)
    else:
        history = model.fit(X_train, y_train, X_val, y_val)
        bar.progress(1.0)

    info.success("Eğitim tamamlandı!")

    _mdir = Path("outputs/models")
    _mdir.mkdir(parents=True, exist_ok=True)
    model.save(str(_mdir / f"{model_choice}.keras"))
    import joblib as _jl
    _jl.dump(scaler, _mdir / f"{model_choice}_scaler.pkl")
    _hz = int(y_train.shape[1]) if y_train.ndim == 2 else 1
    with open(_mdir / f"{model_choice}_meta.json", "w", encoding="utf-8") as _mf:
        json.dump({"all_columns": all_columns, "target_column": target_col,
                   "lookback": lookback, "horizon": _hz}, _mf)
    st.caption(f"Model kaydedildi → outputs/models/{model_choice}")

    y_true, y_pred = get_predictions(model, X_test, y_test, scaler, all_columns, target_col)
    metrics = compute_metrics(y_true, y_pred)
    multistep = y_true.ndim == 2 and y_true.shape[1] > 1

    st.subheader("📊 Sonuçlar")
    m1, m2, m3 = st.columns(3)
    m1.metric("R²",       f"{metrics['r2']:.4f}")
    m2.metric("RMSE m/s", f"{metrics['rmse']:.4f}")
    m3.metric("MAE m/s",  f"{metrics['mae']:.4f}")

    tabs = st.tabs(["📈 Tahmin", "📊 Horizon Profili", "📉 Eğitim Kaybı"])
    with tabs[0]:
        lbl = f"{model_choice.upper()} — +1h Tahmin vs Gerçek" if multistep else f"{model_choice.upper()} — Tahmin vs Gerçek"
        render_prediction_plot(y_true, y_pred, lbl)
    with tabs[1]:
        render_horizon_profile(metrics)
    with tabs[2]:
        render_loss_plot(history)

    st.subheader("💾 İndir")
    d1, d2 = st.columns(2)
    safe = {k: v for k, v in metrics.items() if k != "per_horizon"}
    d1.download_button("📄 Metrik (JSON)", json.dumps(safe, indent=2),
                       f"{model_choice}_metrics.json", "application/json")
    if multistep:
        pred_df = pd.DataFrame(
            {f"gercek_h{i+1}": y_true[:, i] for i in range(y_true.shape[1])}
            | {f"tahmin_h{i+1}": y_pred[:, i] for i in range(y_pred.shape[1])}
        )
    else:
        pred_df = pd.DataFrame({"gercek": y_true, "tahmin": y_pred})
    d2.download_button("📊 Tahminler (CSV)", pred_df.to_csv(index=False),
                       f"{model_choice}_tahminler.csv", "text/csv")

# ═══════════════════════════════════════════════════════════════════════════════
# SAYFA 2 — MODEL KARŞILAŞTIRMASI
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Model Karşılaştırması":
    st.subheader("⚔️ Model Karşılaştırması — LSTM vs GRU vs Doğrusal Regresyon")
    st.caption(f"Aynı veri · Lookback {lookback}h · Horizon {horizon}h · {len(X_train)} eğitim örneği")

    results: dict[str, dict] = {}
    histories: dict[str, dict] = {}
    predictions: dict[str, tuple] = {}

    for mtype in ["lstm", "gru", "xgboost", "linear"]:
        with st.status(f"{mtype.upper()} eğitiliyor…", expanded=False) as status:
            info = st.empty()
            bar  = st.progress(0)
            # XGBoost karşılaştırmada kendi lr/patience'ını kullanmalı
            _mtype_cfg = {**dl_cfg, "horizon": horizon}
            if mtype == "xgboost":
                _mtype_cfg["learning_rate"] = xgb_lr
                _mtype_cfg["patience"]      = xgb_patience
            m = build_model(mtype, _mtype_cfg)
            m.build((X_train.shape[1], X_train.shape[2]))
            if mtype in ("lstm", "gru"):
                h = train_dl_model(m, X_train, y_train, X_val, y_val, dl_cfg, bar, info)
            else:
                h = m.fit(X_train, y_train, X_val, y_val)
                bar.progress(1.0)
            y_true, y_pred = get_predictions(m, X_test, y_test, scaler, all_columns, target_col)
            met = compute_metrics(y_true, y_pred)
            results[mtype]     = met
            histories[mtype]   = h
            predictions[mtype] = (y_true, y_pred)
            status.update(
                label=f"{mtype.upper()} — R²: {met['r2']:.4f}  RMSE: {met['rmse']:.4f}",
                state="complete",
            )

    # ── Persistence baseline ──────────────────────────────────────────────────
    y_pers = inverse_scale_column(y_pers_test, scaler, all_columns, target_col)
    y_true_ref = list(predictions.values())[0][0]  # y_true aynı tüm modeller için
    pers_metrics = compute_metrics(y_true_ref, y_pers)

    # ── Özet metrik tablosu ───────────────────────────────────────────────────
    st.subheader("📊 Karşılaştırma Tablosu")
    rows = []
    for mtype, met in results.items():
        skill = compute_skill_score(met["rmse"], pers_metrics["rmse"])
        rows.append({
            "Model": mtype.upper(),
            "R²": round(met["r2"], 4),
            "RMSE (m/s)": round(met["rmse"], 4),
            "MAE (m/s)": round(met["mae"], 4),
            "Skill Score": round(skill, 4),
        })
    rows.append({
        "Model": "PERSISTENCE",
        "R²": round(pers_metrics["r2"], 4),
        "RMSE (m/s)": round(pers_metrics["rmse"], 4),
        "MAE (m/s)": round(pers_metrics["mae"], 4),
        "Skill Score": 0.0,
    })
    cmp_df = pd.DataFrame(rows).set_index("Model")
    best_r2 = cmp_df["R²"].idxmax()
    st.dataframe(
        cmp_df.style.highlight_max(subset=["R²", "Skill Score"], color="#d4edda")
                    .highlight_min(subset=["RMSE (m/s)", "MAE (m/s)"], color="#d4edda"),
        use_container_width=True,
    )
    best_skill = cmp_df.loc[best_r2, "Skill Score"]
    skill_label = "✓ faydalı (>0.30)" if best_skill >= 0.3 else "△ geliştirilebilir"
    st.success(f"En iyi model: **{best_r2}** — R²={cmp_df.loc[best_r2, 'R²']} · Skill Score={best_skill} {skill_label}")

    # ── Bar chart ─────────────────────────────────────────────────────────────
    st.subheader("📈 Metrik Grafiği")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    colors = ["#4C72B0", "#DD8452", "#8172B2", "#55A868"]
    models = list(results.keys())
    for ax, key, ylabel in zip(axes, ["r2", "rmse", "mae"], ["R²", "RMSE (m/s)", "MAE (m/s)"]):
        vals = [results[m][key] for m in models]
        bars = ax.bar([m.upper() for m in models], vals, color=colors)
        ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=9)
        ax.set_title(ylabel); ax.set_ylabel(ylabel); ax.grid(True, alpha=0.3, axis="y")
        if key == "r2":
            ax.set_ylim(0, 1)
    plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    # ── Üst üste tahmin grafikleri ────────────────────────────────────────────
    st.subheader("📉 Tahmin Karşılaştırması")
    multistep = list(predictions.values())[0][0].ndim == 2 and list(predictions.values())[0][0].shape[1] > 1
    yt_ref = list(predictions.values())[0][0]
    ref_series = yt_ref[:, 0] if multistep else yt_ref

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(ref_series, label="Gerçek", color="black", linewidth=1.2, alpha=0.8)
    for (mtype, (yt, yp)), color in zip(predictions.items(), colors):
        yp_series = yp[:, 0] if multistep else yp
        ax.plot(yp_series, label=mtype.upper(), color=color, alpha=0.7, linewidth=0.9)
    ax.set_title("Test Seti — Tüm Modeller")
    ax.set_xlabel("Zaman adımı"); ax.set_ylabel("Rüzgar hızı (m/s)")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    # ── Horizon profili (çok adımlıysa) ──────────────────────────────────────
    if multistep:
        st.subheader("⏱ Horizon Profili Karşılaştırması")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        steps = list(range(1, horizon + 1))
        for (mtype, met), color in zip(results.items(), colors):
            if "per_horizon" not in met:
                continue
            ph = met["per_horizon"]
            axes[0].plot(steps, [m["rmse"] for m in ph], marker="o", label=mtype.upper(), color=color)
            axes[1].plot(steps, [m["r2"]   for m in ph], marker="o", label=mtype.upper(), color=color)
        for ax, ylabel in zip(axes, ["RMSE (m/s)", "R²"]):
            ax.set_xlabel("Tahmin adımı (saat)"); ax.set_ylabel(ylabel)
            ax.set_title(f"{ylabel} — horizon başına"); ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    # ── Eğitim kaybı ─────────────────────────────────────────────────────────
    st.subheader("📉 Eğitim Kaybı Karşılaştırması")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for (mtype, h), color in zip(histories.items(), colors):
        if not h.get("loss"):
            continue
        axes[0].plot(h["loss"],             label=mtype.upper(), color=color)
        axes[1].plot(h.get("val_loss", []), label=mtype.upper(), color=color)
    for ax, title in zip(axes, ["Train Loss", "Validation Loss"]):
        ax.set_title(title); ax.set_xlabel("Epoch"); ax.set_ylabel("MSE")
        ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    # ── İndir ─────────────────────────────────────────────────────────────────
    st.subheader("💾 İndir")
    safe_results = {m: {k: v for k, v in met.items() if k != "per_horizon"}
                    for m, met in results.items()}
    st.download_button(
        "📄 Karşılaştırma Raporu (JSON)",
        data=json.dumps(safe_results, indent=2, ensure_ascii=False),
        file_name="model_karsilastirma.json",
        mime="application/json",
    )

# ═══════════════════════════════════════════════════════════════════════════════
# SAYFA 3 — HİPERPARAMETRE OPTİMİZASYONU
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Hiperparametre Optimizasyonu":
    st.subheader(f"🔬 {model_choice.upper()} Hiperparametre Optimizasyonu")
    st.caption("Optuna ile Bayesian arama — her deneme val_loss'u minimize eder")

    n_trials = st.slider("Deneme sayısı (trial)", 5, 50, 20)
    st.info(
        f"**Nasıl çalışır?** Optuna {n_trials} farklı hiperparametre kombinasyonunu "
        f"sırayla dener. Her denemede model 50 epoch eğitilir, en iyi val_loss kaydedilir. "
        f"En sonunda tüm denemeler arasındaki kazananın parametreleriyle tam model eğitilir."
    )

    if not run_btn:
        st.stop()

    from src.training.optimizer import run_optimization

    st.subheader("⏳ Optimizasyon")
    bar   = st.progress(0)
    info  = st.empty()
    chart = st.empty()

    trial_vals: list[float] = []

    def _progress(current, total, best_val):
        trial_vals.append(best_val)
        bar.progress(current / total)
        info.info(f"Deneme {current}/{total} — en iyi val_loss: {best_val:.5f}")
        fig, ax = plt.subplots(figsize=(8, 2.5))
        ax.plot(trial_vals, marker="o", markersize=4, linewidth=1)
        ax.set_title("Val Loss — denemeler boyunca")
        ax.set_xlabel("Deneme"); ax.set_ylabel("Val Loss (MSE)")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        chart.pyplot(fig)
        plt.close(fig)

    opt_result = run_optimization(
        X_train, y_train, X_val, y_val,
        model_type=model_choice,
        horizon=horizon,
        n_trials=n_trials,
        progress_callback=_progress,
    )

    bar.progress(1.0)
    info.success(f"Optimizasyon tamamlandı! En iyi val_loss: {opt_result['best_val_loss']:.5f}")

    # ── En iyi parametreler ───────────────────────────────────────────────────
    best = opt_result["best_params"]
    st.subheader("🏆 En İyi Hiperparametreler")
    if model_choice == "xgboost":
        bp1, bp2, bp3 = st.columns(3)
        bp1.metric("Ağaç sayısı",   best["n_estimators"])
        bp2.metric("Maks derinlik", best["max_depth"])
        bp3.metric("Learning rate", best["learning_rate"])
        b1, b2, b3 = st.columns(3)
        b1.metric("Subsample",        best["subsample"])
        b2.metric("Colsample bytree", best["colsample_bytree"])
        b3.metric("Val RMSE",         f"{opt_result['best_val_loss']:.5f}")
    else:
        bp1, bp2, bp3, bp4 = st.columns(4)
        bp1.metric("Katman 1 nöron", best["units"][0])
        bp2.metric("Katman 2 nöron", best["units"][1])
        bp3.metric("Dropout",        best["dropout"])
        bp4.metric("Learning rate",  best["learning_rate"])
        b1, b2 = st.columns(2)
        b1.metric("Batch size", best["batch_size"])
        b2.metric("Val Loss",   f"{opt_result['best_val_loss']:.5f}")

    # ── Tüm denemeler tablosu ─────────────────────────────────────────────────
    with st.expander("📋 Tüm denemeler"):
        rows = []
        for t in sorted(opt_result["trials"], key=lambda x: x["val_loss"]):
            rows.append({
                "Deneme": t["number"] + 1,
                "Val Loss": round(t["val_loss"], 5),
                **t["params"],
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # ── En iyi parametrelerle tam eğitim ──────────────────────────────────────
    st.subheader("🚀 En İyi Parametrelerle Tam Eğitim")
    bar2  = st.progress(0)
    info2 = st.empty()

    final_model = build_model(model_choice, best)
    final_model.build((X_train.shape[1], X_train.shape[2]))
    if model_choice == "xgboost":
        history = final_model.fit(X_train, y_train, X_val, y_val)
        bar2.progress(1.0)
        info2.success("Tam eğitim tamamlandı!")
    else:
        history = train_dl_model(
            final_model, X_train, y_train, X_val, y_val,
            best, bar2, info2,
        )
        info2.success("Tam eğitim tamamlandı!")

    y_true, y_pred = get_predictions(
        final_model, X_test, y_test, scaler, all_columns, target_col
    )
    metrics = compute_metrics(y_true, y_pred)

    st.subheader("📊 Sonuçlar")
    m1, m2, m3 = st.columns(3)
    m1.metric("R²",       f"{metrics['r2']:.4f}")
    m2.metric("RMSE m/s", f"{metrics['rmse']:.4f}")
    m3.metric("MAE m/s",  f"{metrics['mae']:.4f}")

    tabs = st.tabs(["📈 Tahmin", "📉 Eğitim Kaybı"])
    with tabs[0]:
        render_prediction_plot(y_true, y_pred,
                               f"{model_choice.upper()} (optimize) — Tahmin vs Gerçek")
    with tabs[1]:
        render_loss_plot(history)

    st.subheader("💾 İndir")
    d1, d2 = st.columns(2)
    safe = {k: v for k, v in metrics.items() if k != "per_horizon"}
    d1.download_button("📄 Metrik (JSON)",
                       json.dumps({"metrics": safe, "best_params": best}, indent=2),
                       f"{model_choice}_optimized_metrics.json", "application/json")
    pred_df = pd.DataFrame({"gercek": y_true.ravel(), "tahmin": y_pred.ravel()})
    d2.download_button("📊 Tahminler (CSV)", pred_df.to_csv(index=False),
                       f"{model_choice}_optimized_tahminler.csv", "text/csv")

# ═══════════════════════════════════════════════════════════════════════════════
# SAYFA 4 — BELİRSİZLİK TAHMİNİ
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Belirsizlik Tahmini":
    st.subheader(f"🎯 {model_choice.upper()} — Monte Carlo Dropout Belirsizlik Tahmini")
    st.info(
        "**Nasıl çalışır?** Dropout katmanları test sırasında da açık tutulur. "
        "Aynı girdi için N kez tahmin yapılır — her seferinde farklı nöronlar rastgele kapatılır. "
        "Bu N tahminin **ortalaması** nokta tahmini, **standart sapması** ise belirsizlik ölçüsüdür. "
        "Güven bandı: ortalama ± 1.96 × std → %95 güven aralığı."
    )

    n_samples = st.slider("Monte Carlo örneklem sayısı (N)", 20, 200, 100, step=10)
    st.caption("N arttıkça belirsizlik tahmini daha kararlı olur, süre de uzar.")

    if not run_btn:
        st.stop()

    from src.reporting.visualizer import plot_uncertainty

    # Eğitim
    with st.status(f"{model_choice.upper()} eğitiliyor…", expanded=False) as status:
        info = st.empty()
        bar  = st.progress(0)
        model = build_model(model_choice, {**dl_cfg, "horizon": horizon})
        model.build((X_train.shape[1], X_train.shape[2]))
        history = train_dl_model(model, X_train, y_train, X_val, y_val, dl_cfg, bar, info)
        status.update(label=f"{model_choice.upper()} eğitimi tamamlandı", state="complete")

    # MC Dropout inference
    with st.spinner(f"Monte Carlo örneklemesi yapılıyor ({n_samples} tekrar)…"):
        y_mean_s, y_std_s = model.predict_with_uncertainty(X_test, n_samples=n_samples)
        y_mean = inverse_scale_column(y_mean_s, scaler, all_columns, target_col)
        y_std  = inverse_scale_column(
            y_mean_s + y_std_s, scaler, all_columns, target_col
        ) - y_mean
        y_std  = np.abs(y_std)
        y_true_unc = inverse_scale_column(y_test, scaler, all_columns, target_col)

    if y_true_unc.ndim == 2:
        y_true_unc = y_true_unc[:, 0]
        y_mean     = y_mean[:, 0]
        y_std      = y_std[:, 0]

    metrics = compute_metrics(y_true_unc, y_mean)

    st.subheader("📊 Sonuçlar")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("R²",              f"{metrics['r2']:.4f}")
    m2.metric("RMSE (m/s)",      f"{metrics['rmse']:.4f}")
    m3.metric("MAE (m/s)",       f"{metrics['mae']:.4f}")
    m4.metric("Ort. belirsizlik", f"±{y_std.mean():.3f} m/s")

    # Güven bandı grafik
    st.subheader("📈 Tahmin + %95 Güven Bandı")
    lower = y_mean - 1.96 * y_std
    upper = y_mean + 1.96 * y_std
    x = np.arange(len(y_true_unc))

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    axes[0].fill_between(x, lower, upper, alpha=0.25, color="steelblue", label="%95 güven aralığı")
    axes[0].plot(x, y_true_unc, color="black",     linewidth=0.9, label="Gerçek",            alpha=0.8)
    axes[0].plot(x, y_mean,     color="steelblue", linewidth=0.9, label="Tahmin (ortalama)", alpha=0.9)
    axes[0].set_title(f"{model_choice.upper()} — Tahmin + %95 Güven Bandı")
    axes[0].set_xlabel("Zaman adımı"); axes[0].set_ylabel("Rüzgar hızı (m/s)")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(x, y_std, color="orange", linewidth=0.8)
    axes[1].fill_between(x, 0, y_std, alpha=0.3, color="orange")
    axes[1].set_title("Tahmin Belirsizliği (Std Dev)")
    axes[1].set_xlabel("Zaman adımı"); axes[1].set_ylabel("Std Dev (m/s)")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    # Eğitim kaybı
    with st.expander("📉 Eğitim Kaybı"):
        render_loss_plot(history)

    # İndir
    st.subheader("💾 İndir")
    d1, d2 = st.columns(2)
    unc_df = pd.DataFrame({
        "gercek":   y_true_unc,
        "tahmin":   y_mean,
        "std":      y_std,
        "alt_band": lower,
        "ust_band": upper,
    })
    d1.download_button(
        "📊 Tahmin + Güven Bandı (CSV)",
        data=unc_df.to_csv(index=False),
        file_name=f"{model_choice}_belirsizlik.csv",
        mime="text/csv",
    )
    safe = {k: v for k, v in metrics.items() if k != "per_horizon"}
    safe["mean_uncertainty"] = float(y_std.mean())
    d2.download_button(
        "📄 Metrik (JSON)",
        data=json.dumps(safe, indent=2),
        file_name=f"{model_choice}_belirsizlik_metrics.json",
        mime="application/json",
    )
