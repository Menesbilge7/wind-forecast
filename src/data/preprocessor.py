from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple


def add_cyclic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Zaman sütunlarını sin/cos çiftlerine dönüştürür (döngüsel kodlama).
    Desteklenen saat sütunları : saat, hour, hr, ...
    Desteklenen gün sütunları  : gunler, day_of_year, doy, julian, ...
    datetime/timestamp sütunu varsa otomatik olarak saat ve gün çıkarılır.
    """
    _HOUR_NAMES = {"saat", "hour", "hr", "saat_utc", "hour_utc", "saat_local"}
    _DOY_NAMES  = {"gunler", "day_of_year", "doy", "julian", "julian_day", "jday", "yday", "day_of_yr"}
    _DT_NAMES   = {"datetime", "time", "timestamp", "date_time", "tarih", "zaman", "date"}

    out = df.copy()
    col_lower = {c.lower(): c for c in out.columns}

    # Saat sütunu
    _hour_col = next((col_lower[k] for k in _HOUR_NAMES if k in col_lower), None)
    if _hour_col:
        out["saat_sin"] = np.sin(2 * np.pi * out[_hour_col] / 24)
        out["saat_cos"] = np.cos(2 * np.pi * out[_hour_col] / 24)
        out = out.drop(columns=[_hour_col])

    # Gün-yılı sütunu
    _doy_col = next((col_lower[k] for k in _DOY_NAMES if k in col_lower), None)
    if _doy_col:
        out["gunler_sin"] = np.sin(2 * np.pi * out[_doy_col] / 365)
        out["gunler_cos"] = np.cos(2 * np.pi * out[_doy_col] / 365)
        out = out.drop(columns=[_doy_col])

    # datetime sütunu — saat/gün henüz eklenmemişse çıkar
    _dt_col = next((col_lower[k] for k in _DT_NAMES if k in col_lower), None)
    if _dt_col:
        try:
            _dt = pd.to_datetime(out[_dt_col])
            if "saat_sin" not in out.columns:
                out["saat_sin"] = np.sin(2 * np.pi * _dt.dt.hour / 24)
                out["saat_cos"] = np.cos(2 * np.pi * _dt.dt.hour / 24)
            if "gunler_sin" not in out.columns:
                out["gunler_sin"] = np.sin(2 * np.pi * _dt.dt.day_of_year / 365)
                out["gunler_cos"] = np.cos(2 * np.pi * _dt.dt.day_of_year / 365)
        except Exception:
            pass
        # datetime sütununu drop etme — select_dtypes zaten sayısal olmadığı için filtreler

    return out


def add_lag_features(
    df: pd.DataFrame,
    target_col: str,
    lags: list[int] | None = None,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """
    Hedef sütun için gecikmeli (lag) ve kayan pencere (rolling) özellikler ekler.
    XGBoost gibi düz tablo modelleri için zamansal örüntüleri açık hale getirir.
    Tüm özellikler geçmiş değerlere dayalıdır — veri sızıntısı yoktur.
    NaN içeren başlangıç satırları atılır.
    """
    if lags is None:
        lags = [1, 2, 3, 6, 12, 24]
    if windows is None:
        windows = [3, 6, 12, 24]

    out = df.copy()
    s = out[target_col]

    for lag in lags:
        out[f"{target_col}_lag{lag}"] = s.shift(lag)

    for w in windows:
        out[f"{target_col}_ma{w}"]  = s.shift(1).rolling(w).mean()

    for w in [3, 6]:
        out[f"{target_col}_std{w}"] = s.shift(1).rolling(w).std()

    out[f"{target_col}_diff1"] = s.diff(1)
    out[f"{target_col}_diff3"] = s.diff(3)

    return out.dropna().reset_index(drop=True)


def split_data(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]


def fit_scaler(train: pd.DataFrame, columns: list[str]) -> MinMaxScaler:
    scaler = MinMaxScaler()
    scaler.fit(train[columns])
    return scaler


def scale(df: pd.DataFrame, scaler: MinMaxScaler, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    out[columns] = scaler.transform(df[columns])
    return out


def inverse_scale_column(
    values: np.ndarray,
    scaler: MinMaxScaler,
    columns: list[str],
    target: str,
) -> np.ndarray:
    """
    values shape: (n,) veya (n, horizon)
    Döndürür:     (n,)  veya (n, horizon)
    """
    idx = columns.index(target)
    values = np.array(values)
    multistep = values.ndim == 2 and values.shape[1] > 1

    if multistep:
        horizon = values.shape[1]
        result = np.zeros_like(values)
        for h in range(horizon):
            dummy = np.zeros((len(values), len(columns)))
            dummy[:, idx] = values[:, h]
            result[:, h] = scaler.inverse_transform(dummy)[:, idx]
        return result
    else:
        dummy = np.zeros((len(values), len(columns)))
        dummy[:, idx] = values.ravel()
        return scaler.inverse_transform(dummy)[:, idx]
