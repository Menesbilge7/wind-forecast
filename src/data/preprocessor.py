from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple


def add_cyclic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    saat (0-23) ve gunler (1-365) sütunlarını sin/cos çiftlerine dönüştürür.
    Model, zamanın döngüsel doğasını (23:00 → 00:00, 31 Ara → 1 Oca) bu şekilde öğrenir.
    Orijinal sütunlar kaldırılır.
    """
    out = df.copy()
    if "saat" in out.columns:
        out["saat_sin"] = np.sin(2 * np.pi * out["saat"] / 24)
        out["saat_cos"] = np.cos(2 * np.pi * out["saat"] / 24)
        out = out.drop(columns=["saat"])
    if "gunler" in out.columns:
        out["gunler_sin"] = np.sin(2 * np.pi * out["gunler"] / 365)
        out["gunler_cos"] = np.cos(2 * np.pi * out["gunler"] / 365)
        out = out.drop(columns=["gunler"])
    return out


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
