from __future__ import annotations
import numpy as np
import pandas as pd


def make_sequences(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    lookback: int,
    horizon: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    features = df[feature_columns].values
    target = df[target_column].values

    X, y = [], []
    for i in range(lookback, len(df) - horizon + 1):
        X.append(features[i - lookback : i])
        y.append(target[i : i + horizon])

    return np.array(X), np.array(y)


def make_persistence_sequences(
    df: pd.DataFrame,
    target_column: str,
    lookback: int,
    horizon: int = 1,
) -> np.ndarray:
    """
    Persistence baseline: t+h tahmini = t anındaki son bilinen değer.
    Rüzgar tahmininde standart kıyas noktası budur.
    Döndürür: (n, horizon) — her sekans için horizon adımlık sabit tahmin.
    """
    target = df[target_column].values
    y_pers = [
        np.full(horizon, target[i - 1])
        for i in range(lookback, len(df) - horizon + 1)
    ]
    return np.array(y_pers)
