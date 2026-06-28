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
