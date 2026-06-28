from __future__ import annotations
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Tek adım (n,) veya çok adım (n, horizon) için metrik hesaplar.
    Çok adımda her horizon için ayrı metrik + genel ortalama döner.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    multistep = y_true.ndim == 2 and y_true.shape[1] > 1

    if not multistep:
        return _single_step_metrics(y_true.ravel(), y_pred.ravel())

    horizon = y_true.shape[1]
    per_horizon = []
    for h in range(horizon):
        per_horizon.append(_single_step_metrics(y_true[:, h], y_pred[:, h]))

    overall = _single_step_metrics(y_true.ravel(), y_pred.ravel())
    overall["per_horizon"] = per_horizon
    return overall


def _single_step_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "mape": float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100),
    }


def print_metrics(metrics: dict, label: str = "") -> None:
    prefix = f"[{label}] " if label else ""
    print(f"{prefix}RMSE: {metrics['rmse']:.4f}  MAE: {metrics['mae']:.4f}  "
          f"R²: {metrics['r2']:.4f}")
    if "per_horizon" in metrics:
        for h, m in enumerate(metrics["per_horizon"], 1):
            print(f"  +{h:2d}h  RMSE: {m['rmse']:.4f}  MAE: {m['mae']:.4f}  R²: {m['r2']:.4f}")
