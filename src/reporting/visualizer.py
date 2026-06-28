from __future__ import annotations
import matplotlib
matplotlib.use("Agg")  # ekran gerektirmeden dosyaya kaydet
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Tahmin vs Gerçek",
    save_path: str | None = None,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    axes[0].plot(y_true, label="Gerçek", alpha=0.8)
    axes[0].plot(y_pred, label="Tahmin", alpha=0.8)
    axes[0].set_title(title)
    axes[0].set_xlabel("Zaman adımı")
    axes[0].set_ylabel("Rüzgar hızı (m/s)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(y_true, y_pred, alpha=0.4, s=10)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    axes[1].plot(lims, lims, "r--", linewidth=1)
    axes[1].set_xlabel("Gerçek")
    axes[1].set_ylabel("Tahmin")
    axes[1].set_title("Scatter: Gerçek vs Tahmin")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_multistep(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    per_horizon_metrics: list[dict],
    title: str = "Çok Adımlı Tahmin",
    save_path: str | None = None,
) -> None:
    """y_true ve y_pred shape: (n_samples, horizon)"""
    horizon = y_true.shape[1]
    steps = list(range(1, horizon + 1))

    rmse_list = [m["rmse"] for m in per_horizon_metrics]
    mae_list  = [m["mae"]  for m in per_horizon_metrics]
    r2_list   = [m["r2"]   for m in per_horizon_metrics]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(title)

    axes[0].plot(steps, rmse_list, marker="o")
    axes[0].set_title("RMSE — horizon başına")
    axes[0].set_xlabel("Tahmin adımı (saat)")
    axes[0].set_ylabel("RMSE (m/s)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(steps, mae_list, marker="o", color="orange")
    axes[1].set_title("MAE — horizon başına")
    axes[1].set_xlabel("Tahmin adımı (saat)")
    axes[1].set_ylabel("MAE (m/s)")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(steps, r2_list, marker="o", color="green")
    axes[2].set_title("R² — horizon başına")
    axes[2].set_xlabel("Tahmin adımı (saat)")
    axes[2].set_ylabel("R²")
    axes[2].set_ylim(0, 1)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_uncertainty(
    y_true: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    n_std: float = 1.96,
    title: str = "Belirsizlik Tahmini",
    save_path: str | None = None,
) -> None:
    """
    y_mean ± n_std * y_std aralığını güven bandı olarak çizer.
    n_std=1.96 → %95 güven aralığı (normal dağılım varsayımıyla).
    """
    y_true = y_true.ravel()
    y_mean = y_mean.ravel()
    y_std  = y_std.ravel()
    x = np.arange(len(y_true))
    lower = y_mean - n_std * y_std
    upper = y_mean + n_std * y_std

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    axes[0].fill_between(x, lower, upper, alpha=0.25, color="steelblue", label="%95 güven aralığı")
    axes[0].plot(x, y_true, color="black",     linewidth=0.9, label="Gerçek",            alpha=0.8)
    axes[0].plot(x, y_mean, color="steelblue", linewidth=0.9, label="Tahmin (ortalama)", alpha=0.9)
    axes[0].set_title(title)
    axes[0].set_xlabel("Zaman adımı")
    axes[0].set_ylabel("Rüzgar hızı (m/s)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(x, y_std, color="orange", linewidth=0.8)
    axes[1].fill_between(x, 0, y_std, alpha=0.3, color="orange")
    axes[1].set_title("Tahmin Belirsizliği (Std Dev)")
    axes[1].set_xlabel("Zaman adımı")
    axes[1].set_ylabel("Std Dev (m/s)")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_training_history(history: dict, save_path: str | None = None) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(history.get("loss", []), label="Train Loss")
    if "val_loss" in history:
        ax.plot(history["val_loss"], label="Val Loss")
    ax.set_title("Eğitim Kaybı")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
