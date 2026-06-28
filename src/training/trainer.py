from __future__ import annotations
import json
import numpy as np
from pathlib import Path
from src.models.base import BaseModel
from src.evaluation.metrics import compute_metrics, print_metrics
from src.reporting.visualizer import plot_predictions, plot_multistep, plot_training_history


def run_training(
    model: BaseModel,
    X_train, y_train,
    X_val, y_val,
    X_test, y_test,
    scaler,
    all_columns: list[str],
    target_column: str,
    cfg: dict,
    label: str = "model",
) -> dict:
    input_shape = (X_train.shape[1], X_train.shape[2])
    model.build(input_shape)

    print(f"\n--- {label.upper()} egitimi basliyor ---")
    history = model.fit(X_train, y_train, X_val, y_val)

    from src.data.preprocessor import inverse_scale_column
    y_pred_scaled = model.predict(X_test)
    y_pred = inverse_scale_column(y_pred_scaled, scaler, all_columns, target_column)
    y_true = inverse_scale_column(y_test,         scaler, all_columns, target_column)

    metrics = compute_metrics(y_true, y_pred)
    print_metrics(metrics, label=label)

    reports_dir = Path(cfg.get("output", {}).get("reports_dir", "outputs/reports"))
    reports_dir.mkdir(parents=True, exist_ok=True)
    metrics_to_save = {k: v for k, v in metrics.items() if k != "per_horizon"}
    with open(reports_dir / f"{label}_metrics.json", "w") as f:
        json.dump(metrics_to_save, f, indent=2)

    out_dir = Path(cfg.get("output", {}).get("models_dir", "outputs/models"))
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(out_dir / f"{label}.keras"))

    import joblib
    joblib.dump(scaler, out_dir / f"{label}_scaler.pkl")
    _lb = int(X_train.shape[1])
    _hz = int(y_train.shape[1]) if y_train.ndim == 2 else 1
    with open(out_dir / f"{label}_meta.json", "w", encoding="utf-8") as _mf:
        json.dump({"all_columns": all_columns, "target_column": target_column,
                   "lookback": _lb, "horizon": _hz}, _mf)

    plots_dir = cfg.get("output", {}).get("plots_dir", "outputs/plots")
    multistep = y_true.ndim == 2 and y_true.shape[1] > 1

    if multistep:
        plot_multistep(
            y_true, y_pred,
            per_horizon_metrics=metrics["per_horizon"],
            title=f"{label} — Çok Adımlı Tahmin Hata Profili",
            save_path=f"{plots_dir}/{label}_multistep.png",
        )
        # ilk ve son adım için zaman serisi grafiği
        plot_predictions(
            y_true[:, 0], y_pred[:, 0],
            title=f"{label} — +1h Tahmin vs Gerçek",
            save_path=f"{plots_dir}/{label}_predictions_h1.png",
        )
        plot_predictions(
            y_true[:, -1], y_pred[:, -1],
            title=f"{label} — +{y_true.shape[1]}h Tahmin vs Gerçek",
            save_path=f"{plots_dir}/{label}_predictions_hN.png",
        )
    else:
        plot_predictions(
            y_true, y_pred,
            title=f"{label} — Tahmin vs Gerçek",
            save_path=f"{plots_dir}/{label}_predictions.png",
        )

    if isinstance(history, dict) and "loss" in history:
        plot_training_history(history, save_path=f"{plots_dir}/{label}_loss.png")

    return metrics
