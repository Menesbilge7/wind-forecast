import yaml
from pathlib import Path

from src.data.loader import load_from_config
from src.data.preprocessor import add_cyclic_features, add_lag_features, split_data, fit_scaler, scale
from src.data.sequencer import make_sequences, make_persistence_sequences
from src.models.lstm import LSTMModel
from src.models.gru import GRUModel
from src.models.linear import LinearModel
from src.models.xgboost_model import XGBoostModel
from src.training.trainer import run_training
from src.evaluation.metrics import compute_metrics, compute_skill_score
from src.data.preprocessor import inverse_scale_column


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_model(model_type: str, cfg: dict):
    models = {
        "lstm": LSTMModel,
        "gru": GRUModel,
        "linear": LinearModel,
        "xgboost": XGBoostModel,
    }
    cls = models.get(model_type.lower())
    if cls is None:
        raise ValueError(f"Bilinmeyen model tipi: {model_type}")
    return cls(cfg)


def main(config_path: str = "configs/terkos_baseline.yaml"):
    cfg = load_config(config_path)
    data_cfg = cfg["data"]
    seq_cfg = cfg["sequence"]
    model_cfg = {**cfg["model"], **cfg.get("output", {}), "output": cfg.get("output", {})}

    df = load_from_config(data_cfg)
    df = add_cyclic_features(df)
    model_type = cfg["model"]["type"].lower()
    if model_type == "xgboost":
        df = add_lag_features(df, data_cfg["target_column"])
    print(f"Veri yuklendi: {df.shape[0]} satir, sutunlar: {list(df.columns)}")

    feature_cols_base = data_cfg["feature_columns"]
    if model_type == "xgboost":
        lag_cols = [c for c in df.columns if c.startswith(f"{data_cfg['target_column']}_")]
        feature_cols_base = list(dict.fromkeys(feature_cols_base + lag_cols))
    all_columns = list(dict.fromkeys(feature_cols_base + [data_cfg["target_column"]]))

    train_df, val_df, test_df = split_data(df, data_cfg["train_ratio"], data_cfg["val_ratio"])
    scaler = fit_scaler(train_df, all_columns)

    train_s = scale(train_df, scaler, all_columns)
    val_s = scale(val_df, scaler, all_columns)
    test_s = scale(test_df, scaler, all_columns)

    feature_cols = feature_cols_base
    target_col = data_cfg["target_column"]
    lookback = seq_cfg["lookback"]
    horizon = seq_cfg["horizon"]

    X_train, y_train = make_sequences(train_s, feature_cols, target_col, lookback, horizon)
    X_val, y_val = make_sequences(val_s, feature_cols, target_col, lookback, horizon)
    X_test, y_test = make_sequences(test_s, feature_cols, target_col, lookback, horizon)

    print(f"Sekans boyutlari — Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # Persistence baseline
    y_pers_scaled = make_persistence_sequences(test_s, target_col, lookback, horizon)
    y_pers = inverse_scale_column(y_pers_scaled, scaler, all_columns, target_col)
    y_true_ref = inverse_scale_column(y_test, scaler, all_columns, target_col)
    pers_metrics = compute_metrics(y_true_ref, y_pers)
    print(f"\n[Persistence] RMSE: {pers_metrics['rmse']:.4f}  MAE: {pers_metrics['mae']:.4f}  R²: {pers_metrics['r2']:.4f}")

    model = get_model(model_type, {**cfg["model"], "horizon": horizon})
    model_metrics = run_training(
        model, X_train, y_train, X_val, y_val, X_test, y_test,
        scaler, all_columns, target_col, cfg, label=model_type,
    )
    skill = compute_skill_score(model_metrics["rmse"], pers_metrics["rmse"])
    skill_label = "faydali (>=0.30)" if skill >= 0.3 else "gelistirilebilir"
    print(f"Skill Score ({model_type}): {skill:.4f}  [{skill_label}]")


if __name__ == "__main__":
    import sys
    config = sys.argv[1] if len(sys.argv) > 1 else "configs/terkos_baseline.yaml"
    main(config)
