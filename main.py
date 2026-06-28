import yaml
from pathlib import Path

from src.data.loader import load_from_config
from src.data.preprocessor import add_cyclic_features, split_data, fit_scaler, scale
from src.data.sequencer import make_sequences
from src.models.lstm import LSTMModel
from src.models.gru import GRUModel
from src.models.linear import LinearModel
from src.training.trainer import run_training


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_model(model_type: str, cfg: dict):
    models = {
        "lstm": LSTMModel,
        "gru": GRUModel,
        "linear": LinearModel,
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
    print(f"Veri yuklendi: {df.shape[0]} satir, sutunlar: {list(df.columns)}")

    all_columns = data_cfg["feature_columns"] + [data_cfg["target_column"]]
    all_columns = list(dict.fromkeys(all_columns))

    train_df, val_df, test_df = split_data(df, data_cfg["train_ratio"], data_cfg["val_ratio"])
    scaler = fit_scaler(train_df, all_columns)

    train_s = scale(train_df, scaler, all_columns)
    val_s = scale(val_df, scaler, all_columns)
    test_s = scale(test_df, scaler, all_columns)

    feature_cols = data_cfg["feature_columns"]
    target_col = data_cfg["target_column"]
    lookback = seq_cfg["lookback"]
    horizon = seq_cfg["horizon"]

    X_train, y_train = make_sequences(train_s, feature_cols, target_col, lookback, horizon)
    X_val, y_val = make_sequences(val_s, feature_cols, target_col, lookback, horizon)
    X_test, y_test = make_sequences(test_s, feature_cols, target_col, lookback, horizon)

    print(f"Sekans boyutlari — Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    model_type = cfg["model"]["type"]
    model = get_model(model_type, {**cfg["model"], "horizon": horizon})
    run_training(
        model, X_train, y_train, X_val, y_val, X_test, y_test,
        scaler, all_columns, target_col, cfg, label=model_type,
    )


if __name__ == "__main__":
    import sys
    config = sys.argv[1] if len(sys.argv) > 1 else "configs/terkos_baseline.yaml"
    main(config)
