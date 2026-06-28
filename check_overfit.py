import warnings; warnings.filterwarnings("ignore")
import os; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import yaml
from src.data.loader import load_from_config
from src.data.preprocessor import split_data, fit_scaler, scale, inverse_scale_column
from src.data.sequencer import make_sequences
from src.evaluation.metrics import compute_metrics
from tensorflow import keras

cfg = yaml.safe_load(open("configs/terkos_baseline.yaml"))
data_cfg = cfg["data"]
seq_cfg = cfg["sequence"]
all_columns = list(dict.fromkeys(data_cfg["feature_columns"] + [data_cfg["target_column"]]))

df = load_from_config(data_cfg)
train_df, val_df, test_df = split_data(df, data_cfg["train_ratio"], data_cfg["val_ratio"])
scaler = fit_scaler(train_df, all_columns)

fc = data_cfg["feature_columns"]
tc = data_cfg["target_column"]
lb = seq_cfg["lookback"]

X_train, y_train = make_sequences(scale(train_df, scaler, all_columns), fc, tc, lb)
X_test,  y_test  = make_sequences(scale(test_df,  scaler, all_columns), fc, tc, lb)

print(f"{'Model':<8} {'Set':<6} {'R2':>7} {'RMSE':>7} {'MAE':>7}")
print("-" * 40)

for name in ["lstm", "gru"]:
    model = keras.models.load_model(f"outputs/models/{name}.keras")
    for split_name, X, y in [("TRAIN", X_train, y_train), ("TEST", X_test, y_test)]:
        yp = inverse_scale_column(model.predict(X, verbose=0), scaler, all_columns, tc)
        yt = inverse_scale_column(y, scaler, all_columns, tc)
        m = compute_metrics(yt, yp)
        print(f"{name.upper():<8} {split_name:<6} {m['r2']:>7.4f} {m['rmse']:>7.4f} {m['mae']:>7.4f}")
    print()
