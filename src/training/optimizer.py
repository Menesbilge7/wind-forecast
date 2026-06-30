from __future__ import annotations
import optuna
import numpy as np
from typing import Callable


optuna.logging.set_verbosity(optuna.logging.WARNING)


def _make_dl_objective(
    X_train, y_train,
    X_val,   y_val,
    model_type: str,
    horizon: int,
    input_shape: tuple,
) -> Callable:

    def objective(trial: optuna.Trial) -> float:
        from tensorflow import keras

        units_1  = trial.suggest_categorical("units_1",  [32, 64, 128, 256])
        units_2  = trial.suggest_categorical("units_2",  [16, 32, 64, 128])
        dropout  = trial.suggest_float("dropout",        0.0, 0.4, step=0.1)
        lr       = trial.suggest_categorical("lr",       [0.01, 0.001, 0.0005, 0.0001])
        batch_sz = trial.suggest_categorical("batch_size", [16, 32, 64])

        cfg = {
            "units": [units_1, units_2],
            "dropout": dropout,
            "learning_rate": lr,
            "batch_size": batch_sz,
            "epochs": 50,
            "patience": 7,
            "horizon": horizon,
        }

        if model_type == "lstm":
            from src.models.lstm import LSTMModel
            model = LSTMModel(cfg)
        else:
            from src.models.gru import GRUModel
            model = GRUModel(cfg)

        model.build(input_shape)

        early = keras.callbacks.EarlyStopping(patience=7, restore_best_weights=True)
        history = model.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=50,
            batch_size=batch_sz,
            callbacks=[early],
            verbose=0,
        )
        return min(history.history["val_loss"])

    return objective


def _make_xgb_objective(
    X_train, y_train,
    X_val,   y_val,
) -> Callable:

    Xf_train = X_train.reshape(len(X_train), -1)
    Xf_val   = X_val.reshape(len(X_val), -1)
    yt_train = y_train.ravel()
    yt_val   = y_val.ravel()

    def objective(trial: optuna.Trial) -> float:
        from xgboost import XGBRegressor

        n_estimators     = trial.suggest_int("n_estimators",     100, 1000, step=100)
        max_depth        = trial.suggest_int("max_depth",        3, 10)
        lr               = trial.suggest_categorical("lr",       [0.3, 0.1, 0.05, 0.01])
        subsample        = trial.suggest_float("subsample",      0.6, 1.0, step=0.1)
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.6, 1.0, step=0.1)

        model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=lr,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            early_stopping_rounds=20,
            eval_metric="rmse",
            random_state=42,
            n_jobs=-1,
        )
        model.fit(
            Xf_train, yt_train,
            eval_set=[(Xf_train, yt_train), (Xf_val, yt_val)],
            verbose=False,
        )
        return min(model.evals_result_["validation_1"]["rmse"])

    return objective


def run_optimization(
    X_train, y_train,
    X_val,   y_val,
    model_type: str,
    horizon: int,
    n_trials: int = 20,
    progress_callback: Callable | None = None,
) -> dict:
    """
    En iyi hiperparametreleri döner.
    progress_callback(trial_number, n_trials, best_value) şeklinde çağrılır.
    """
    input_shape = (X_train.shape[1], X_train.shape[2])

    if model_type == "xgboost":
        objective = _make_xgb_objective(X_train, y_train, X_val, y_val)
    else:
        objective = _make_dl_objective(X_train, y_train, X_val, y_val,
                                       model_type, horizon, input_shape)

    study = optuna.create_study(direction="minimize", storage=None)

    def _cb(study, trial):
        if progress_callback:
            progress_callback(trial.number + 1, n_trials, study.best_value)

    study.optimize(objective, n_trials=n_trials, callbacks=[_cb], show_progress_bar=False)

    best = study.best_params

    if model_type == "xgboost":
        best["learning_rate"]    = best.pop("lr")
        best["horizon"]          = horizon
        best["patience"]         = 20
        best["subsample"]        = best.get("subsample", 0.8)
        best["colsample_bytree"] = best.get("colsample_bytree", 0.8)
    else:
        best["epochs"]        = 100
        best["patience"]      = 10
        best["horizon"]       = horizon
        best["units"]         = [best.pop("units_1"), best.pop("units_2")]
        best["learning_rate"] = best.pop("lr")

    return {
        "best_params": best,
        "best_val_loss": study.best_value,
        "trials": [
            {
                "number": t.number,
                "val_loss": t.value,
                "params": t.params,
            }
            for t in study.trials
        ],
    }
