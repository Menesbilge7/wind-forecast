from __future__ import annotations
import numpy as np
from .base import BaseModel


class XGBoostModel(BaseModel):
    """
    XGBoost regresyon modeli. Girdi dizisi (lookback x feature) düzleştirilerek
    2D tablo olarak beslenir. Hem tek adımlı hem çok adımlı horizon desteklenir.
    """

    def build(self, input_shape: tuple) -> None:
        from xgboost import XGBRegressor
        self._horizon  = self.cfg.get("horizon", 1)
        self._patience = self.cfg.get("patience", 20)
        params = dict(
            n_estimators          = self.cfg.get("n_estimators", 500),
            max_depth             = self.cfg.get("max_depth", 6),
            learning_rate         = self.cfg.get("learning_rate", 0.05),
            subsample             = self.cfg.get("subsample", 0.8),
            colsample_bytree      = self.cfg.get("colsample_bytree", 0.8),
            early_stopping_rounds = self._patience,
            eval_metric           = "rmse",
            random_state          = 42,
            n_jobs                = -1,
        )
        if self._horizon > 1:
            from sklearn.multioutput import MultiOutputRegressor
            self.model = MultiOutputRegressor(XGBRegressor(**params))
            self._multi = True
        else:
            self.model = XGBRegressor(**params)
            self._multi = False

    def fit(self, X_train, y_train, X_val=None, y_val=None) -> dict:
        Xf_train = X_train.reshape(len(X_train), -1)

        # XGBoost (n,) bekler; sequencer (n,1) döndürebilir
        yt_train = y_train.ravel() if not self._multi else y_train

        fit_kwargs: dict = {"verbose": False}

        if X_val is not None and not self._multi:
            Xf_val = X_val.reshape(len(X_val), -1)
            yt_val = y_val.ravel()
            fit_kwargs["eval_set"] = [(Xf_train, yt_train), (Xf_val, yt_val)]

        self.model.fit(Xf_train, yt_train, **fit_kwargs)

        # Eğitim kaybı geçmişi
        if not self._multi and hasattr(self.model, "evals_result_"):
            res = self.model.evals_result_
            train_loss = res.get("validation_0", {}).get("rmse", [])
            val_loss   = res.get("validation_1", {}).get("rmse", [])
            return {"loss": train_loss, "val_loss": val_loss}
        return {}

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xf = X.reshape(len(X), -1)
        out = self.model.predict(Xf)
        if self._horizon == 1 and out.ndim == 1:
            out = out.reshape(-1, 1)
        return out

    def save(self, path: str) -> None:
        import joblib
        joblib.dump(self.model, path)

    def load(self, path: str) -> None:
        import joblib
        self.model = joblib.load(path)
