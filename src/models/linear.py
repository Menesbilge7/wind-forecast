import numpy as np
from sklearn.linear_model import LinearRegression
from .base import BaseModel


class LinearModel(BaseModel):
    def build(self, input_shape: tuple) -> None:
        self.model = LinearRegression()
        self._input_shape = input_shape

    def fit(self, X_train, y_train, X_val=None, y_val=None) -> dict:
        X_flat = X_train.reshape(len(X_train), -1)
        self.model.fit(X_flat, y_train)
        train_preds = self.model.predict(X_flat)
        train_mse = float(np.mean((train_preds - y_train) ** 2))
        return {"loss": [train_mse]}

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_flat = X.reshape(len(X), -1)
        return self.model.predict(X_flat)

    def save(self, path: str) -> None:
        import joblib
        joblib.dump(self.model, path)

    def load(self, path: str) -> None:
        import joblib
        self.model = joblib.load(path)
