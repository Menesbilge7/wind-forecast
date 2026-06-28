import numpy as np
from tensorflow import keras
from .base import BaseModel


class LSTMModel(BaseModel):
    def build(self, input_shape: tuple) -> None:
        units = self.cfg.get("units", [64, 32])
        dropout = self.cfg.get("dropout", 0.2)
        horizon = self.cfg.get("horizon", 1)

        inp = keras.Input(shape=input_shape)
        x = inp
        for i, u in enumerate(units):
            return_seq = i < len(units) - 1
            x = keras.layers.LSTM(u, return_sequences=return_seq)(x)
            x = keras.layers.Dropout(dropout)(x)
        out = keras.layers.Dense(horizon)(x)

        self.model = keras.Model(inp, out)
        self.model.compile(
            optimizer=keras.optimizers.Adam(self.cfg.get("learning_rate", 0.001)),
            loss="mse",
            metrics=["mae"],
        )

    def fit(self, X_train, y_train, X_val, y_val) -> dict:
        cb = [
            keras.callbacks.EarlyStopping(
                patience=self.cfg.get("patience", 10),
                restore_best_weights=True,
            )
        ]
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.cfg.get("epochs", 100),
            batch_size=self.cfg.get("batch_size", 32),
            callbacks=cb,
            verbose=1,
        )
        return history.history

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X, verbose=0)
