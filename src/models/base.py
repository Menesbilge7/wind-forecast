from abc import ABC, abstractmethod
import numpy as np


class BaseModel(ABC):
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.model = None

    @abstractmethod
    def build(self, input_shape: tuple) -> None: ...

    @abstractmethod
    def fit(self, X_train, y_train, X_val, y_val) -> dict: ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray: ...

    def predict_with_uncertainty(
        self, X: np.ndarray, n_samples: int = 100
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Monte Carlo Dropout ile belirsizlik tahmini.
        Dropout katmanlarını test sırasında açık bırakarak N kez tahmin yapar.
        Döner: (ortalama tahmin, standart sapma)
        """
        import tensorflow as tf
        preds = np.stack(
            [self.model(X, training=True).numpy() for _ in range(n_samples)],
            axis=0,
        )
        return preds.mean(axis=0), preds.std(axis=0)

    def save(self, path: str) -> None:
        if self.model is not None:
            self.model.save(path)

    def load(self, path: str) -> None:
        from tensorflow import keras
        self.model = keras.models.load_model(path)
