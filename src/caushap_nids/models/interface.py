from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class AnomalyDetector(ABC):
    """Base class for all anomaly detectors in caushap_nids."""

    @abstractmethod
    def fit(self, X_benign: np.ndarray) -> None:
        """Train on benign-only flows. X_benign shape: (N, D)."""
        ...

    @abstractmethod
    def score(self, X: np.ndarray) -> np.ndarray:
        """Return anomaly score per flow. Higher = more anomalous. Shape: (N,)."""
        ...

    @abstractmethod
    def predict(self, X: np.ndarray, threshold: float) -> np.ndarray:
        """Return binary labels: 1 = anomaly, 0 = benign. Shape: (N,)."""
        ...

    def save(self, path: str | Path) -> None:
        raise NotImplementedError(f"{type(self).__name__} does not implement save()")

    def load(self, path: str | Path) -> None:
        raise NotImplementedError(f"{type(self).__name__} does not implement load()")
