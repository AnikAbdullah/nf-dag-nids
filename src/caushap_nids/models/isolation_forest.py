import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import IsolationForest

from .interface import AnomalyDetector


class IFDetector(AnomalyDetector):
    """Benign-only Isolation Forest wrapper.

    Uses score_samples() (raw depth-based score) rather than decision_function()
    to avoid dependence on the contamination offset. Negated so higher = more anomalous.
    """

    def __init__(
        self,
        n_estimators: int = 300,
        max_samples: int | str = 10000,
        contamination: float = 1e-6,
        random_state: int = 42,
        n_jobs: int = -1,
    ) -> None:
        self.n_estimators  = n_estimators
        self.max_samples   = max_samples
        self.contamination = contamination
        self.random_state  = random_state
        self.n_jobs        = n_jobs
        self._model        = IsolationForest(
            n_estimators=n_estimators,
            max_samples=max_samples,
            contamination=contamination,
            random_state=random_state,
            n_jobs=n_jobs,
        )

    def fit(self, X_benign: np.ndarray) -> None:
        self._model.fit(X_benign)

    def score(self, X: np.ndarray) -> np.ndarray:
        """Anomaly score — higher = more anomalous."""
        return -self._model.score_samples(X)

    def predict(self, X: np.ndarray, threshold: float) -> np.ndarray:
        return (self.score(X) >= threshold).astype(int)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self._model, f)

    def load(self, path: str | Path) -> None:
        with Path(path).open("rb") as f:
            self._model = pickle.load(f)
