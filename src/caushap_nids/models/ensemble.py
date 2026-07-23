import numpy as np

_EPS = 1e-9


class QuantileNormalizer:
    """Map raw anomaly scores to [0, 1] via 1st–99th percentile clipping."""

    def __init__(self, low: float = 1.0, high: float = 99.0) -> None:
        self.low  = low
        self.high = high
        self.lo_: float | None = None
        self.hi_: float | None = None

    def fit(self, X: np.ndarray) -> "QuantileNormalizer":
        x = np.asarray(X, dtype=np.float64).ravel()
        self.lo_ = float(np.percentile(x, self.low))
        self.hi_ = float(np.percentile(x, self.high))
        if self.hi_ <= self.lo_:
            self.hi_ = self.lo_ + 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        x = np.asarray(X, dtype=np.float64).ravel()
        return np.clip((x - self.lo_) / (self.hi_ - self.lo_), 0.0, 1.0)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


def geometric_ensemble(
    ae_scores_norm: np.ndarray,
    if_scores_norm: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """Weighted geometric mean: ae_norm^alpha * if_norm^(1-alpha).

    Both inputs must already be normalised to [0, 1] by QuantileNormalizer.
    More conservative than arithmetic mean: near-zero on either detector
    suppresses the combined score, reducing single-model noise FPs.
    """
    ae = np.clip(ae_scores_norm, _EPS, 1.0)
    if_ = np.clip(if_scores_norm, _EPS, 1.0)
    return (ae ** alpha) * (if_ ** (1.0 - alpha))
