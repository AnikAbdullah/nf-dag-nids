from __future__ import annotations

import numpy as np


def _quantus_available() -> bool:
    try:
        import quantus  # noqa: F401
        return True
    except ImportError:
        return False


def eraser_sufficiency(
    detector,
    X: np.ndarray,
    attributions: np.ndarray,
    top_k: int = 5,
    background: np.ndarray | None = None,
) -> np.ndarray:
    """
    Batch ERASER sufficiency. Uses quantus if available, falls back to local impl.
    Returns shape (N,) array of per-flow sufficiency scores. Higher is better.
    """
    if _quantus_available():
        try:
            return _quantus_batch(detector, X, attributions, top_k, mode="sufficiency")
        except Exception:
            pass
    from caushap_nids.evaluation.faithfulness import sufficiency
    return np.array([
        sufficiency(detector, X[i], attributions[i], top_k, background)
        for i in range(len(X))
    ])


def eraser_comprehensiveness(
    detector,
    X: np.ndarray,
    attributions: np.ndarray,
    top_k: int = 5,
    background: np.ndarray | None = None,
) -> np.ndarray:
    """
    Batch ERASER comprehensiveness. Uses quantus if available, falls back to local impl.
    Returns shape (N,) array of per-flow comprehensiveness scores. Higher is better.
    """
    if _quantus_available():
        try:
            return _quantus_batch(detector, X, attributions, top_k, mode="comprehensiveness")
        except Exception:
            pass
    from caushap_nids.evaluation.faithfulness import comprehensiveness
    return np.array([
        comprehensiveness(detector, X[i], attributions[i], top_k, background)
        for i in range(len(X))
    ])


def _quantus_batch(
    detector,
    X: np.ndarray,
    attributions: np.ndarray,
    top_k: int,
    mode: str,
) -> np.ndarray:
    """
    Internal: run quantus Sufficiency or Comprehensiveness metric.
    Wraps detector in a minimal torch.nn.Module so quantus can call forward().
    """
    import torch
    import quantus

    class _Wrapper(torch.nn.Module):
        def __init__(self, det):
            super().__init__()
            self._det = det

        def forward(self, x):
            scores = self._det.score(x.numpy() if isinstance(x, torch.Tensor) else x)
            if isinstance(scores, np.ndarray):
                scores = torch.tensor(scores, dtype=torch.float32)
            return scores.unsqueeze(-1)

    model = _Wrapper(detector)
    kwargs = dict(
        top_k=top_k,
        normalise=True,
        disable_warnings=True,
    )
    if mode == "sufficiency":
        metric = quantus.Sufficiency(**kwargs)
    else:
        metric = quantus.Comprehensiveness(**kwargs)

    return np.array(metric(
        model=model,
        x_batch=X.astype(np.float32),
        y_batch=np.ones(len(X), dtype=int),
        a_batch=attributions.astype(np.float32),
        explain_func=None,
    ), dtype=float)
