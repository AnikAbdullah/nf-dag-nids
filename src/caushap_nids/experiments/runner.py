"""
Single-run and matrix-run orchestration for ablation experiments.

A "run" = (config_name, dataset_key, seed) → RunResult saved to
artifacts/results/<config>/<dataset>/<seed>/result.json

The runner is deliberately thin: it wires together existing modules
(data_pipeline, models, evaluation) without re-implementing them.
"""
from __future__ import annotations

import json
import time
import subprocess
import csv
from hashlib import sha1
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from caushap_nids.experiments.seed_management import set_all_seeds


DATASETS = ("nf_cic2018", "nf_unsw15", "edge_iiotset", "5g_nidd")
CONFIG_NAMES = ("A0_baseline", "A1_stl_only", "A2_causal_shap", "A3_moocf", "A4_full", "A5_random_dag")
DEFAULT_SEEDS = (42, 43, 44)
CRITERIA_VERSION = "strict-ablation-v7"
MAX_FAITHFULNESS_FLOWS = 20
MAX_CF_FLOWS = 10
_OPERATING_POINT_CACHE: dict[tuple[Any, ...], tuple[Any, Any, float, dict[str, Any]]] = {}
_RUNTIME_TUNED = False


def _tune_runtime_once(verbose: bool = False) -> dict[str, Any]:
    """One-shot environment tuning for the host running the ablation matrix.

    Idempotent: the global guard means repeat calls are free.
      • CUDA  — enables TF32 + cuDNN benchmark autotune (RTX 30/40/50 series).
      • CPU   — leaves sklearn/numpy at their default thread counts so the
                IsolationForest's `n_jobs=-1` keeps every i7/i9 core busy.
                We just pin torch's intra-op pool to the physical core count
                to avoid hyperthread contention with sklearn's joblib workers.
    """
    global _RUNTIME_TUNED
    info: dict[str, Any] = {}
    if _RUNTIME_TUNED:
        return info
    try:
        import os
        import torch

        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass
            info["cuda_device"] = torch.cuda.get_device_name(0)
            info["cuda_capability"] = ".".join(str(c) for c in torch.cuda.get_device_capability(0))
            try:
                info["cuda_total_gb"] = round(
                    torch.cuda.get_device_properties(0).total_memory / 1024**3, 1
                )
            except Exception:
                pass
        elif torch.backends.mps.is_available() and not os.environ.get("CAUSHAP_DEVICE"):
            # Apple Silicon: the AE is a tiny 39-dim net and the XAI layers issue
            # hundreds of thousands of single-flow inferences, where MPS kernel-launch
            # latency dominates — measured 5-12x slower than CPU on an M4. Default to
            # CPU unless the user explicitly sets CAUSHAP_DEVICE=mps. This also lets
            # several dataset streams run in parallel without GPU contention.
            os.environ["CAUSHAP_DEVICE"] = "cpu"
            info["apple_silicon"] = "auto-selected CPU (set CAUSHAP_DEVICE=mps to override)"

        # Thread pool: pin torch's intra-op pool to the physical core count to avoid
        # hyperthread contention with sklearn's joblib workers. When running several
        # matrix streams in parallel, set CAUSHAP_TORCH_THREADS low (e.g. 2) so the
        # processes don't oversubscribe the cores.
        try:
            env_threads = os.environ.get("CAUSHAP_TORCH_THREADS")
            n_threads = int(env_threads) if env_threads else max(1, (os.cpu_count() or 2) // 2)
            torch.set_num_threads(n_threads)
            info["torch_threads"] = n_threads
        except Exception:
            pass
        info["device"] = os.environ.get("CAUSHAP_DEVICE") or (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
    except Exception:
        pass
    _RUNTIME_TUNED = True
    if verbose and info:
        print(f"[runtime] tuned → {info}")
    return info


@dataclass
class RunConfig:
    """Parsed ablation configuration (mirrors YAML structure)."""
    name: str
    description: str
    detector: dict[str, Any]
    shapley: dict[str, Any]
    counterfactual: dict[str, Any]
    attack_typing: dict[str, Any]
    dag: dict[str, Any] | None
    concepts: dict[str, Any] | bool


@dataclass
class RunResult:
    config_name: str
    dataset: str
    seed: int
    elapsed_seconds: float
    detection: dict[str, Any]
    faithfulness: dict[str, Any] = field(default_factory=dict)
    cf_metrics: dict[str, Any] = field(default_factory=dict)
    concept_metrics: dict[str, Any] = field(default_factory=dict)
    status: str = "ok"
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))


@dataclass
class _EnsembleScoreAdapter:
    """Expose the trained AE+IF ensemble through the detector.score(X) interface."""
    ae: Any
    if_det: Any
    ae_norm: Any
    if_norm: Any
    alpha: float
    ae_feature_indices: tuple[int, ...] | None = None

    def score(self, X: np.ndarray) -> np.ndarray:
        from caushap_nids.models import geometric_ensemble

        x_arr = np.asarray(X, dtype=np.float64)
        if self.ae_feature_indices is None:
            ae_scores = self.ae.score(x_arr)
        else:
            ae_scores = self.ae.score(x_arr, feature_indices=self.ae_feature_indices)
        return geometric_ensemble(
            self.ae_norm.transform(ae_scores),
            self.if_norm.transform(self.if_det.score(x_arr)),
            alpha=self.alpha,
        )

    def concept_score(self, X: np.ndarray) -> np.ndarray:
        """Holistic AE reconstruction score used by Layer-C concept fidelity.

        The detector's operating score may intentionally use a validation-tuned
        residual subset for strict low-FPR detection.  Concept ablation fidelity
        is defined over the full AE reconstruction signal, matching the Module-5c
        assumption that concepts should not be judged only against a narrow
        threshold-calibration subset.
        """
        x_arr = np.asarray(X, dtype=np.float64)
        return self.ae.score(x_arr)


def load_run_config(config_name: str, configs_dir: Path) -> RunConfig:
    """Load a YAML ablation config and return a RunConfig."""
    try:
        import yaml
    except ImportError as e:
        raise ImportError("PyYAML is required: pip install pyyaml") from e

    path = configs_dir / f"{config_name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path) as f:
        raw = yaml.safe_load(f)

    return RunConfig(
        name=raw["name"],
        description=raw.get("description", ""),
        detector=raw.get("detector", {}),
        shapley=raw.get("shapley", {}),
        counterfactual=raw.get("counterfactual", {}),
        attack_typing=raw.get("attack_typing", {}),
        dag=raw.get("dag"),
        concepts=raw.get("concepts", False),
    )


def _stable_json_hash(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return sha1(raw.encode("utf-8")).hexdigest()[:12]


def _detector_model_fingerprint(detector_cfg: dict[str, Any]) -> str:
    # These keys affect validation-time operating point calibration, not the
    # saved AE or IF model parameters.
    operating_keys = {
        "target_fpr",
        "ae_alpha",
        "ae_alpha_grid",
        "ae_feature_indices",
        "ae_feature_selection",
        "ae_feature_selection_max_features",
        "ae_feature_selection_top_candidates",
    }
    model_cfg = {
        k: v for k, v in detector_cfg.items()
        if k not in operating_keys
    }
    return _stable_json_hash(model_cfg)


def _result_path(artifact_dir: Path, config_name: str, dataset: str, seed: int) -> Path:
    return artifact_dir / "results" / config_name / dataset / str(seed) / "result.json"


def _git_commit_hash(project_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).strip()
    except Exception:
        return "unavailable"


def _load_result_file(path: Path) -> RunResult | None:
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text())
        return RunResult(**raw)
    except Exception:
        return None


def _reusable_result(path: Path, *, n_explain: int, config_hash: str) -> RunResult | None:
    result = _load_result_file(path)
    if result is None or result.status != "ok":
        return None
    meta = result.metadata or {}
    if meta.get("criteria_version") != CRITERIA_VERSION:
        return None
    if meta.get("n_explain") != n_explain:
        return None
    if meta.get("config_hash") != config_hash:
        return None
    return result


def _binary_fpr(y_true: np.ndarray, y_pred: np.ndarray, benign_label: int = 0) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = y_true == benign_label
    if mask.sum() == 0:
        return float("nan")
    return float((y_pred[mask] != benign_label).mean())


def _binary_macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    from sklearn.metrics import f1_score

    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def _operating_point(scores: np.ndarray, labels: np.ndarray, target_fpr: float) -> dict[str, float]:
    """Best threshold and validation metrics under a benign-FPR cap."""
    scores = np.asarray(scores, dtype=np.float64).ravel()
    labels = (np.asarray(labels).ravel() != 0).astype(np.int64)
    if len(scores) == 0:
        return {
            "threshold": 0.0,
            "val_fpr": float("nan"),
            "val_macro_f1": float("nan"),
            "val_attack_recall": float("nan"),
            "val_attack_precision": float("nan"),
        }

    order = np.argsort(-scores, kind="mergesort")
    s_sorted = scores[order]
    y_sorted = labels[order]
    n_pos = int(y_sorted.sum())
    n_neg = int(len(y_sorted) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return {
            "threshold": float(s_sorted[0]),
            "val_fpr": float("nan"),
            "val_macro_f1": float("nan"),
            "val_attack_recall": float("nan"),
            "val_attack_precision": float("nan"),
        }

    eps = 1e-12
    block_ends = np.flatnonzero(np.r_[s_sorted[1:] != s_sorted[:-1], True])
    thresholds = s_sorted[block_ends]
    tp = np.cumsum(y_sorted == 1).astype(np.float64)[block_ends]
    fp = np.cumsum(y_sorted == 0).astype(np.float64)[block_ends]
    fn = n_pos - tp
    tn = n_neg - fp

    val_fpr = fp / n_neg
    attack_recall = tp / n_pos
    attack_precision = tp / np.maximum(tp + fp, eps)
    attack_f1 = (
        2.0 * attack_precision * attack_recall
        / np.maximum(attack_precision + attack_recall, eps)
    )
    benign_precision = tn / np.maximum(tn + fn, eps)
    benign_recall = tn / n_neg
    benign_f1 = (
        2.0 * benign_precision * benign_recall
        / np.maximum(benign_precision + benign_recall, eps)
    )
    macro_f1 = 0.5 * (attack_f1 + benign_f1)

    threshold_none = np.nextafter(float(s_sorted[0]), np.inf)
    benign_precision0 = n_neg / max(n_pos + n_neg, 1)
    benign_f10 = 2.0 * benign_precision0 / max(benign_precision0 + 1.0, eps)
    thresholds = np.r_[threshold_none, thresholds]
    val_fpr = np.r_[0.0, val_fpr]
    attack_recall = np.r_[0.0, attack_recall]
    attack_precision = np.r_[0.0, attack_precision]
    macro_f1 = np.r_[0.5 * benign_f10, macro_f1]

    mask = val_fpr <= float(target_fpr)
    pool = np.where(mask)[0] if mask.any() else np.arange(len(macro_f1))
    composite = macro_f1[pool] + 1e-6 * attack_recall[pool] + 1e-9 * attack_precision[pool]
    best = pool[int(np.argmax(composite))]
    return {
        "threshold": float(thresholds[best]),
        "val_fpr": float(val_fpr[best]),
        "val_macro_f1": float(macro_f1[best]),
        "val_attack_recall": float(attack_recall[best]),
        "val_attack_precision": float(attack_precision[best]),
    }


def _better_operating_point(candidate: dict[str, float], incumbent: dict[str, float]) -> bool:
    c = (
        candidate.get("val_macro_f1", float("nan")),
        candidate.get("val_attack_recall", float("nan")),
        -candidate.get("val_fpr", float("inf")),
    )
    i = (
        incumbent.get("val_macro_f1", float("nan")),
        incumbent.get("val_attack_recall", float("nan")),
        -incumbent.get("val_fpr", float("inf")),
    )
    return np.nan_to_num(c, nan=-np.inf).tolist() > np.nan_to_num(i, nan=-np.inf).tolist()


def _normalise_feature_indices(raw: Any, n_features: int) -> tuple[int, ...]:
    idx = tuple(int(i) for i in np.asarray(raw, dtype=np.int64).ravel())
    if not idx:
        raise ValueError("ae_feature_indices must contain at least one feature")
    if min(idx) < 0 or max(idx) >= n_features:
        raise ValueError(f"ae_feature_indices must be in [0, {n_features})")
    return idx


def _ae_feature_scores(
    ae: Any,
    X_val: np.ndarray,
    y_val: np.ndarray,
    target_fpr: float,
    detector_cfg: dict[str, Any],
    feature_names: list[str],
) -> tuple[tuple[int, ...] | None, np.ndarray, dict[str, Any]]:
    """Select an AE residual subset on validation data and return val scores."""
    mode = detector_cfg.get("ae_feature_selection", "greedy_val")
    mode_text = str(mode).lower()
    explicit = detector_cfg.get("ae_feature_indices")
    disabled = (mode is False) or mode_text in {"0", "false", "off", "none", "disabled"}

    if explicit is not None:
        indices = _normalise_feature_indices(explicit, X_val.shape[1])
        scores = ae.score(X_val, feature_indices=indices)
        op = _operating_point(scores, y_val, target_fpr)
        return indices, np.asarray(scores, dtype=np.float64), {
            "ae_score_mode": "configured_feature_mean",
            "ae_feature_indices": list(indices),
            "ae_feature_names": [feature_names[i] for i in indices],
            "ae_feature_selection_val_macro_f1": op["val_macro_f1"],
            "ae_feature_selection_val_fpr": op["val_fpr"],
        }

    if disabled:
        scores = ae.score(X_val)
        op = _operating_point(scores, y_val, target_fpr)
        return None, np.asarray(scores, dtype=np.float64), {
            "ae_score_mode": "mean_all_features",
            "ae_feature_indices": [],
            "ae_feature_names": "all",
            "ae_feature_selection_val_macro_f1": op["val_macro_f1"],
            "ae_feature_selection_val_fpr": op["val_fpr"],
        }

    err = ae.per_feature_error(X_val).astype(np.float32, copy=False)
    full_scores = np.asarray(err.mean(axis=1), dtype=np.float64)
    full_op = _operating_point(full_scores, y_val, target_fpr)
    best_indices: tuple[int, ...] | None = None
    best_scores = np.array(full_scores, dtype=np.float64, copy=True)
    best_op = full_op
    best_mode = "mean_all_features"

    single_rows: list[tuple[float, int, dict[str, float]]] = []
    for j in range(err.shape[1]):
        op = _operating_point(err[:, j], y_val, target_fpr)
        single_rows.append((op["val_macro_f1"], j, op))
        if _better_operating_point(op, best_op):
            best_indices = (j,)
            best_scores = np.array(err[:, j], dtype=np.float64, copy=True)
            best_op = op
            best_mode = "single_feature_mean"

    single_rows.sort(key=lambda row: (row[0], row[2]["val_attack_recall"]), reverse=True)
    top_candidates = max(1, int(detector_cfg.get("ae_feature_selection_top_candidates", 15)))
    max_features = max(1, int(detector_cfg.get("ae_feature_selection_max_features", 5)))
    candidate_features = [j for _, j, _ in single_rows[:min(top_candidates, len(single_rows))]]

    chosen: list[int] = []
    last_step_op: dict[str, float] | None = None
    for _ in range(min(max_features, len(candidate_features))):
        step_best: tuple[tuple[float, float, float], tuple[int, ...], dict[str, float]] | None = None
        for j in candidate_features:
            if j in chosen:
                continue
            idx = tuple(chosen + [j])
            scores = err[:, idx[0]] if len(idx) == 1 else err[:, idx].mean(axis=1)
            op = _operating_point(scores, y_val, target_fpr)
            key = (op["val_macro_f1"], op["val_attack_recall"], -op["val_fpr"])
            if step_best is None or key > step_best[0]:
                step_best = (key, idx, op)
        if step_best is None:
            break
        if last_step_op is not None and not _better_operating_point(step_best[2], last_step_op):
            break
        chosen = list(step_best[1])
        last_step_op = step_best[2]
        if _better_operating_point(step_best[2], best_op):
            best_indices = step_best[1]
            scores = err[:, best_indices[0]] if len(best_indices) == 1 else err[:, best_indices].mean(axis=1)
            best_scores = np.array(scores, dtype=np.float64, copy=True)
            best_op = step_best[2]
            best_mode = "greedy_feature_mean"

    top_single = [
        {
            "feature": feature_names[j],
            "index": int(j),
            "val_macro_f1": round(float(op["val_macro_f1"]), 6),
            "val_fpr": round(float(op["val_fpr"]), 6),
        }
        for _, j, op in single_rows[:5]
    ]
    del err
    return best_indices, best_scores, {
        "ae_score_mode": best_mode,
        "ae_feature_indices": list(best_indices or []),
        "ae_feature_names": [feature_names[i] for i in best_indices] if best_indices else "all",
        "ae_feature_selection_val_macro_f1": best_op["val_macro_f1"],
        "ae_feature_selection_val_fpr": best_op["val_fpr"],
        "ae_feature_selection_top_single": top_single,
        "ae_feature_selection_max_features": max_features,
        "ae_feature_selection_top_candidates": top_candidates,
    }


def _calibrate_detector_operating_point(
    ae: Any,
    if_det: Any,
    X_val: np.ndarray,
    y_val: np.ndarray,
    detector_cfg: dict[str, Any],
    feature_names: list[str],
) -> tuple[tuple[Any, Any, float, tuple[int, ...] | None], float, dict[str, Any]]:
    from caushap_nids.models import QuantileNormalizer, geometric_ensemble

    target_fpr = float(detector_cfg.get("target_fpr", 0.01))
    ae_indices, ae_val_raw, feature_diag = _ae_feature_scores(
        ae, X_val, y_val, target_fpr, detector_cfg, feature_names
    )
    if_val_raw = if_det.score(X_val)
    ae_norm = QuantileNormalizer(1.0, 99.0).fit(ae_val_raw)
    if_norm = QuantileNormalizer(1.0, 99.0).fit(if_val_raw)
    ae_val = ae_norm.transform(ae_val_raw)
    if_val = if_norm.transform(if_val_raw)

    alpha_cfg = detector_cfg.get("ae_alpha", "auto")
    alpha_auto = alpha_cfg is None or str(alpha_cfg).lower() == "auto"
    if alpha_auto:
        grid = detector_cfg.get("ae_alpha_grid")
        if grid is None:
            grid = [round(float(x), 2) for x in np.linspace(0.0, 1.0, 21)]
    else:
        grid = [float(alpha_cfg)]
    alphas = sorted({float(a) for a in grid if 0.0 <= float(a) <= 1.0})
    if not alphas:
        raise ValueError("ae_alpha_grid must contain at least one value in [0, 1]")

    best_alpha = alphas[0]
    best_scores = geometric_ensemble(ae_val, if_val, alpha=best_alpha)
    best_op = _operating_point(best_scores, y_val, target_fpr)
    for alpha in alphas[1:]:
        scores = geometric_ensemble(ae_val, if_val, alpha=alpha)
        op = _operating_point(scores, y_val, target_fpr)
        if _better_operating_point(op, best_op):
            best_alpha = alpha
            best_scores = scores
            best_op = op

    del best_scores
    diagnostics = {
        "threshold": best_op["threshold"],
        "target_fpr": target_fpr,
        "val_fpr": best_op["val_fpr"],
        "val_macro_f1": best_op["val_macro_f1"],
        "val_attack_recall": best_op["val_attack_recall"],
        "val_attack_precision": best_op["val_attack_precision"],
        "ae_alpha": float(best_alpha),
        "ae_alpha_source": "tuned" if alpha_auto else "config",
        **feature_diag,
    }
    return (ae_norm, if_norm, float(best_alpha), ae_indices), float(best_op["threshold"]), diagnostics


def _train_or_load_detector(
    cfg: RunConfig,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    artifact_dir: Path,
    config_name: str,
    dataset: str,
    seed: int,
) -> tuple[Any, Any, float, dict[str, Any]]:
    """Train AE+IF ensemble or load from checkpoint.

    Returns (detectors, normalizers, threshold, diagnostics) where
    detectors = (ae, if_det) and normalizers =
    (ae_norm, if_norm, alpha, ae_feature_indices).
    """
    from caushap_nids.data_pipeline.loaders import NF_V2_FEATURE_COLS
    from caushap_nids.models import DeepAutoEncoder, IFDetector

    d = cfg.detector
    fingerprint = _detector_model_fingerprint(d)
    ckpt_dir = artifact_dir / "models" / "_shared" / f"detector-{fingerprint}" / dataset / str(seed)
    ae_path = ckpt_dir / "ae.pt"
    if_path = ckpt_dir / "if.pkl"

    def _new_pair() -> tuple[Any, Any]:
        ae = DeepAutoEncoder(
            in_dim=X_train.shape[1],
            hidden_dims=d.get("ae_hidden_dims", [64, 32, 16]),
            dropout=d.get("ae_dropout", 0.1),
            lr=d.get("ae_lr", 3e-4),
            epochs=d.get("ae_epochs", 100),
            batch_size=d.get("ae_batch_size", 2048),
            seed=seed,
        )
        if_det = IFDetector(
            n_estimators=d.get("if_n_estimators", 300),
            max_samples=d.get("if_max_samples", 10000),
            random_state=seed,
        )
        return ae, if_det

    def _load_pair(ae_ckpt: Path, if_ckpt: Path) -> tuple[Any, Any]:
        ae, if_det = _new_pair()
        ae.load(ae_ckpt)
        if_det.load(if_ckpt)
        return ae, if_det

    def _stat_key(path: Path) -> tuple[str, int, int]:
        st = path.stat()
        return (str(path.resolve()), int(st.st_size), int(st.st_mtime_ns))

    op_cfg = {
        k: d.get(k)
        for k in (
            "target_fpr",
            "ae_alpha",
            "ae_alpha_grid",
            "ae_feature_indices",
            "ae_feature_selection",
            "ae_feature_selection_max_features",
            "ae_feature_selection_top_candidates",
        )
        if k in d
    }

    def _calibrate_cached(
        source: str,
        ae_ckpt: Path,
        if_ckpt: Path,
        ae: Any,
        if_det: Any,
    ) -> tuple[tuple[Any, Any, float, tuple[int, ...] | None], float, dict[str, Any]]:
        cache_key = (
            CRITERIA_VERSION,
            fingerprint,
            dataset,
            int(seed),
            _stat_key(ae_ckpt),
            _stat_key(if_ckpt),
            _stable_json_hash(op_cfg),
            tuple(X_val.shape),
            int(len(y_val)),
        )
        cached = _OPERATING_POINT_CACHE.get(cache_key)
        if cached is None:
            normalizers, threshold, diagnostics = _calibrate_detector_operating_point(
                ae, if_det, X_val, y_val, d, NF_V2_FEATURE_COLS
            )
            _OPERATING_POINT_CACHE[cache_key] = (normalizers, threshold, dict(diagnostics))
        else:
            normalizers, threshold, diagnostics = cached
            diagnostics = dict(diagnostics)
        diagnostics["detector_checkpoint_source"] = source
        return normalizers, threshold, diagnostics

    candidates: list[tuple[str, Path, Path]] = []
    # AE/IF checkpoint resolution (v2 plan §5 canonical path: artifacts/models/...).
    # Priority order:
    #   1. root  — artifacts/{ae.pt, if.pkl}: the Module-2 notebook-01 checkpoint
    #              for NF-CIC2018 seed 42, kept at the top of artifacts/ for
    #              notebook-04/05/06 backward-compat. Used only for that
    #              dataset+seed cell and still has to win on validation.
    #   2. shared — artifacts/models/_shared/detector-<fingerprint>/<ds>/<seed>/
    #              ae.pt: the canonical training-output path for experiments.
    #   3. legacy — artifacts/models/<config>/<ds>/<seed>/ae.pt: pre-shared-cache
    #              layout, kept readable for old smoke runs.
    root_ae = artifact_dir / "ae.pt"
    root_if = artifact_dir / "if.pkl"
    if dataset == "nf_cic2018" and seed == 42 and root_ae.exists() and root_if.exists():
        candidates.append(("root_module2", root_ae, root_if))
    if ae_path.exists() and if_path.exists():
        candidates.append(("shared", ae_path, if_path))

    legacy_dirs = [
        artifact_dir / "models" / cfg_name / dataset / str(seed)
        for cfg_name in (config_name, *CONFIG_NAMES)
    ]
    for legacy in legacy_dirs:
        legacy_ae = legacy / "ae.pt"
        legacy_if = legacy / "if.pkl"
        if legacy_ae.exists() and legacy_if.exists():
            candidates.append((f"legacy:{legacy.parent.name}", legacy_ae, legacy_if))

    unique_candidates: list[tuple[str, Path, Path]] = []
    seen_paths: set[tuple[str, str]] = set()
    for source, cand_ae, cand_if in candidates:
        key = (str(cand_ae.resolve()), str(cand_if.resolve()))
        if key not in seen_paths:
            unique_candidates.append((source, cand_ae, cand_if))
            seen_paths.add(key)

    selected: tuple[Any, Any, tuple[Any, Any, float, tuple[int, ...] | None], float, dict[str, Any]] | None = None
    candidate_summaries: list[dict[str, Any]] = []
    for source, cand_ae, cand_if in unique_candidates:
        try:
            cand_ae_obj, cand_if_obj = _load_pair(cand_ae, cand_if)
            normalizers, threshold, diag = _calibrate_cached(
                source, cand_ae, cand_if, cand_ae_obj, cand_if_obj
            )
        except Exception as exc:
            candidate_summaries.append({"source": source, "status": "skipped", "error": str(exc)})
            continue

        candidate_summaries.append({
            "source": source,
            "status": "ok",
            "val_macro_f1": round(float(diag["val_macro_f1"]), 6),
            "val_fpr": round(float(diag["val_fpr"]), 6),
            "ae_alpha": round(float(diag["ae_alpha"]), 4),
            "ae_score_mode": diag.get("ae_score_mode"),
            "ae_feature_names": diag.get("ae_feature_names"),
        })
        if selected is None or _better_operating_point(diag, selected[4]):
            selected = (cand_ae_obj, cand_if_obj, normalizers, threshold, diag)

    if selected is None:
        ae, if_det = _new_pair()
        X_train_benign = X_train[y_train == 0]
        ae.fit(X_train_benign)
        if_det.fit(X_train_benign)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ae.save(ae_path)
        if_det.save(if_path)
        normalizers, threshold, diagnostics = _calibrate_cached(
            "trained", ae_path, if_path, ae, if_det
        )
    else:
        ae, if_det, normalizers, threshold, diagnostics = selected
        if diagnostics.get("detector_checkpoint_source") != "shared":
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            ae.save(ae_path)
            if_det.save(if_path)

    diagnostics.update({
        "detector_fingerprint": fingerprint,
        "detector_checkpoint": str(ckpt_dir),
        "detector_checkpoint_candidates": candidate_summaries,
    })

    return (ae, if_det), normalizers, threshold, diagnostics


_PreloadedData = tuple  # (X_train, X_val, X_test, y_train, y_val, y_test)


def _artifact_relative_path(artifact_dir: Path, raw_path: str | Path | None, default: str) -> Path:
    """Resolve config paths that may already include the artifacts directory name."""
    path = Path(raw_path or default)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] in {artifact_dir.name, "artifacts"}:
        path = Path(*path.parts[1:])
    return artifact_dir / path


def _load_module5c_gate_metrics(artifact_dir: Path) -> dict[str, Any] | None:
    """Load validated Layer-C gate metrics produced by notebook 05/Module 5c.

    The ablation runner's `n_explain` can be a tiny smoke-test sample, so concept
    metrics should use the labelled concept-gate evaluation when it is available
    instead of one arbitrary predicted anomaly.  That gate is the coding-plan
    acceptance test for concept fidelity, rank stability, expected concept
    firing, and benign concept FPR.
    """
    path = artifact_dir / "module5c_gate_summary.csv"
    if not path.exists():
        return None

    rows: dict[str, dict[str, str]] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            criterion = (row.get("criterion") or "").strip()
            if criterion:
                rows[criterion] = row

    fidelity = rows.get("concept_fidelity_ge_80pct_across_targets")
    if fidelity is None:
        return None

    try:
        mean_fidelity = float(fidelity["value"])
    except (KeyError, TypeError, ValueError):
        return None

    def _value(name: str) -> str | None:
        row = rows.get(name)
        return None if row is None else row.get("value")

    def _float_value(name: str) -> float | None:
        raw = _value(name)
        try:
            return None if raw is None else float(raw)
        except ValueError:
            return None

    statuses = {
        key: (row.get("status") or "").strip().upper()
        for key, row in rows.items()
    }
    passed = all(status == "PASS" for status in statuses.values()) if statuses else False

    return {
        "mean_fidelity": mean_fidelity,
        "rank_stability": _float_value("rank_stability_ge_80pct_across_targets"),
        "expected_concepts": _value("expected_concepts_fire_on_labelled_targets"),
        "benign_fpr": _float_value("fpr_lt_5pct_benign"),
        "gate_passed": passed,
        "source": str(path),
        "method": "labelled_module5c_gate",
        "n_explained": len(rows),
        "statuses": statuses,
    }


def _random_rewired_dag(source: Any, seed: int) -> Any:
    """Return an acyclic random DAG with the same nodes and edge count as source."""
    import networkx as nx

    nodes = list(source.nodes())
    edge_count = source.number_of_edges()
    rng = np.random.default_rng(seed)
    order = list(rng.permutation(nodes))
    candidates = [(order[i], order[j]) for i in range(len(order)) for j in range(i + 1, len(order))]
    rng.shuffle(candidates)

    dag = nx.DiGraph()
    dag.add_nodes_from(nodes)
    dag.add_edges_from(candidates[:min(edge_count, len(candidates))])
    return dag


def _load_configured_dag(cfg: RunConfig, artifact_dir: Path, seed: int) -> Any | None:
    """Load the configured DAG, including the random-DAG ablation variant."""
    if cfg.dag is None:
        return None

    import networkx as nx

    dag_cfg = cfg.dag
    if dag_cfg.get("random", False):
        source_path = _artifact_relative_path(
            artifact_dir,
            dag_cfg.get("source_path") or dag_cfg.get("path"),
            "nf_dag_v1.graphml",
        )
        if not source_path.exists():
            return None
        source = nx.read_graphml(str(source_path))
        random_seed = int(dag_cfg.get("random_seed", seed))
        return _random_rewired_dag(source, random_seed)

    dag_path = _artifact_relative_path(artifact_dir, dag_cfg.get("path"), "nf_dag_v1.graphml")
    if not dag_path.exists():
        return None
    return nx.read_graphml(str(dag_path))


def _load_expert_dag(artifact_dir: Path) -> Any | None:
    import networkx as nx

    dag_path = artifact_dir / "nf_dag_v1.graphml"
    if not dag_path.exists():
        return None
    return nx.read_graphml(str(dag_path))


def _empty_feature_dag(feature_names: list[str]) -> Any:
    import networkx as nx

    dag = nx.DiGraph()
    dag.add_nodes_from(feature_names)
    return dag


def _evaluate_cfs_against_dag(cfs: list[Any], dag: Any, feature_names: list[str]) -> None:
    """Rescore each counterfactual's feasibility against ``dag``, in place.

    Used for the random-DAG control, whose counterfactuals are searched under a
    random structure but scored against the expert graph so that both arms share
    one yardstick. Note what that comparison does and does not show: the control
    is graded on constraints it was not given, so the resulting gap reflects the
    mismatch between the search objective and the scoring graph as well as any
    difference in the quality of the graphs. ``_mean_feasibility_against`` keeps
    the own-graph number so both readings can be reported.
    """
    from caushap_nids.xai_layers.multi_obj_cf.objectives import feasibility as _feasibility

    for cf in cfs:
        cf.feasibility_rate = 1.0 - _feasibility(cf.x_orig, cf.x_cf, dag, feature_names)


def _mean_feasibility_against(cfs: list[Any], dag: Any, feature_names: list[str]) -> float:
    """Mean feasibility of ``cfs`` against ``dag``, without mutating them."""
    from caushap_nids.xai_layers.multi_obj_cf.objectives import feasibility as _feasibility

    if not cfs:
        return float("nan")
    return float(
        np.mean([1.0 - _feasibility(cf.x_orig, cf.x_cf, dag, feature_names) for cf in cfs])
    )


def _select_explanation_indices(
    y_pred: np.ndarray,
    rng: np.random.Generator,
    n_explain: int,
) -> np.ndarray:
    """Select predicted anomalies for XAI evaluation; n_explain=0 means all."""
    if n_explain < 0:
        raise ValueError("n_explain must be non-negative; use 0 for all predicted anomalies")

    anomaly_idx = np.where(np.asarray(y_pred) == 1)[0]
    if n_explain == 0 or len(anomaly_idx) <= n_explain:
        return anomaly_idx
    return rng.choice(anomaly_idx, size=n_explain, replace=False)


def _preprocess(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply the five-step preprocessing pipeline from notebook 01.

    Steps (all fit parameters derived from benign training only):
      1. clip >= 0
      2. Benign-only [p1, p99] per-feature clip
      3. log1p
      4. RobustScaler (applied manually in float32 — avoids sklearn's float64 cast
         which would otherwise spike peak RAM by ~4 GB on nf_cic2018)
      5. Final safety clip to [-10, 10]
    """
    X_train = np.clip(X_train, 0.0, None)
    X_val   = np.clip(X_val,   0.0, None)
    X_test  = np.clip(X_test,  0.0, None)

    X_benign = X_train[y_train == 0]
    pct_low, pct_high = np.percentile(X_benign, [1, 99], axis=0)
    pct_low  = np.nan_to_num(pct_low,  nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    pct_high = np.nan_to_num(pct_high, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    pct_high = np.maximum(pct_high, pct_low + 1e-6)

    X_train = np.log1p(np.clip(X_train, pct_low, pct_high))
    X_val   = np.log1p(np.clip(X_val,   pct_low, pct_high))
    X_test  = np.log1p(np.clip(X_test,  pct_low, pct_high))

    # Fit RobustScaler on benign, then apply manually to stay in float32.
    # sklearn's transform() upcasts to float64 internally; doing it by hand avoids
    # a ~4 GB temporary allocation on the 12M-row training split.
    from sklearn.preprocessing import RobustScaler
    _scaler = RobustScaler()
    _scaler.fit(np.log1p(np.clip(X_benign, pct_low, pct_high)))
    center = _scaler.center_.astype(np.float32)
    scale  = _scaler.scale_.astype(np.float32)

    X_train = np.clip((X_train - center) / scale, -10.0, 10.0)
    X_val   = np.clip((X_val   - center) / scale, -10.0, 10.0)
    X_test  = np.clip((X_test  - center) / scale, -10.0, 10.0)

    return X_train, X_val, X_test


def _load_dataset(
    dataset: str,
    seed: int,
    data_dir: Path,
) -> _PreloadedData:
    """Load, preprocess, and return one dataset as float32 numpy arrays."""
    from caushap_nids.data_pipeline.loaders import load_nf_v2, NF_V2_FEATURE_COLS
    import polars as pl

    feat_cols = NF_V2_FEATURE_COLS
    train_df = load_nf_v2(dataset, "train", seed=seed, data_dir=str(data_dir))
    val_df   = load_nf_v2(dataset, "val",   seed=seed, data_dir=str(data_dir))
    test_df  = load_nf_v2(dataset, "test",  seed=seed, data_dir=str(data_dir))

    def _raw(df: "pl.DataFrame") -> np.ndarray:
        return df.select([
            pl.col(c).cast(pl.Float32, strict=False).fill_null(0.0)
            for c in feat_cols
        ]).to_numpy()

    y_train = train_df["label"].to_numpy().astype(int)
    y_val   = val_df["label"].to_numpy().astype(int)
    y_test  = test_df["label"].to_numpy().astype(int)

    X_train, X_val, X_test = _preprocess(_raw(train_df), _raw(val_df), _raw(test_df), y_train)

    return X_train, X_val, X_test, y_train, y_val, y_test


def run_single(
    config_name: str,
    dataset: str,
    seed: int,
    *,
    data_dir: str | Path = "data",
    artifact_dir: str | Path = "artifacts",
    configs_dir: str | Path = "configs",
    n_explain: int = 200,
    verbose: bool = True,
    _preloaded: _PreloadedData | None = None,
) -> RunResult:
    """
    Run one ablation cell: (config_name, dataset, seed) → RunResult.

    Parameters
    ----------
    n_explain : int
        Number of anomalous test flows to run XAI layers on. Use 0 for all
        predicted anomalies; n_explain=200 gives fast smoke-test results.
    _preloaded : tuple, optional
        Pre-loaded (X_train, X_val, X_test, y_train, y_val, y_test) to avoid
        redundant parquet reads when looping over configs for the same dataset.
    """
    t0 = time.perf_counter()
    data_dir = Path(data_dir)
    artifact_dir = Path(artifact_dir)
    configs_dir = Path(configs_dir)

    _tune_runtime_once(verbose=verbose)
    rng = set_all_seeds(seed)
    if n_explain < 0:
        return RunResult(
            config_name,
            dataset,
            seed,
            0.0,
            {},
            status="error",
            error="n_explain must be non-negative; use 0 for all predicted anomalies",
        )

    try:
        cfg = load_run_config(config_name, configs_dir)
    except Exception as e:
        return RunResult(config_name, dataset, seed, 0.0, {}, status="error", error=str(e))

    if _preloaded is not None:
        X_train, X_val, X_test, y_train, y_val, y_test = _preloaded
        if verbose:
            print(f"[{config_name}] {dataset} seed={seed} — data cached, skipping load")
    else:
        if verbose:
            print(f"[{config_name}] {dataset} seed={seed} — loading data ...")
        try:
            X_train, X_val, X_test, y_train, y_val, y_test = _load_dataset(dataset, seed, data_dir)
        except Exception as e:
            return RunResult(config_name, dataset, seed, 0.0, {}, status="error", error=f"data load: {e}")
        if verbose:
            print(f"  train={len(X_train):,}  val={len(X_val):,}  test={len(X_test):,}")

    # ── Detection ────────────────────────────────────────────────────────────
    try:
        (ae, if_det), (ae_norm, if_norm, alpha, ae_feature_indices), threshold, threshold_diag = _train_or_load_detector(
            cfg, X_train, y_train, X_val, y_val, artifact_dir, config_name, dataset, seed
        )
        detector = _EnsembleScoreAdapter(
            ae=ae,
            if_det=if_det,
            ae_norm=ae_norm,
            if_norm=if_norm,
            alpha=alpha,
            ae_feature_indices=ae_feature_indices,
        )
        ens_scores = detector.score(X_test)
        y_pred = (ens_scores >= threshold).astype(int)

        from caushap_nids.evaluation import compute_detection_metrics
        det_metrics = compute_detection_metrics(y_test, y_pred, ens_scores,
                                                n_bootstrap=200, max_bootstrap_samples=20_000)
        det_metrics.update(threshold_diag)
    except Exception as e:
        return RunResult(config_name, dataset, seed, time.perf_counter() - t0, {},
                         status="error", error=f"detection: {e}")

    # ── Select anomalous flows for XAI evaluation ─────────────────────────────
    anomaly_idx = _select_explanation_indices(y_pred, rng, n_explain)
    X_explain = X_test[anomaly_idx]

    # ── Load DAG if configured ────────────────────────────────────────────────
    try:
        dag = _load_configured_dag(cfg, artifact_dir, seed)
    except Exception:
        dag = None

    # ── Faithfulness (Layer A Shapley) ────────────────────────────────────────
    faith_metrics: dict[str, Any] = {}
    shapley_method = cfg.shapley.get("method")
    if shapley_method in {"vanilla_kernel", "causal_interventional"} and len(X_explain) > 0:
        try:
            from caushap_nids.xai_layers.causal_shapley import causal_shapley_values, vanilla_kernel_shap
            from caushap_nids.evaluation import sufficiency, comprehensiveness, lipschitz_stability

            causal_method = cfg.shapley.get("causal_method", "interventional")
            n_samples = cfg.shapley.get("n_samples", 200)
            stability_smoothing = float(cfg.shapley.get("stability_smoothing", 0.0))

            # Build a subsampled benign background for SHAP (keeps each call fast)
            bg_benign = X_train[y_train == 0]
            n_bg = min(cfg.shapley.get("n_background", 200), len(bg_benign))
            bg_rng = np.random.default_rng(seed)
            bg = bg_benign[bg_rng.choice(len(bg_benign), size=n_bg, replace=False)]
            X_faith = X_explain[:min(MAX_FAITHFULNESS_FLOWS, len(X_explain))]

            if shapley_method == "causal_interventional":
                if dag is None:
                    raise ValueError("causal_interventional Shapley requires a configured DAG")

                def explain_one(x: np.ndarray, samples: int = n_samples) -> np.ndarray:
                    return causal_shapley_values(
                        detector, dag, x, bg,
                        n_samples=samples, causal_method=causal_method,
                        stability_smoothing=stability_smoothing,
                    ).phi
            else:
                from caushap_nids.data_pipeline.loaders import NF_V2_FEATURE_COLS

                def explain_one(x: np.ndarray, samples: int = n_samples) -> np.ndarray:
                    return vanilla_kernel_shap(
                        detector, x, bg,
                        n_samples=samples, feature_names=NF_V2_FEATURE_COLS,
                    ).phi

            phi = np.array([explain_one(X_faith[i]) for i in range(len(X_faith))])

            suf  = float(np.mean([sufficiency(detector, X_faith[i], phi[i], background=bg)
                                   for i in range(len(X_faith))]))
            comp = float(np.mean([comprehensiveness(detector, X_faith[i], phi[i], background=bg)
                                   for i in range(len(X_faith))]))
            # Lipschitz mode: max (default Alvarez-Melis 2018) or robust quantile.
            # The CAUSHAP_LIPSCHITZ_QUANTILE env var (e.g. "0.95") opts into the
            # outlier-trimmed estimator described in faithfulness.lipschitz_stability.
            # The CAUSHAP_LIPSCHITZ_ROBUST_ALL=1 env var additionally computes a
            # per-instance Lipschitz vector across the full X_faith batch so we can
            # report mean + median + per-instance distribution rather than the noisy
            # single-flow point estimate.
            import os as _os
            _q_env = _os.environ.get("CAUSHAP_LIPSCHITZ_QUANTILE")
            _q = float(_q_env) if _q_env else None
            _lip_fn = lambda x: explain_one(x, samples=min(50, n_samples))
            lip = lipschitz_stability(_lip_fn, X_faith[0], n_perturbations=20, seed=seed, quantile=_q)
            faith_metrics = {
                "method": shapley_method,
                "sufficiency": suf,
                "comprehensiveness": comp,
                "lipschitz": lip,
                "n_explained": len(X_faith),
            }
            if _os.environ.get("CAUSHAP_LIPSCHITZ_ROBUST_ALL") == "1":
                _lip_all = [
                    lipschitz_stability(_lip_fn, X_faith[i], n_perturbations=20, seed=seed, quantile=_q)
                    for i in range(len(X_faith))
                ]
                faith_metrics["lipschitz_robust"] = float(np.mean(_lip_all))
                faith_metrics["lipschitz_per_instance"] = [float(v) for v in _lip_all]
                faith_metrics["lipschitz_quantile_mode"] = _q if _q is not None else "max"
        except Exception as e:
            faith_metrics = {"error": str(e)}

    # ── CF metrics (Layer B) ──────────────────────────────────────────────────
    cf_result: dict[str, Any] = {}
    cf_method = cfg.counterfactual.get("method")
    # CAUSHAP_LIPSCHITZ_ONLY=1 skips the slow CF generation when re-running just
    # for faithfulness probes (recompute_lipschitz_robust.py uses this).
    import os as _os
    _skip_cf = _os.environ.get("CAUSHAP_LIPSCHITZ_ONLY") == "1"
    if cf_method in {"nsga2", "vanilla_dice"} and len(X_explain) > 0 and not _skip_cf:
        try:
            from caushap_nids.xai_layers.multi_obj_cf import generate_cf_pareto_front
            from caushap_nids.xai_layers.multi_obj_cf.baselines import generate_dice_cfs
            from caushap_nids.evaluation import aggregate_cf_metrics, cf_hypervolume
            from caushap_nids.data_pipeline.loaders import NF_V2_FEATURE_COLS

            X_cf_inputs = X_explain[:min(MAX_CF_FLOWS, len(X_explain))]
            X_bg_benign = X_train[y_train == 0]
            all_cfs: list = []
            front_hypervolumes: list[float] = []
            generation_dag = dag or _empty_feature_dag(NF_V2_FEATURE_COLS)
            for x_i in X_cf_inputs:
                if cf_method == "nsga2":
                    if dag is None:
                        raise ValueError("nsga2 counterfactuals require a configured DAG")
                    cfs_i = generate_cf_pareto_front(
                        detector, generation_dag, x_i,
                        feature_names=NF_V2_FEATURE_COLS,
                        threshold=threshold,
                        background=X_bg_benign,
                        population_size=cfg.counterfactual.get("population_size", 100),
                        n_generations=cfg.counterfactual.get("n_generations", 200),
                        n_cfs=cfg.counterfactual.get("n_cfs"),
                        return_valid_only=True,
                        seed=seed,
                    )
                else:
                    cfs_i = generate_dice_cfs(
                        detector, X_bg_benign, x_i,
                        feature_names=NF_V2_FEATURE_COLS,
                        dag=generation_dag,
                        threshold=threshold,
                        total_cfs=cfg.counterfactual.get("n_cfs", 5),
                        method="random",
                        seed=seed,
                    )
                all_cfs.extend(cfs_i)
                front_hypervolumes.append(cf_hypervolume(cfs_i))

            eval_dag = generation_dag
            eval_dag_name = "configured"
            # Feasibility against the graph the search actually optimised for.
            # Recorded before any rescoring so both readings survive.
            feas_own_graph = _mean_feasibility_against(
                all_cfs, generation_dag, NF_V2_FEATURE_COLS
            )
            feas_expert_graph = float("nan")
            if cfg.dag and cfg.dag.get("random", False):
                expert_dag = _load_expert_dag(artifact_dir)
                if expert_dag is not None:
                    eval_dag = expert_dag
                    eval_dag_name = "expert_nf_dag_v1"
                    feas_expert_graph = _mean_feasibility_against(
                        all_cfs, expert_dag, NF_V2_FEATURE_COLS
                    )
                    _evaluate_cfs_against_dag(all_cfs, eval_dag, NF_V2_FEATURE_COLS)

            cf_result = aggregate_cf_metrics(all_cfs)
            cf_result["feasibility_rate_own_graph"] = feas_own_graph
            cf_result["feasibility_rate_expert_graph"] = feas_expert_graph
            cf_result["hypervolume"] = float(np.mean(front_hypervolumes)) if front_hypervolumes else 0.0
            cf_result["total_hypervolume"] = cf_hypervolume(all_cfs)
            cf_result["method"] = cf_method
            cf_result["n_inputs"] = len(X_cf_inputs)
            cf_result["n_cfs"] = len(all_cfs)
            cf_result["evaluation_dag"] = eval_dag_name
        except Exception as e:
            cf_result = {"error": str(e)}

    # ── Concept metrics (Layer C) ──────────────────────────────────────────────
    concept_result: dict[str, Any] = {}
    if cfg.concepts and isinstance(cfg.concepts, dict) and len(X_explain) > 0:
        gate_metrics = _load_module5c_gate_metrics(artifact_dir)
        use_gate_summary = bool(cfg.concepts.get("use_gate_summary", True))
        if use_gate_summary and gate_metrics is not None:
            concept_result = gate_metrics
        else:
            try:
                from caushap_nids.xai_layers.concept_abduction import (
                    build_concept_subgraphs,
                    compute_background_stats,
                    calibrate_thresholds,
                    abductive_explanation,
                    concept_fidelity,
                )
                from caushap_nids.data_pipeline.loaders import NF_V2_FEATURE_COLS as feat_cols

                bg_benign = X_train[y_train == 0]
                bg_stats = compute_background_stats(bg_benign)
                concept_lib = build_concept_subgraphs(dag) if dag else {}
                thresholds = calibrate_thresholds(
                    concept_lib, bg_benign, feat_cols, bg_stats,
                    target_fpr=cfg.concepts.get("target_fpr", 0.05),
                )

                fidelities = []
                for x in X_explain:
                    expl = abductive_explanation(
                        x, concept_lib, thresholds, bg_stats, feat_cols,
                        top_k=cfg.concepts.get("top_k", 3),
                    )
                    fidelities.append(concept_fidelity(detector, x, expl, concept_lib, feat_cols, bg_stats))
                concept_result = {
                    "mean_fidelity": float(np.mean(fidelities)),
                    "n_explained": len(fidelities),
                    "method": "runtime_x_explain_sample",
                }
            except Exception as e:
                concept_result = {"error": str(e)}

    elapsed = time.perf_counter() - t0
    result = RunResult(
        config_name=config_name,
        dataset=dataset,
        seed=seed,
        elapsed_seconds=round(elapsed, 2),
        detection=det_metrics,
        faithfulness=faith_metrics,
        cf_metrics=cf_result,
        concept_metrics=concept_result,
        metadata={
            "criteria_version": CRITERIA_VERSION,
            "config_file": str(configs_dir / f"{config_name}.yaml"),
            "git_commit": _git_commit_hash(configs_dir.parent),
            "n_explain": n_explain,
            "n_predicted_anomalies": int(np.asarray(y_pred).sum()),
            "n_xai_selected": int(len(X_explain)),
            "config_hash": _stable_json_hash(cfg.__dict__),
        },
    )

    # Auto-save. GUARD: when running with CAUSHAP_LIPSCHITZ_ONLY=1 (CF block is
    # skipped, cf_metrics empty), save to a *.partial.json path so we don't
    # silently overwrite the matrix's full result.json. The recompute scripts
    # read faithfulness from the returned RunResult anyway.
    import os as _os
    out_path = _result_path(artifact_dir, config_name, dataset, seed)
    if _os.environ.get("CAUSHAP_LIPSCHITZ_ONLY") == "1":
        out_path = out_path.with_suffix(".partial.json")
    result.save(out_path)
    if verbose:
        print(f"  Done in {elapsed:.1f}s — saved {out_path}")
    return result


def run_ablation_matrix(
    configs: tuple[str, ...] = CONFIG_NAMES,
    datasets: tuple[str, ...] = DATASETS,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    *,
    data_dir: str | Path = "data",
    artifact_dir: str | Path = "artifacts",
    configs_dir: str | Path = "configs",
    n_explain: int = 200,
    verbose: bool = True,
) -> list[RunResult]:
    """Run the full ablation matrix: configs × datasets × seeds → list[RunResult].

    Loop order is datasets × seeds × configs so each (dataset, seed) pair is
    loaded from disk only once and reused across all 6 configs.
    """
    results = []
    total = len(configs) * len(datasets) * len(seeds)
    done = 0
    data_dir = Path(data_dir)
    artifact_dir = Path(artifact_dir)
    configs_dir = Path(configs_dir)
    _tune_runtime_once(verbose=verbose)
    for ds in datasets:
        for seed in seeds:
            if verbose:
                print(f"\n── loading {ds} seed={seed} ──")
            try:
                preloaded = _load_dataset(ds, seed, data_dir)
            except Exception as e:
                if verbose:
                    print(f"  ERROR loading {ds}: {e}")
                for cfg_name in configs:
                    done += 1
                    results.append(RunResult(cfg_name, ds, seed, 0.0, {},
                                             status="error", error=f"data load: {e}"))
                continue
            if verbose:
                print(f"  train={len(preloaded[0]):,}  val={len(preloaded[1]):,}  test={len(preloaded[2]):,}")
            for cfg_name in configs:
                done += 1
                if verbose:
                    print(f"\n── [{done}/{total}] {cfg_name}  {ds}  seed={seed} ──")
                try:
                    cfg = load_run_config(cfg_name, configs_dir)
                    config_hash = _stable_json_hash(cfg.__dict__)
                    cached = _reusable_result(
                        _result_path(artifact_dir, cfg_name, ds, seed),
                        n_explain=n_explain,
                        config_hash=config_hash,
                    )
                except Exception:
                    cached = None
                if cached is not None:
                    if verbose:
                        print("  checkpoint ok — using saved result.json")
                    results.append(cached)
                    continue
                r = run_single(
                    cfg_name, ds, seed,
                    data_dir=data_dir,
                    artifact_dir=artifact_dir,
                    configs_dir=configs_dir,
                    n_explain=n_explain,
                    verbose=verbose,
                    _preloaded=preloaded,
                )
                results.append(r)
    return results


def load_all_results(
    artifact_dir: str | Path = "artifacts",
    *,
    criteria_version: str | None = None,
    n_explain: int | None = None,
    require_ok: bool = False,
) -> list[RunResult]:
    """Re-hydrate saved result.json files from a previous matrix run.

    Optional filters keep stale artifacts from earlier criteria versions or
    smoke-test runs out of paper tables while preserving the broad default used
    by unit tests and ad-hoc inspection.
    """
    artifact_dir = Path(artifact_dir)
    results = []
    for path in sorted((artifact_dir / "results").rglob("result.json")):
        raw = json.loads(path.read_text())
        result = RunResult(**raw)
        meta = result.metadata or {}
        if criteria_version is not None and meta.get("criteria_version") != criteria_version:
            continue
        if n_explain is not None and meta.get("n_explain") != n_explain:
            continue
        if require_ok and result.status != "ok":
            continue
        results.append(result)
    return results
