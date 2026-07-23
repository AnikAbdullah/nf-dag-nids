# Module 7 — Experiments (ablation matrix orchestration)
# Owner: Abdullah Al Taieb
# Ablation matrix: A0–A5 × 4 datasets × 3 seeds = 72 runs
# Primary interface: notebooks/07_ablation_runs.ipynb
# CLI: caushap-run (see experiments/cli.py)
from caushap_nids.experiments.seed_management import set_all_seeds, derive_seed
from caushap_nids.experiments.runner import (
    RunConfig,
    RunResult,
    load_run_config,
    run_single,
    run_ablation_matrix,
    load_all_results,
    CONFIG_NAMES,
    DATASETS,
    DEFAULT_SEEDS,
    CRITERIA_VERSION,
)

__all__ = [
    "set_all_seeds",
    "derive_seed",
    "RunConfig",
    "RunResult",
    "load_run_config",
    "run_single",
    "run_ablation_matrix",
    "load_all_results",
    "CONFIG_NAMES",
    "DATASETS",
    "DEFAULT_SEEDS",
    "CRITERIA_VERSION",
]
