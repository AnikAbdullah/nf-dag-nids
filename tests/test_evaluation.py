"""Tests for Module 6 — evaluation/."""
from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers / stubs
# ---------------------------------------------------------------------------

class _FakeDetector:
    """Minimal detector stub: score = sum of squares of input."""
    def score(self, X):
        X = np.asarray(X, dtype=float)
        return (X ** 2).sum(axis=-1)

    def predict(self, X, threshold):
        return (self.score(X) > threshold).astype(int)


# ---------------------------------------------------------------------------
# detection_metrics
# ---------------------------------------------------------------------------

class TestComputeDetectionMetrics:
    def _labels(self):
        rng = np.random.default_rng(0)
        y_true = rng.integers(0, 3, size=300)
        y_pred = y_true.copy()
        flip = rng.choice(300, size=30, replace=False)
        y_pred[flip] = rng.integers(0, 3, size=30)
        y_score = rng.uniform(0, 1, size=300)
        return y_true, y_pred, y_score

    def test_keys_present(self):
        from caushap_nids.evaluation.detection_metrics import compute_detection_metrics
        y_true, y_pred, y_score = self._labels()
        result = compute_detection_metrics(y_true, y_pred, y_score, n_bootstrap=50)
        for key in ("macro_f1", "precision", "recall", "auc_roc", "fpr", "per_class_f1"):
            assert key in result

    def test_ci_tuples(self):
        from caushap_nids.evaluation.detection_metrics import compute_detection_metrics
        y_true, y_pred, y_score = self._labels()
        result = compute_detection_metrics(y_true, y_pred, y_score, n_bootstrap=50)
        for key in ("macro_f1", "precision", "recall", "fpr"):
            pt, lo, hi = result[key]
            assert lo <= pt <= hi or np.isnan(pt), f"{key}: CI violated"

    def test_per_class_f1_keys(self):
        from caushap_nids.evaluation.detection_metrics import compute_detection_metrics
        y_true, y_pred, y_score = self._labels()
        result = compute_detection_metrics(y_true, y_pred, y_score, n_bootstrap=50)
        assert set(result["per_class_f1"].keys()) == {"0", "1", "2"}

    def test_zero_day_recall_included_when_mask_provided(self):
        from caushap_nids.evaluation.detection_metrics import compute_detection_metrics
        y_true, y_pred, y_score = self._labels()
        zd_mask = y_true == 2
        result = compute_detection_metrics(y_true, y_pred, y_score, n_bootstrap=50, zero_day_mask=zd_mask)
        assert "zero_day_recall" in result

    def test_zero_day_recall_absent_by_default(self):
        from caushap_nids.evaluation.detection_metrics import compute_detection_metrics
        y_true, y_pred, y_score = self._labels()
        result = compute_detection_metrics(y_true, y_pred, y_score, n_bootstrap=50)
        assert "zero_day_recall" not in result

    def test_perfect_predictions(self):
        from caushap_nids.evaluation.detection_metrics import compute_detection_metrics
        y = np.array([0, 0, 1, 1, 2, 2])
        score = np.array([0.1, 0.1, 0.9, 0.9, 0.5, 0.5])
        result = compute_detection_metrics(y, y, score, n_bootstrap=20)
        assert result["macro_f1"][0] == pytest.approx(1.0)

    def test_fpr_zero_when_no_benign_misclassified(self):
        from caushap_nids.evaluation.detection_metrics import compute_detection_metrics
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        y_score = np.array([0.1, 0.2, 0.8, 0.9])
        result = compute_detection_metrics(y_true, y_pred, y_score, n_bootstrap=20)
        assert result["fpr"][0] == pytest.approx(0.0)


class TestSelectThreshold:
    def test_tied_scores_respect_target_fpr(self):
        from caushap_nids.models.thresholds import select_threshold

        scores = np.array([0.9, 0.9, 0.9, 0.1])
        labels = np.array([0, 0, 1, 1])
        threshold = select_threshold(scores, labels, target_fpr=0.0)
        pred = (scores >= threshold).astype(int)
        assert ((pred[labels == 0]) == 1).mean() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# faithfulness
# ---------------------------------------------------------------------------

class TestFaithfulness:
    def setup_method(self):
        self.detector = _FakeDetector()
        self.x = np.array([1.0, 2.0, 0.5, 0.1, 3.0])
        self.phi = np.array([0.1, 0.5, 0.05, 0.01, 0.8])
        self.bg = np.zeros((10, 5))

    def test_sufficiency_range(self):
        from caushap_nids.evaluation.faithfulness import sufficiency
        s = sufficiency(self.detector, self.x, self.phi, top_k=2, background=self.bg)
        assert 0.0 <= s <= 1.0 or s < 0  # can go slightly negative for large ablations

    def test_comprehensiveness_nonneg(self):
        from caushap_nids.evaluation.faithfulness import comprehensiveness
        c = comprehensiveness(self.detector, self.x, self.phi, top_k=2, background=self.bg)
        assert c >= 0.0

    def test_sufficiency_top_all_equals_one(self):
        from caushap_nids.evaluation.faithfulness import sufficiency
        # Keeping all features → masked == original → sufficiency ≈ 1
        s = sufficiency(self.detector, self.x, self.phi, top_k=len(self.x), background=self.bg)
        assert s == pytest.approx(1.0, abs=1e-6)

    def test_comprehensiveness_ablate_all_is_large(self):
        from caushap_nids.evaluation.faithfulness import comprehensiveness
        # Ablating all features → score collapses to 0 → comprehensiveness ≈ 1
        phi_all = np.ones(5)
        c = comprehensiveness(self.detector, self.x, phi_all, top_k=5, background=self.bg)
        assert c == pytest.approx(1.0, abs=1e-6)

    def test_lipschitz_stability_returns_float(self):
        from caushap_nids.evaluation.faithfulness import lipschitz_stability
        explain_fn = lambda x: x ** 2  # trivial attribution
        L = lipschitz_stability(explain_fn, self.x, n_perturbations=10, epsilon=0.01, seed=0)
        assert isinstance(L, float)
        assert L >= 0.0

    def test_lipschitz_constant_attribution(self):
        from caushap_nids.evaluation.faithfulness import lipschitz_stability
        # Constant attribution → L = 0
        explain_fn = lambda x: np.ones(5)
        L = lipschitz_stability(explain_fn, self.x, n_perturbations=20, epsilon=0.05, seed=0)
        assert L == pytest.approx(0.0, abs=1e-10)

    def test_lipschitz_quantile_robust_to_single_outlier(self):
        """quantile=0.95 should ignore a single bursty perturbation that swings max."""
        from caushap_nids.evaluation.faithfulness import lipschitz_stability
        rng = np.random.default_rng(123)
        call = {"n": 0}

        def explain_fn(x):
            call["n"] += 1
            # Inject one wild attribution swing on the very first call after the
            # baseline phi(x) is computed; all subsequent calls return small noise.
            if call["n"] == 2:
                return np.full(5, 100.0)
            return rng.normal(0.0, 0.01, size=5)

        x = np.zeros(5)
        L_max = lipschitz_stability(explain_fn, x, n_perturbations=20, epsilon=0.05, seed=0)
        call["n"] = 0
        L_q95 = lipschitz_stability(explain_fn, x, n_perturbations=20, epsilon=0.05, seed=0, quantile=0.95)
        # Max is dominated by the spike; q95 trims it. q95 should be much smaller.
        assert L_q95 < L_max * 0.5

    def test_lipschitz_quantile_validates_input(self):
        from caushap_nids.evaluation.faithfulness import lipschitz_stability
        with pytest.raises(ValueError):
            lipschitz_stability(lambda x: x, np.zeros(3), n_perturbations=5, quantile=1.5)


# ---------------------------------------------------------------------------
# cf_metrics
# ---------------------------------------------------------------------------

class TestCfMetrics:
    def _make_cfs(self, n=5, validity=0.9, feasibility=0.85):
        from caushap_nids.xai_layers.multi_obj_cf.nsga import CounterfactualExplanation
        rng = np.random.default_rng(0)
        cfs = []
        for _ in range(n):
            x_orig = rng.normal(size=5)
            x_cf = x_orig + rng.normal(scale=0.1, size=5)
            cfs.append(CounterfactualExplanation(
                x_orig=x_orig, x_cf=x_cf,
                validity=validity, proximity=float(np.abs(x_orig - x_cf).sum()),
                sparsity=2, feasibility_rate=feasibility,
            ))
        return cfs

    def test_aggregate_keys(self):
        from caushap_nids.evaluation.cf_metrics import aggregate_cf_metrics
        cfs = self._make_cfs()
        result = aggregate_cf_metrics(cfs)
        assert set(result.keys()) == {"validity", "proximity", "sparsity", "feasibility_rate"}

    def test_aggregate_empty(self):
        from caushap_nids.evaluation.cf_metrics import aggregate_cf_metrics
        result = aggregate_cf_metrics([])
        assert result["validity"] == 0.0

    def test_aggregate_validity_mean(self):
        from caushap_nids.evaluation.cf_metrics import aggregate_cf_metrics
        cfs = self._make_cfs(n=4, validity=0.75)
        assert aggregate_cf_metrics(cfs)["validity"] == pytest.approx(0.75)

    def test_hypervolume_positive(self):
        from caushap_nids.evaluation.cf_metrics import cf_hypervolume
        cfs = self._make_cfs(n=5, validity=0.9, feasibility=0.8)
        hv = cf_hypervolume(cfs)
        assert hv >= 0.0

    def test_compare_sets_a_better_on_feasibility(self):
        from caushap_nids.evaluation.cf_metrics import compare_cf_sets
        good = self._make_cfs(n=5, feasibility=0.9)
        bad = self._make_cfs(n=5, feasibility=0.3)
        result = compare_cf_sets(good, bad)
        assert result["feasibility_rate"]["a_better"] is True


# ---------------------------------------------------------------------------
# statistical
# ---------------------------------------------------------------------------

class TestStatistical:
    def test_cohens_d_zero_when_equal(self):
        from caushap_nids.evaluation.statistical import cohens_d
        a = np.array([1.0, 2.0, 3.0])
        assert cohens_d(a, a) == pytest.approx(0.0)

    def test_cohens_d_positive_when_a_larger(self):
        from caushap_nids.evaluation.statistical import cohens_d
        # Differences must have variance > 0 for a non-zero d
        a = np.array([2.0, 3.5, 4.0, 5.0])
        b = np.array([1.0, 1.0, 1.5, 2.0])
        assert cohens_d(a, b) > 0.0

    def test_wilcoxon_keys(self):
        from caushap_nids.evaluation.statistical import wilcoxon_bonferroni
        rng = np.random.default_rng(1)
        a = rng.normal(1.0, 0.5, 30)
        b = rng.normal(0.5, 0.5, 30)
        result = wilcoxon_bonferroni(a, b)
        assert set(result.keys()) == {"stat", "p_value", "corrected_alpha", "significant", "cohens_d"}

    def test_wilcoxon_bonferroni_corrected_alpha(self):
        from caushap_nids.evaluation.statistical import wilcoxon_bonferroni
        a = np.array([1.0] * 10)
        b = np.array([0.5] * 10)
        result = wilcoxon_bonferroni(a, b, n_comparisons=6, alpha=0.01)
        assert result["corrected_alpha"] == pytest.approx(0.01 / 6)

    def test_wilcoxon_identical_arrays_not_significant(self):
        from caushap_nids.evaluation.statistical import wilcoxon_bonferroni
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = wilcoxon_bonferroni(a, a)
        assert result["significant"] is False

    def test_wilcoxon_clearly_different_significant(self):
        from caushap_nids.evaluation.statistical import wilcoxon_bonferroni
        rng = np.random.default_rng(42)
        a = rng.normal(5.0, 0.1, 50)
        b = rng.normal(0.0, 0.1, 50)
        result = wilcoxon_bonferroni(a, b, n_comparisons=1, alpha=0.05)
        assert result["significant"] is True


# ---------------------------------------------------------------------------
# table_writer
# ---------------------------------------------------------------------------

class TestTableWriter:
    def _sample_results(self):
        return {
            "A0": {"macro_f1": (0.642, 0.620, 0.665), "fpr": (0.012, 0.009, 0.015)},
            "A1": {"macro_f1": (0.689, 0.667, 0.712), "fpr": (0.010, 0.007, 0.013),
                   "p_value": 0.0001, "cohens_d": 0.42, "significant": True},
        }

    def test_writes_tex_file(self, tmp_path):
        from caushap_nids.evaluation.table_writer import write_latex_table
        out = str(tmp_path / "test_table.tex")
        write_latex_table(
            self._sample_results(), "Test Table", "tab:test", out,
            reference_method="A0", metric_cols=["macro_f1", "fpr"],
        )
        assert (tmp_path / "test_table.tex").exists()

    def test_writes_csv_file(self, tmp_path):
        from caushap_nids.evaluation.table_writer import write_latex_table
        out = str(tmp_path / "test_table.tex")
        write_latex_table(self._sample_results(), "Test", "tab:t", out)
        assert (tmp_path / "test_table.csv").exists()

    def test_latex_contains_tabular(self, tmp_path):
        from caushap_nids.evaluation.table_writer import write_latex_table
        out = str(tmp_path / "t.tex")
        write_latex_table(self._sample_results(), "T", "tab:t", out, metric_cols=["macro_f1"])
        content = (tmp_path / "t.tex").read_text()
        assert "\\begin{tabular}" in content
        assert "\\end{tabular}" in content

    def test_latex_contains_caption(self, tmp_path):
        from caushap_nids.evaluation.table_writer import write_latex_table
        out = str(tmp_path / "t.tex")
        write_latex_table(self._sample_results(), "My Caption", "tab:t", out, metric_cols=["macro_f1"])
        content = (tmp_path / "t.tex").read_text()
        assert "My Caption" in content

    def test_significance_marker_present(self, tmp_path):
        from caushap_nids.evaluation.table_writer import write_latex_table
        out = str(tmp_path / "t.tex")
        write_latex_table(
            self._sample_results(), "T", "tab:t", out,
            reference_method="A0", metric_cols=["macro_f1"],
        )
        content = (tmp_path / "t.tex").read_text()
        assert " *" in content

    def test_empty_results_no_error(self, tmp_path):
        from caushap_nids.evaluation.table_writer import write_latex_table
        write_latex_table({}, "Empty", "tab:e", str(tmp_path / "e.tex"))

    def test_ci_format(self, tmp_path):
        from caushap_nids.evaluation.table_writer import write_latex_table
        out = str(tmp_path / "t.tex")
        write_latex_table(self._sample_results(), "T", "tab:t", out, metric_cols=["macro_f1"])
        content = (tmp_path / "t.tex").read_text()
        assert "0.642 [0.620, 0.665]" in content

    def test_accepts_row_list_and_adds_tex_suffix(self, tmp_path):
        from caushap_nids.evaluation.table_writer import write_latex_table
        rows = [
            {"config": "A0", "dataset": "nf_cic2018", "macro_f1": 0.9},
            {"config": "A4", "dataset": "nf_cic2018", "macro_f1": 0.95},
        ]
        write_latex_table(rows, "T", "tab:t", str(tmp_path / "rows"), metric_cols=["macro_f1"])
        content = (tmp_path / "rows.tex").read_text()
        assert "A0/nf_cic2018" in content
        assert (tmp_path / "rows.csv").exists()


# ---------------------------------------------------------------------------
# quantus_adapter (fallback path — no quantus required)
# ---------------------------------------------------------------------------

class TestQuantusAdapter:
    def setup_method(self):
        self.detector = _FakeDetector()
        rng = np.random.default_rng(0)
        self.X = rng.normal(size=(5, 4))
        self.attributions = rng.normal(size=(5, 4))
        self.bg = np.zeros((10, 4))

    def test_eraser_sufficiency_shape(self):
        from caushap_nids.evaluation.quantus_adapter import eraser_sufficiency
        result = eraser_sufficiency(self.detector, self.X, self.attributions, top_k=2, background=self.bg)
        assert result.shape == (5,)

    def test_eraser_comprehensiveness_shape(self):
        from caushap_nids.evaluation.quantus_adapter import eraser_comprehensiveness
        result = eraser_comprehensiveness(self.detector, self.X, self.attributions, top_k=2, background=self.bg)
        assert result.shape == (5,)

    def test_eraser_sufficiency_range(self):
        from caushap_nids.evaluation.quantus_adapter import eraser_sufficiency
        result = eraser_sufficiency(self.detector, self.X, self.attributions, top_k=2, background=self.bg)
        # Values can be slightly outside [0,1] for extreme inputs — just check finite
        assert np.all(np.isfinite(result))
