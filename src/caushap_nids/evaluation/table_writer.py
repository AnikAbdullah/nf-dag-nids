from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def _fmt(value: Any) -> str:
    """Format a (point, lower, upper) CI tuple or plain float."""
    if isinstance(value, (list, tuple)) and len(value) == 3:
        pt, lo, hi = value
        return f"{float(pt):.3f} [{float(lo):.3f}, {float(hi):.3f}]"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _normalise_results(
    results: Mapping[str, Mapping[str, Any]] | Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Accept both keyed table rows and notebook-friendly row dictionaries."""
    if isinstance(results, Mapping):
        return {str(k): dict(v) for k, v in results.items()}

    normalised: dict[str, dict[str, Any]] = {}
    label_keys = ("method", "Method", "comparison", "config")
    for idx, row in enumerate(results):
        row_dict = dict(row)
        label = None
        if "config" in row_dict and "dataset" in row_dict:
            label = f"{row_dict['config']}/{row_dict['dataset']}"
        else:
            for key in label_keys:
                if key in row_dict and row_dict[key] not in (None, ""):
                    label = str(row_dict[key])
                    break
        if label is None:
            label = f"row_{idx + 1}"

        base = label
        suffix = 2
        while label in normalised:
            label = f"{base}_{suffix}"
            suffix += 1

        normalised[label] = row_dict
    return normalised


def write_latex_table(
    results: Mapping[str, Mapping[str, Any]] | Sequence[Mapping[str, Any]],
    caption: str,
    label: str,
    output_path: str,
    reference_method: str | None = None,
    metric_cols: list[str] | None = None,
) -> None:
    """
    Emit a LaTeX table and a companion CSV.

    results: {method_name: {metric: (point, lo, hi) | float | str}}
             or a list of row dictionaries from a notebook dataframe.
    Significance marker * appended when p_value < corrected_alpha and significant=True.
    Also writes a .csv alongside the .tex for quick inspection.
    """
    path = Path(output_path)
    if path.suffix == "":
        path = path.with_suffix(".tex")
    path.parent.mkdir(parents=True, exist_ok=True)

    if not results:
        return

    results = _normalise_results(results)
    methods = list(results.keys())
    if metric_cols is None:
        metric_cols = [
            k for k in results[methods[0]]
            if k not in ("p_value", "cohens_d", "significant", "corrected_alpha", "stat")
        ]

    headers = ["Method"] + [f"{m} (95\\% CI)" for m in metric_cols]
    if reference_method:
        headers += ["$p$ vs ref", "Cohen's $d$"]

    col_fmt = "l" + "r" * (len(headers) - 1)

    lines = [
        "\\begin{table}[ht]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{col_fmt}}}",
        "\\toprule",
        " & ".join(headers) + " \\\\",
        "\\midrule",
    ]

    for method, row in results.items():
        cells = [method]
        for m in metric_cols:
            cells.append(_fmt(row.get(m, "—")))
        if reference_method and method != reference_method:
            p = row.get("p_value", "—")
            d = row.get("cohens_d", "—")
            sig = row.get("significant", False)
            p_str = (f"{p:.3f}" if isinstance(p, float) else str(p)) + (" *" if sig else "")
            d_str = f"{d:.2f}" if isinstance(d, float) else str(d)
            cells += [p_str, d_str]
        elif reference_method:
            cells += ["—", "—"]
        lines.append(" & ".join(cells) + " \\\\")

    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]

    path.write_text("\n".join(lines) + "\n")

    csv_path = path.with_suffix(".csv")
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Method"] + metric_cols + (["p_vs_ref", "cohens_d"] if reference_method else []))
        for method, row in results.items():
            csv_row = [method] + [_fmt(row.get(m, "")) for m in metric_cols]
            if reference_method and method != reference_method:
                csv_row += [row.get("p_value", ""), row.get("cohens_d", "")]
            elif reference_method:
                csv_row += ["", ""]
            writer.writerow(csv_row)


def write_all_paper_tables(results_dir: str, output_dir: str) -> None:
    """
    Scan results_dir for *.json table specs and emit .tex + .csv for each.
    Called by notebooks/08_results_tables.ipynb.

    Each JSON must have: results, metric_cols; optionally caption, label, reference_method.
    """
    results_dir = Path(results_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for json_path in sorted(results_dir.glob("*.json")):
        with json_path.open() as f:
            spec = json.load(f)
        if "results" not in spec or "metric_cols" not in spec:
            continue
        table_name = json_path.stem
        write_latex_table(
            results=spec["results"],
            caption=spec.get("caption", table_name.replace("_", " ")),
            label=spec.get("label", f"tab:{table_name}"),
            output_path=str(output_dir / f"{table_name}.tex"),
            reference_method=spec.get("reference_method"),
            metric_cols=spec["metric_cols"],
        )
