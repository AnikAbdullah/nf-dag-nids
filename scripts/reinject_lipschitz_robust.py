"""Re-inject lipschitz_robust values from lipschitz_robust_recompute.json into
result.json files for any cell that's missing them.

Idempotent — safe to run repeatedly. Triggered after `run_single` recovery
which only populates the single-instance `lipschitz` field; the robust mean
needs to be carried over from the prior recompute artifact.
"""
import json
import sys
from pathlib import Path

ART = Path(__file__).resolve().parent.parent / "artifacts"
LR_PATH = ART / "lipschitz_robust_recompute.json"
RESULTS = ART / "results"


def main():
    lr = json.loads(LR_PATH.read_text())
    patched = 0
    skipped = 0
    for key, entry in lr.items():
        cfg, ds, seed = key.split("/")
        rp = RESULTS / cfg / ds / seed / "result.json"
        if not rp.exists():
            continue
        r = json.loads(rp.read_text())
        fa = r.get("faithfulness") or {}
        if "lipschitz_robust" in fa and fa["lipschitz_robust"] is not None:
            skipped += 1
            continue
        lip_robust = entry.get("lip_robust")
        if lip_robust is None:
            continue
        fa["lipschitz_robust"] = float(lip_robust)
        r["faithfulness"] = fa
        rp.write_text(json.dumps(r, indent=2))
        print(f"  patched {key}: lipschitz_robust={lip_robust:.4f}")
        patched += 1
    print(f"\nReinjected {patched} cells, skipped {skipped} (already present).")


if __name__ == "__main__":
    main()
