import hashlib
import json
from pathlib import Path


def verify_dataset(
    dataset_name: str,
    manifest_path: str = "data/manifests/sha256_manifest.json",
) -> None:
    """Verify SHA-256 of raw dataset file against manifest.

    Raises RuntimeError if hash mismatches. Silent on success.
    """
    manifest = Path(manifest_path)
    if not manifest.exists():
        raise FileNotFoundError(f"SHA-256 manifest not found: {manifest}")

    with manifest.open() as f:
        records: dict[str, dict] = json.load(f)

    if dataset_name not in records:
        raise KeyError(f"Dataset '{dataset_name}' not in manifest {manifest}")

    entry = records[dataset_name]
    file_path = Path(entry["path"])
    expected = entry["sha256"]

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    actual = _sha256(file_path)
    if actual != expected:
        raise RuntimeError(
            f"SHA-256 mismatch for '{dataset_name}':\n"
            f"  expected: {expected}\n"
            f"  actual  : {actual}\n"
            f"File may be corrupted or tampered."
        )


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()
