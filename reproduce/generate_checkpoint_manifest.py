"""Generate a checkpoint manifest from frozen replay metadata.

No result value is entered here: expected paths and hashes are read from the
frozen per-seed spectral table, and any bundled file is hashed byte-for-byte.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NMNIST_SOURCE = ROOT / "results/frozen_sources/nmnist/spectral_metrics_per_seed.csv"
DVS_SOURCE = ROOT / "results/frozen_sources/dvs/gate_per_seed.csv"
OUTPUT = ROOT / "metadata/checkpoints/checkpoint_manifest.csv"
BUNDLED = {
    ("N-MNIST", "SNN-DFA", "20260830"): ROOT
    / "metadata/checkpoints/nmnist_snn_dfa_seed_20260830_epoch_100_release.pt"
}
RELEASE_REPORT = ROOT / "metadata/checkpoints/checkpoint_release_provenance.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    source_rows: list[tuple[dict[str, str], str]] = []
    for path in (NMNIST_SOURCE, DVS_SOURCE):
        with path.open(newline="", encoding="utf-8-sig") as handle:
            source_rows.extend(
                (row, path.relative_to(ROOT).as_posix()) for row in csv.DictReader(handle)
            )

    unique: dict[tuple[str, str, str], tuple[dict[str, str], str]] = {}
    for row, provenance_file in source_rows:
        if (
            row["checkpoint_rule"] == "completed_epoch_100"
            and row["probe_split"] == "basis_eval"
            and row.get("temporal_mode", "timestep") == "timestep"
        ):
            identity = (row["dataset"], row["method"], row["seed"])
            unique[identity] = (row, provenance_file)

    rows: list[dict[str, str | int]] = []
    for identity in sorted(unique):
        source, provenance_file = unique[identity]
        bundled = BUNDLED.get(identity)
        bundled_exists = bool(bundled and bundled.is_file())
        bundled_hash = sha256(bundled) if bundled_exists and bundled else ""
        expected_hash = source["checkpoint_sha256"]
        availability = "hash_and_path_only"
        transform_report = ""
        if bundled_exists:
            release = json.loads(RELEASE_REPORT.read_text(encoding="utf-8"))
            if release["canonical_original_sha256"] != expected_hash:
                raise RuntimeError("Release report does not match canonical frozen hash")
            if release["release_sha256"] != bundled_hash:
                raise RuntimeError("Bundled release checkpoint hash mismatch")
            availability = "bundled_sanitized_release"
            transform_report = RELEASE_REPORT.relative_to(ROOT).as_posix()
        rows.append(
            {
                "dataset": source["dataset"],
                "method": source["method"],
                "seed": source["seed"],
                "checkpoint_rule": source["checkpoint_rule"],
                "canonical_checkpoint_path": source["checkpoint"],
                "expected_sha256": expected_hash,
                "bundled_relative_path": (
                    bundled.relative_to(ROOT).as_posix() if bundled_exists and bundled else ""
                ),
                "bundled_size_bytes": bundled.stat().st_size if bundled_exists and bundled else "",
                "bundled_sha256": bundled_hash,
                "availability": availability,
                "release_transform_report": transform_report,
                "provenance_file": provenance_file,
            }
        )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {OUTPUT}")


if __name__ == "__main__":
    main()
