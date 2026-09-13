"""Verify the public artifact without training or changing frozen results."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "ARTIFACT_MANIFEST_SHA256.csv"
RESULTS = ROOT / "results"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def close(observed: float, expected: float, tolerance: float = 1e-12) -> None:
    if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=tolerance):
        raise RuntimeError(f"Expected {expected}, observed {observed}")


def verify_manifest() -> None:
    if not MANIFEST.is_file():
        raise FileNotFoundError(f"Missing manifest: {MANIFEST}")
    rows = read_rows(MANIFEST)
    seen: set[str] = set()
    for row in rows:
        relative = row["path"]
        if relative in seen:
            raise RuntimeError(f"Duplicate manifest row: {relative}")
        seen.add(relative)
        path = ROOT / Path(relative)
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != int(row["size_bytes"]):
            raise RuntimeError(f"Size mismatch: {relative}")
        if sha256(path) != row["sha256"]:
            raise RuntimeError(f"SHA-256 mismatch: {relative}")


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def verify_headline_values() -> None:
    cutoff = read_rows(RESULTS / "cutoff_metrics_per_seed.csv")
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in cutoff:
        grouped[(row["method"], row["layer"])].append(float(row["r95"]))
    expected_r95 = {
        ("SNN-DFA", "H1"): 553.8,
        ("SNN-DFA", "H2"): 533.0,
        ("SNN-DFA", "H3"): 518.8,
        ("matched SNN-BPTT", "H1"): 13.4,
        ("matched SNN-BPTT", "H2"): 12.4,
        ("matched SNN-BPTT", "H3"): 9.6,
        ("temporal ANN-DFA", "H1"): 533.6,
        ("temporal ANN-DFA", "H2"): 248.8,
        ("temporal ANN-DFA", "H3"): 121.8,
    }
    for identity, expected in expected_r95.items():
        values = grouped[identity]
        if len(values) != 5:
            raise RuntimeError(f"Expected five seeds for {identity}, got {len(values)}")
        close(mean(values), expected)

    covariance = read_rows(
        RESULTS
        / "frozen_sources/covariance/covariance_decomposition_per_seed.csv"
    )
    within: dict[str, list[float]] = defaultdict(list)
    for row in covariance:
        if (
            row["experiment"] == "nmnist"
            and row["condition"] == "actual"
            and row["signal"] == "delta"
            and row["component"] == "within_time"
        ):
            within[row["layer"]].append(float(row["trace_fraction_of_total"]))
    expected_within = {
        "H1": 0.8925540113342567,
        "H2": 0.8831983053820581,
        "H3": 0.9207374178097846,
    }
    for layer, expected in expected_within.items():
        if len(within[layer]) != 5:
            raise RuntimeError(f"Expected five covariance rows for {layer}")
        close(mean(within[layer]), expected)


def verify_table_inventory() -> None:
    expected_rows = {
        "nmnist_main_per_seed.csv": 45,
        "cutoff_metrics_per_seed.csv": 75,
        "threshold_free_metrics_per_seed.csv": 75,
        "homogenization_per_seed_layer_alpha.csv": 75,
        "controls_per_seed.csv": 945,
        "memoryless_per_seed.csv": 9,
        "dvs_per_seed.csv": 9,
        "probe_size_sensitivity.csv": 2700,
        "width_robustness.csv": 27,
    }
    for name, expected in expected_rows.items():
        rows = read_rows(RESULTS / name)
        if len(rows) != expected:
            raise RuntimeError(f"{name}: expected {expected} rows, got {len(rows)}")
        provenance_columns = [
            column
            for column in rows[0]
            if column == "provenance_file"
            or column == "source_file"
            or column.startswith("source_file_")
        ]
        if not provenance_columns:
            raise RuntimeError(f"{name}: missing row-level provenance column")
        for row_number, row in enumerate(rows, start=2):
            if not any(row[column].strip() for column in provenance_columns):
                raise RuntimeError(f"{name}: row {row_number} has no provenance value")

    homogenization = read_rows(RESULTS / "homogenization_per_seed_layer_alpha.csv")
    observed_alphas = sorted({float(row["alpha"]) for row in homogenization})
    if observed_alphas != [0.0, 0.25, 0.5, 0.75, 1.0]:
        raise RuntimeError(f"Unexpected homogenization alpha grid: {observed_alphas}")


def verify_bundled_checkpoint() -> None:
    rows = read_rows(ROOT / "metadata/checkpoints/checkpoint_manifest.csv")
    bundled = [row for row in rows if row["availability"].startswith("bundled_")]
    if len(bundled) != 1:
        raise RuntimeError(f"Expected one bundled smoke checkpoint, got {len(bundled)}")
    row = bundled[0]
    path = ROOT / row["bundled_relative_path"]
    if sha256(path) != row["bundled_sha256"]:
        raise RuntimeError("Bundled release checkpoint does not match its SHA-256")
    report = json.loads((ROOT / row["release_transform_report"]).read_text(encoding="utf-8"))
    if report["canonical_original_sha256"] != row["expected_sha256"]:
        raise RuntimeError("Release transform does not identify the frozen canonical SHA-256")
    if report["release_sha256"] != row["bundled_sha256"]:
        raise RuntimeError("Release transform report does not match the bundled checkpoint")
    if not report["scientific_state_equal_after_reload"]:
        raise RuntimeError("Release transform did not preserve scientific checkpoint state")


def main() -> None:
    verify_manifest()
    verify_headline_values()
    verify_table_inventory()
    verify_bundled_checkpoint()
    print("PASS: manifest, table inventory, checkpoint, and frozen values verified")


if __name__ == "__main__":
    main()
