"""Aggregate the offline covariance decomposition into paper-ready source data."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
V9 = HERE.parent
EXPECTED = {
    "nmnist": (20260830, 20260831, 20260901, 20260908, 20260909),
    "ann": (20260830, 20260831, 20260901, 20260908, 20260909),
    "stateless": (20260830, 20260831, 20260901),
    "dvs": (20260830, 20260831, 20260901),
}
METRICS = (
    "r50",
    "r80",
    "r90",
    "r95",
    "r99",
    "stable_rank",
    "entropy_effective_rank",
    "participation_ratio",
    "scatter_trace",
    "trace_fraction_of_total",
    "identity_relative_frobenius",
)
GROUP_KEYS = (
    "experiment",
    "dataset",
    "method",
    "layer",
    "condition",
    "signal",
    "component",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf8") as source:
        return list(csv.DictReader(source))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"no rows for {path}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def as_bool(value: str) -> bool:
    return value.strip().lower() == "true"


def numeric(values: Iterable[str]) -> list[float]:
    return [float(value) for value in values if value not in ("", None)]


def mean_sd(values: Iterable[float]) -> tuple[int, float, float]:
    data = list(values)
    return len(data), mean(data), stdev(data) if len(data) > 1 else 0.0


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def aggregate(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in GROUP_KEYS)].append(row)
    output: list[dict[str, Any]] = []
    for key, group in sorted(grouped.items()):
        summary: dict[str, Any] = dict(zip(GROUP_KEYS, key))
        summary["seed_count"] = len({int(row["seed"]) for row in group})
        for metric in METRICS:
            values = numeric(row.get(metric, "") for row in group)
            if values:
                _, center, spread = mean_sd(values)
                summary[f"{metric}_mean"] = center
                summary[f"{metric}_std"] = spread
        output.append(summary)
    return output


def paired_accuracy() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = read_csv(V9 / "03_five_seed/all_runs.csv")
    selected = [
        row
        for row in rows
        if row["dataset"] == "N-MNIST"
        and row["checkpoint_rule"] == "validation_accuracy_max_tie_lower_loss"
        and row["method"] in ("SNN-DFA", "MeanGate", "ShuffledGate")
    ]
    lookup = {
        (row["method"], int(row["seed"])): float(row["test_accuracy"])
        for row in selected
    }
    per_seed: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for intervention in ("MeanGate", "ShuffledGate"):
        drops: list[float] = []
        for seed in EXPECTED["nmnist"]:
            actual = lookup[("SNN-DFA", seed)]
            changed = lookup[(intervention, seed)]
            drop = actual - changed
            drops.append(drop)
            per_seed.append(
                {
                    "seed": seed,
                    "intervention": intervention,
                    "actual_accuracy": actual,
                    "intervention_accuracy": changed,
                    "accuracy_drop_fraction": drop,
                    "accuracy_drop_percentage_points": 100.0 * drop,
                    "drop_positive": drop > 0.0,
                }
            )
        n, center, spread = mean_sd(drops)
        summaries.append(
            {
                "intervention": intervention,
                "seed_count": n,
                "positive_drop_count": sum(value > 0.0 for value in drops),
                "all_seeds_drop": all(value > 0.0 for value in drops),
                "mean_accuracy_drop_fraction": center,
                "std_accuracy_drop_fraction": spread,
                "mean_accuracy_drop_percentage_points": 100.0 * center,
                "std_accuracy_drop_percentage_points": 100.0 * spread,
            }
        )
    return per_seed, summaries


def select(
    summary: list[dict[str, Any]],
    *,
    experiment: str,
    condition: str,
    signal: str,
    component: str,
) -> list[dict[str, Any]]:
    return [
        row
        for row in summary
        if row["experiment"] == experiment
        and row["condition"] == condition
        and row["signal"] == signal
        and row["component"] == component
    ]


def build_report(
    summary: list[dict[str, Any]],
    accuracy_summary: list[dict[str, Any]],
    max_identity_error: float,
) -> str:
    lines = [
        "# Covariance Decomposition Audit",
        "",
        "## Verdict",
        "",
        "PASS — all outputs are checkpoint replays; no optimizer step or training was performed. "
        "Every replayed final timestep r95 exactly matches its frozen reference.",
        "",
        "The centered-scatter identity is",
        "",
        r"$$S_{\mathrm{total}}=S_{\mathrm{within}}+S_{\mathrm{between}},$$",
        "",
        "where the between-sample scatter includes the factor $T$. "
        f"The maximum observed relative Frobenius residual was {max_identity_error:.3e}.",
        "",
        "## N-MNIST SNN-DFA: local-error decomposition (five seeds)",
        "",
        "| Layer | Component | r95 (mean±SD) | Stable rank | Entropy rank | Trace fraction |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for layer in ("H1", "H2", "H3"):
        for component in ("total", "within_time", "between_sample"):
            row = next(
                item
                for item in summary
                if item["experiment"] == "nmnist"
                and item["layer"] == layer
                and item["condition"] == "actual"
                and item["signal"] == "delta"
                and item["component"] == component
            )
            lines.append(
                "| "
                + " | ".join(
                    (
                        layer,
                        component,
                        f"{fmt(row['r95_mean'], 1)}±{fmt(row['r95_std'], 1)}",
                        f"{fmt(row['stable_rank_mean'], 2)}±{fmt(row['stable_rank_std'], 2)}",
                        f"{fmt(row['entropy_effective_rank_mean'], 1)}±{fmt(row['entropy_effective_rank_std'], 1)}",
                        f"{fmt(row['trace_fraction_of_total_mean'], 3)}±{fmt(row['trace_fraction_of_total_std'], 3)}",
                    )
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "## Full temporal homogenization (alpha=1)",
            "",
            "The within-time term is analytically zero. Tiny float32 residual traces "
            "(≤1e-12 of total) are retained in raw fields for auditability but their "
            "scale-invariant ranks are reported as zero.",
            "",
            "| Layer | Total/between r95 | Within trace fraction | Between trace fraction |",
            "|---|---:|---:|---:|",
        ]
    )
    for layer in ("H1", "H2", "H3"):
        total = next(
            item
            for item in summary
            if item["experiment"] == "nmnist"
            and item["layer"] == layer
            and item["condition"] == "alpha_1_homogenized_frobenius_matched"
            and item["signal"] == "delta"
            and item["component"] == "total"
        )
        within = next(
            item
            for item in summary
            if item["experiment"] == "nmnist"
            and item["layer"] == layer
            and item["condition"] == "alpha_1_homogenized_frobenius_matched"
            and item["signal"] == "delta"
            and item["component"] == "within_time"
        )
        between = next(
            item
            for item in summary
            if item["experiment"] == "nmnist"
            and item["layer"] == layer
            and item["condition"] == "alpha_1_homogenized_frobenius_matched"
            and item["signal"] == "delta"
            and item["component"] == "between_sample"
        )
        lines.append(
            f"| {layer} | {fmt(total['r95_mean'],1)}±{fmt(total['r95_std'],1)} | "
            f"{within['trace_fraction_of_total_mean']:.2e} | "
            f"{fmt(between['trace_fraction_of_total_mean'],6)} |"
        )
    lines.extend(
        [
            "",
            "## Temporal ANN-DFA versus memoryless surrogate",
            "",
            "| Model | Layer | Gate within trace | Delta within trace | Gate r95 | Delta r95 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for experiment, label in (("ann", "Temporal ANN-DFA"), ("stateless", "Memoryless")):
        for layer in ("H1", "H2", "H3"):
            gate = next(
                item
                for item in summary
                if item["experiment"] == experiment
                and item["layer"] == layer
                and item["condition"] == "actual"
                and item["signal"] == "gate"
                and item["component"] == "within_time"
            )
            delta = next(
                item
                for item in summary
                if item["experiment"] == experiment
                and item["layer"] == layer
                and item["condition"] == "actual"
                and item["signal"] == "delta"
                and item["component"] == "within_time"
            )
            lines.append(
                f"| {label} | {layer} | "
                f"{fmt(gate['trace_fraction_of_total_mean'],3)}±{fmt(gate['trace_fraction_of_total_std'],3)} | "
                f"{fmt(delta['trace_fraction_of_total_mean'],3)}±{fmt(delta['trace_fraction_of_total_std'],3)} | "
                f"{fmt(gate['r95_mean'],1)}±{fmt(gate['r95_std'],1)} | "
                f"{fmt(delta['r95_mean'],1)}±{fmt(delta['r95_std'],1)} |"
            )
    lines.extend(
        [
            "",
            "## DVS-Gesture projected local error (three seeds)",
            "",
            "| Layer | Within trace fraction | Between trace fraction | Delta r95 | Carrier q r95 | q active coordinates |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for layer in ("H1", "H2", "H3"):
        within = next(
            item
            for item in summary
            if item["experiment"] == "dvs"
            and item["layer"] == layer
            and item["signal"] == "delta"
            and item["component"] == "within_time"
        )
        between = next(
            item
            for item in summary
            if item["experiment"] == "dvs"
            and item["layer"] == layer
            and item["signal"] == "delta"
            and item["component"] == "between_sample"
        )
        total = next(
            item
            for item in summary
            if item["experiment"] == "dvs"
            and item["layer"] == layer
            and item["signal"] == "delta"
            and item["component"] == "total"
        )
        carrier_rows = [
            row
            for row in ALL_ROWS
            if row["experiment"] == "dvs"
            and row["layer"] == layer
            and row["signal"] == "q_carrier"
        ]
        _, q_r95, q_r95_sd = mean_sd(float(row["r95"]) for row in carrier_rows)
        _, q_active, q_active_sd = mean_sd(
            float(row["q_active_coordinate_fraction"]) for row in carrier_rows
        )
        lines.append(
            f"| {layer} | {fmt(within['trace_fraction_of_total_mean'],3)}±{fmt(within['trace_fraction_of_total_std'],3)} | "
            f"{fmt(between['trace_fraction_of_total_mean'],3)}±{fmt(between['trace_fraction_of_total_std'],3)} | "
            f"{fmt(total['r95_mean'],1)}±{fmt(total['r95_std'],1)} | "
            f"{fmt(q_r95,1)}±{fmt(q_r95_sd,1)} | {fmt(q_active,3)}±{fmt(q_active_sd,3)} |"
        )
    lines.extend(["", "## Paired accuracy controls", ""])
    for row in accuracy_summary:
        lines.append(
            f"- {row['intervention']}: {row['positive_drop_count']}/{row['seed_count']} "
            f"seeds decreased; paired drop "
            f"{row['mean_accuracy_drop_percentage_points']:.3f}±"
            f"{row['std_accuracy_drop_percentage_points']:.3f} percentage points."
        )
    lines.extend(
        [
            "",
            "## Provenance",
            "",
            "- Fixed archived diagnostic-probe indices and basis_eval split.",
            "- Completed epoch-100 checkpoints only.",
            "- Existing PCA/spectral centering and float32-scatter/float64-merge implementation.",
            "- DVS H1/H2 are spatially averaged channel-space diagnostics; no raw high-dimensional covariance was fabricated.",
            "- Full singular spectra are saved as .npy files referenced by each source-data row.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    global ALL_ROWS
    paths: list[Path] = []
    for experiment, seeds in EXPECTED.items():
        for seed in seeds:
            path = HERE / "per_seed" / f"{experiment}_seed_{seed}.csv"
            complete = path.with_suffix(".complete.json")
            if not path.exists() or not complete.exists():
                raise FileNotFoundError(f"missing completed replay: {path}")
            record = json.loads(complete.read_text(encoding="utf8"))
            if record["status"] != "complete":
                raise RuntimeError(f"incomplete replay: {complete}")
            if not all(item["passed"] for item in record["frozen_r95_parity"]):
                raise RuntimeError(f"frozen r95 mismatch: {complete}")
            paths.append(path)
    ALL_ROWS = [row for path in paths for row in read_csv(path)]
    for row in ALL_ROWS:
        spectrum = Path(row.get("spectrum_file", ""))
        if not spectrum:
            raise RuntimeError("missing spectrum provenance")
        if not spectrum.is_absolute():
            spectrum = V9.parents[1] / spectrum
        if not spectrum.exists():
            raise FileNotFoundError(spectrum)
    max_identity = max(
        float(row["identity_relative_frobenius"])
        for row in ALL_ROWS
        if row.get("identity_relative_frobenius", "") not in ("", None)
    )
    if max_identity > 5e-6:
        raise RuntimeError(f"decomposition identity failure: {max_identity}")
    summary = aggregate(ALL_ROWS)
    paired, paired_summary = paired_accuracy()
    write_csv(HERE / "covariance_decomposition_per_seed.csv", ALL_ROWS)
    write_csv(HERE / "covariance_decomposition_summary.csv", summary)
    write_csv(HERE / "paired_accuracy_drops.csv", paired)
    write_csv(HERE / "paired_accuracy_drop_summary.csv", paired_summary)
    report = build_report(summary, paired_summary, max_identity)
    (HERE / "COVARIANCE_DECOMPOSITION_AUDIT.md").write_text(
        report, encoding="utf8"
    )
    completion = {
        "status": "complete",
        "per_seed_jobs": len(paths),
        "per_seed_rows": len(ALL_ROWS),
        "summary_rows": len(summary),
        "all_frozen_r95_exact": True,
        "maximum_identity_relative_frobenius": max_identity,
        "training_performed": False,
        "frozen_v9_modified": False,
    }
    (HERE / "complete.json").write_text(
        json.dumps(completion, indent=2) + "\n", encoding="utf8"
    )
    print(json.dumps(completion))


ALL_ROWS: list[dict[str, str]] = []


if __name__ == "__main__":
    main()
