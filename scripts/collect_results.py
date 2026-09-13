"""Collect only complete, self-describing experiment summaries."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FIELDS = [
    "dataset",
    "model",
    "neuron",
    "time_steps",
    "method",
    "rank",
    "energy_threshold",
    "loss_weights",
    "seed",
    "best_accuracy",
    "best_epoch",
    "config",
    "checkpoint",
    "log",
    "source",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="outputs")
    parser.add_argument("--output", default="results/collected_results.csv")
    args = parser.parse_args()
    root = Path(args.root)
    rows = []
    for summary_path in sorted(root.rglob("summary.json")) if root.exists() else []:
        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        if summary.get("status") != "complete":
            continue
        row = {field: summary.get(field) for field in FIELDS}
        row["source"] = summary_path.as_posix()
        for field in ("rank", "loss_weights"):
            row[field] = json.dumps(row[field], ensure_ascii=False, sort_keys=True)
        rows.append(row)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"collected {len(rows)} complete runs -> {output}")


if __name__ == "__main__":
    main()
