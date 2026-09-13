"""Build the Fig.2D evidence audit from the frozen canonical summary CSV."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/frozen_sources/nmnist/fig2_subspace_overlap.csv"
OUTPUT = ROOT / "supplement/fig2d_data_audit.md"


def main() -> None:
    with SOURCE.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    selected = [
        row
        for row in rows
        if row["epoch"] == "100" and row["temporal_mode"] == "timestep"
    ]
    layers = ("hidden_1", "hidden_2", "hidden_3")
    table = [
        "| layer | per-seed adaptive k | normalized overlap mean +/- SD | mean principal angle (deg) +/- SD |",
        "|---|---|---:|---:|",
    ]
    for layer in layers:
        part = [row for row in selected if row["layer"] == layer]
        if len(part) != 3:
            raise RuntimeError(f"Expected three canonical rows for {layer}, got {len(part)}")
        if len({row["overlap_mean"] for row in part}) != 1:
            raise RuntimeError(f"Inconsistent overlap summary for {layer}")
        if len({row["mean_principal_angle_mean"] for row in part}) != 1:
            raise RuntimeError(f"Inconsistent angle summary for {layer}")
        ks = "/".join(row["k"] for row in sorted(part, key=lambda item: item["seed"]))
        row = part[0]
        table.append(
            f"| {layer.replace('hidden_', 'H')} | {ks} | "
            f"{row['overlap_mean']} +/- {row['overlap_std']} | "
            f"{row['mean_principal_angle_mean']} +/- {row['mean_principal_angle_std']} |"
        )

    text = f"""# Fig.2D Data Audit

## Verdict

**C for the current canonical independently trained DFA-vs-BPTT comparison.**

The archive retains the real per-seed overlap and principal-angle summaries but
not the matching PCA basis tensors or the underlying teaching-signal matrices.
It therefore cannot support a genuine fixed top-32 `32 x 32` heatmap.

## Evidence

- Canonical summary: `results/frozen_sources/nmnist/fig2_subspace_overlap.csv`
  ({len(rows)} rows), copied byte-for-byte from the frozen paper figure-source
  table.
- Generator: `scripts/diagnostics/paper1_stageA_aggregate.py:154-204`. It loaded
  temporary `bases.pt` files from separately trained DFA and BPTT diagnostic
  directories. Those matching temporary files were absent from the final
  archive during the read-only audit.
- Formula: `src/analysis/feedback_expansion/core.py:131-150`.

The canonical comparison uses

```text
k = min(r95_DFA, r95_BPTT)
A = Q_DFA[:, :k]
B = Q_BPTT[:, :k]
s = svdvals(A.T @ B)
normalized_overlap = mean(s**2)
principal_angles_deg = degrees(acos(s))
```

It is an adaptive-rank comparison, not a fixed top-32 comparison.

## Numerical consistency

The values below are read directly from the canonical CSV; SD is the archived
three-seed sample SD.

{chr(10).join(table)}

The approximate values previously described as `0.009/0.008/0.009` and
`85.8/86.1/86.4 deg` do not exactly match this frozen table. The table above is
the canonical evidence and must not be silently changed to fit those rounded
descriptions.

## Why the 32 x 32 heatmap cannot be reconstructed

The desired matrix is `M = abs(Q_DFA[:, :32].T @ Q_BPTT[:, :32])`. A summary
overlap is one scalar, and the principal-angle cosines are only the singular
values of the unobserved cross-basis matrix. They do not determine its 1,024
entries. Neither summary is sufficient to invert `M`.

A different Experiment 02 archive does contain 800 x 800 DFA and BPTT basis
tensors, but its BPTT signal is counterfactual BPTT evaluated on the same
DFA-trained model. It is not the independently trained BPTT protocol used by
the canonical table, and its fixed-32 values are numerically incompatible.
Those tensors must not be substituted.

## Safe paper display

Show the real H1/H2/H3 normalized overlap and mean principal angle, preferably
with per-seed points or mean +/- SD, and label `k=min(r95_DFA,r95_BPTT)`. Do not
show a reconstructed or simulated 32 x 32 heatmap.
"""
    OUTPUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
