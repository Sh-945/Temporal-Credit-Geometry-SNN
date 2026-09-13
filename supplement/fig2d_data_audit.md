# Fig.2D Data Audit

## Verdict

**C for the current canonical independently trained DFA-vs-BPTT comparison.**

The archive retains the real per-seed overlap and principal-angle summaries but
not the matching PCA basis tensors or the underlying teaching-signal matrices.
It therefore cannot support a genuine fixed top-32 `32 x 32` heatmap.

## Evidence

- Canonical summary: `results/frozen_sources/nmnist/fig2_subspace_overlap.csv`
  (18 rows), copied byte-for-byte from the frozen paper figure-source
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

| layer | per-seed adaptive k | normalized overlap mean +/- SD | mean principal angle (deg) +/- SD |
|---|---|---:|---:|
| H1 | 23/4/8 | 0.008841286969752301 +/- 0.0072740609474017046 | 86.14981306057469 +/- 1.6569813753027975 |
| H2 | 17/4/9 | 0.0090612307241536 +/- 0.005997826690670106 | 85.6242351840636 +/- 1.7230764504746692 |
| H3 | 12/3/5 | 0.009884033428227698 +/- 0.00926420155381689 | 85.85851550781813 +/- 2.466379083774046 |

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
