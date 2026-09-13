#!/usr/bin/env bash
# Check source-level DFA gradient parity, then replay one canonical checkpoint.
# No optimizer step or training is performed.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
python_bin="${PYTHON_BIN:-python}"
output_dir="${SMOKE_OUTPUT_DIR:-${repo_root}/reproduce/_outputs/smoke}"
source_only=0

if [[ "${1:-}" == "--source-only" ]]; then
  source_only=1
  shift
fi
if [[ "$#" -ne 0 ]]; then
  echo "usage: $0 [--source-only]" >&2
  exit 2
fi

mkdir -p "${output_dir}"
"${python_bin}" "${repo_root}/diagnostics/local_gradient_parity.py" \
  --output "${output_dir}/local_gradient_parity.json" --force

if [[ "${source_only}" -eq 1 ]]; then
  echo '{"status":"passed","scope":"source_only","checkpoint_replay":"not_requested"}'
  exit 0
fi

checkpoint="${CHECKPOINT_PATH:-${repo_root}/metadata/checkpoints/nmnist_snn_dfa_seed_20260830_epoch_100_release.pt}"
probe="${PROBE_CSV:-${repo_root}/metadata/splits/nmnist_diagnostic_probe.csv}"
data_root="${DATA_ROOT:-${repo_root}/data}"
device="${SMOKE_DEVICE:-cpu}"

# Frozen references come from the exact per-seed spectral/covariance tables:
# results/frozen_sources/nmnist/spectral_metrics_per_seed.csv
# results/frozen_sources/covariance/covariance_decomposition_per_seed.csv
expected_sha256="$("${python_bin}" -c 'import csv,sys; rows=list(csv.DictReader(open(sys.argv[1], encoding="utf-8"))); matches=[r["bundled_sha256"] for r in rows if r["availability"].startswith("bundled_")]; assert len(matches)==1; print(matches[0])' "${repo_root}/metadata/checkpoints/checkpoint_manifest.csv")"
expected_r95="546"
expected_within_fraction="0.8928910852694444"

set +e
"${python_bin}" "${repo_root}/diagnostics/checkpoint_smoke.py" \
  --checkpoint "${checkpoint}" \
  --expected-sha256 "${expected_sha256}" \
  --probe "${probe}" \
  --data-root "${data_root}" \
  --seed 20260830 \
  --layer H1 \
  --device "${device}" \
  --expected-r95 "${expected_r95}" \
  --expected-within-fraction "${expected_within_fraction}" \
  --output "${output_dir}/checkpoint_smoke.json" \
  --force
checkpoint_status=$?
set -e

if [[ "${checkpoint_status}" -eq 3 ]]; then
  echo "Checkpoint smoke was skipped; see ${output_dir}/checkpoint_smoke.json" >&2
elif [[ "${checkpoint_status}" -ne 0 ]]; then
  echo "Checkpoint smoke failed; see ${output_dir}/checkpoint_smoke.json" >&2
fi
exit "${checkpoint_status}"
