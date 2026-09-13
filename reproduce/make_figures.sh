#!/usr/bin/env bash
# Regenerate paper figures from frozen per-seed CSVs; no checkpoint or training.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
python_bin="${PYTHON_BIN:-python}"
source_root="${FIGURE_SOURCE_ROOT:-${repo_root}}"
output_dir="${FIGURE_OUTPUT_DIR:-${repo_root}/reproduce/_outputs/figures}"

exec "${python_bin}" "${repo_root}/diagnostics/make_figures.py" \
  --source-root "${source_root}" \
  --output-dir "${output_dir}" \
  --force \
  "$@"
