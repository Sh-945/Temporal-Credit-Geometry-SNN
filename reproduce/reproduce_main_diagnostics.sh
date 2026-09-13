#!/usr/bin/env bash
# Reviewer entry point: integrity check, non-training smoke replay, and figures.
set -uo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
python_bin="${PYTHON_BIN:-python}"

if [[ -f "${repo_root}/ARTIFACT_MANIFEST_SHA256.csv" ]]; then
  "${python_bin}" "${repo_root}/verify_artifact.py"
  integrity_status=$?
else
  echo "Artifact manifest is missing." >&2
  integrity_status=1
fi
if [[ "${integrity_status}" -ne 0 ]]; then
  exit "${integrity_status}"
fi

"${script_dir}/smoke_test.sh" "$@"
smoke_status=$?

"${script_dir}/make_figures.sh"
figure_status=$?

if [[ "${figure_status}" -ne 0 ]]; then
  exit "${figure_status}"
fi
if [[ "${smoke_status}" -eq 3 ]]; then
  echo "Frozen-table figures were regenerated, but checkpoint replay was skipped." >&2
  exit 3
fi
exit "${smoke_status}"

