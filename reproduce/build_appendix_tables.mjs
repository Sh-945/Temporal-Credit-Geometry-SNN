import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { Workbook, SpreadsheetFile } from "@oai/artifact-tool";

const here = path.dirname(fileURLToPath(import.meta.url));
const repo = path.resolve(here, "..");
const resultsDir = path.join(repo, "results");
const supplementDir = path.join(repo, "supplement");
const qaDir = path.join(supplementDir, ".qa_table_previews");

const sourcePaths = {
  spectral: "results/frozen_sources/nmnist/spectral_metrics_per_seed.csv",
  samples: "results/frozen_sources/nmnist/sample_metrics.csv",
  gates: "results/frozen_sources/nmnist/gate_per_seed.csv",
  runs: "results/frozen_sources/nmnist/all_runs.csv",
  homogenization: "results/frozen_sources/mechanisms/temporal_homogenization_per_seed.csv",
  magnitude: "results/frozen_sources/mechanisms/magnitude_sparsification_per_seed.csv",
  random: "results/frozen_sources/mechanisms/random_mask_control_per_seed.csv",
  marginal: "results/frozen_sources/mechanisms/marginal_distribution_shuffle_per_seed.csv",
  memoryless: "results/frozen_sources/mechanisms/stateless_surrogate_per_seed.csv",
  dvsGates: "results/frozen_sources/dvs/gate_per_seed.csv",
  dvsTests: "results/frozen_sources/dvs/test_results.csv",
  dvsCausal: "results/frozen_sources/dvs/dvs_h3_causal_results.csv",
  covariance: "results/frozen_sources/covariance/covariance_decomposition_per_seed.csv",
  width: "results/frozen_sources/robustness/width_robustness_canonical.csv",
};

function key(...parts) {
  return parts.map((x) => String(x ?? "")).join("\u241f");
}

async function loadRows(relativePath) {
  const absolutePath = path.join(repo, ...relativePath.split("/"));
  const text = await fs.readFile(absolutePath, "utf8");
  const imported = await Workbook.fromCSV(text, { sheetName: "source" });
  const values = imported.worksheets.getItem("source").getUsedRange(true).values;
  const headers = values[0].map((v) => String(v ?? "").replace(/^\uFEFF/, ""));
  return values.slice(1).filter((row) => row.some((v) => v !== null && v !== "")).map((row, index) => {
    const out = { __source_row: index + 2 };
    headers.forEach((header, column) => { out[header] = row[column] ?? ""; });
    return out;
  });
}

function mapBy(rows, fields) {
  return new Map(rows.map((row) => [key(...fields.map((field) => row[field])), row]));
}

function pick(row, columns) {
  const out = {};
  for (const column of columns) out[column] = row?.[column] ?? "";
  return out;
}

function sourceRef(relativePath, row) {
  return `${relativePath}#row=${row.__source_row}`;
}

function csvEscape(value) {
  if (value === null || value === undefined) return "";
  const text = String(value);
  return /[",\r\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

function toCsv(rows) {
  if (!rows.length) throw new Error("Refusing to emit an empty table");
  const headers = Object.keys(rows[0]);
  const lines = [headers.map(csvEscape).join(",")];
  for (const row of rows) lines.push(headers.map((header) => csvEscape(row[header])).join(","));
  return `${lines.join("\n")}\n`;
}

function workbookValue(value) {
  if (value === null || value === undefined || value === "") return null;
  if (typeof value !== "string") return value;
  if (value === "True") return true;
  if (value === "False") return false;
  if (/^-?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$/.test(value)) {
    const number = Number(value);
    if (Number.isFinite(number)) return number;
  }
  return value.startsWith("=") ? `'${value}` : value;
}

function matrixFor(rows) {
  const headers = Object.keys(rows[0]);
  return [headers, ...rows.map((row) => headers.map((header) => workbookValue(row[header])))];
}

function selectedAccuracyMaps(runs) {
  const finalRows = runs.filter((row) => row.checkpoint_rule === "completed_epoch_100");
  const selectedRows = runs.filter((row) => row.checkpoint_rule === "validation_accuracy_max_tie_lower_loss");
  return {
    finalMap: mapBy(finalRows, ["method", "seed"]),
    selectedMap: mapBy(selectedRows, ["method", "seed"]),
  };
}

function buildNmnistMain(spectral, runs) {
  const core = new Set(["SNN-DFA", "matched SNN-BPTT", "temporal ANN-DFA"]);
  const { finalMap, selectedMap } = selectedAccuracyMaps(runs);
  return spectral.filter((row) =>
    core.has(row.method) && row.probe_split === "basis_eval" &&
    row.temporal_mode === "timestep" && row.checkpoint_rule === "completed_epoch_100"
  ).map((row) => {
    const finalRun = finalMap.get(key(row.method, row.seed));
    const selectedRun = selectedMap.get(key(row.method, row.seed));
    return {
      dataset: row.dataset,
      method: row.method,
      seed: row.seed,
      layer: row.layer,
      width: row.width,
      checkpoint_rule_for_geometry: row.checkpoint_rule,
      probe_split: row.probe_split,
      temporal_mode: row.temporal_mode,
      epoch100_test_accuracy: finalRun?.test_accuracy ?? "",
      epoch100_test_loss: finalRun?.test_loss ?? "",
      selected_test_accuracy: selectedRun?.test_accuracy ?? "",
      selected_test_loss: selectedRun?.test_loss ?? "",
      best_validation_epoch: selectedRun?.best_validation_epoch ?? "",
      r50: row.r50,
      r80: row.r80,
      r90: row.r90,
      r95: row.r95,
      r99: row.r99,
      stable_rank: row.stable_rank,
      spectral_entropy: row.spectral_entropy,
      entropy_effective_rank: row.entropy_effective_rank,
      stable_rank_over_D: row.stable_rank_over_D,
      entropy_rank_over_D: row.entropy_rank_over_D,
      checkpoint_sha256: row.checkpoint_sha256,
      source_file_geometry: sourceRef(sourcePaths.spectral, row),
      source_file_accuracy_epoch100: finalRun ? sourceRef(sourcePaths.runs, finalRun) : "",
      source_file_accuracy_selected: selectedRun ? sourceRef(sourcePaths.runs, selectedRun) : "",
    };
  });
}

function canonicalSpectralRows(spectral) {
  return spectral.filter((row) => row.probe_split === "basis_eval" &&
    row.temporal_mode === "timestep" && row.checkpoint_rule === "completed_epoch_100");
}

function buildCutoffs(spectral) {
  return canonicalSpectralRows(spectral).map((row) => ({
    ...pick(row, ["dataset", "method", "seed", "layer", "width", "probe_split", "temporal_mode", "checkpoint_rule", "r50", "r80", "r90", "r95", "r99", "checkpoint_sha256"]),
    source_file: sourceRef(sourcePaths.spectral, row),
  }));
}

function buildThresholdFree(spectral) {
  return canonicalSpectralRows(spectral).map((row) => ({
    ...pick(row, ["dataset", "method", "seed", "layer", "width", "probe_split", "temporal_mode", "checkpoint_rule", "stable_rank", "spectral_entropy", "entropy_effective_rank", "stable_rank_over_D", "entropy_rank_over_D", "checkpoint_sha256"]),
    source_file: sourceRef(sourcePaths.spectral, row),
  }));
}

function buildHomogenization(rows) {
  const columns = ["dataset", "source_model", "seed", "layer", "checkpoint", "checkpoint_sha256", "probe_split", "N", "T", "D", "coordinate_system", "offline_only", "family", "condition", "alpha", "scale_mode", "mean_abs_g", "rms_g", "l2_norm_all_T_N_D", "mean_per_sample_timestep_l2", "exact_zero_fraction", "near_zero_fraction", "temporal_gate_cosine", "adjacent_timestep_cosine", "norm_cv_over_time", "gate_temporal_variance", "r50", "r80", "r90", "r95", "r99", "stable_rank", "spectral_entropy", "entropy_effective_rank", "normalized_entropy_effective_rank"];
  return rows.map((row) => ({ ...pick(row, columns), source_file: sourceRef(sourcePaths.homogenization, row) }));
}

function offlineControlRows(rows, relativePath) {
  const columns = ["dataset", "source_model", "seed", "layer", "probe_split", "N", "T", "D", "coordinate_system", "offline_only", "family", "condition", "alpha", "scale_mode", "mask_seed", "target_zero_fraction", "achieved_zero_fraction", "mask_replicate", "marginal_preservation_check", "r50", "r80", "r90", "r95", "r99", "stable_rank", "spectral_entropy", "entropy_effective_rank", "normalized_entropy_effective_rank", "mean_abs_g", "rms_g", "exact_zero_fraction", "near_zero_fraction", "temporal_gate_cosine", "adjacent_timestep_cosine", "checkpoint_sha256"];
  return rows.map((row) => ({
    control_scope: "offline_post_hoc",
    ...pick(row, columns),
    method: "",
    epoch100_test_accuracy: "",
    selected_test_accuracy: "",
    source_file: sourceRef(relativePath, row),
  }));
}

function buildControls(spectral, gates, runs, magnitude, randomRows, marginal) {
  const output = [
    ...offlineControlRows(magnitude, sourcePaths.magnitude),
    ...offlineControlRows(randomRows, sourcePaths.random),
    ...offlineControlRows(marginal, sourcePaths.marginal),
  ];
  const { finalMap, selectedMap } = selectedAccuracyMaps(runs);
  const gateMap = mapBy(gates.filter((row) => row.probe_split === "basis_eval"), ["method", "seed", "layer"]);
  for (const row of canonicalSpectralRows(spectral).filter((item) => ["MeanGate", "ShuffledGate"].includes(item.method))) {
    const gate = gateMap.get(key(row.method, row.seed, row.layer));
    const finalRun = finalMap.get(key(row.method, row.seed));
    const selectedRun = selectedMap.get(key(row.method, row.seed));
    output.push({
      control_scope: "trained_control",
      dataset: row.dataset,
      source_model: row.method,
      seed: row.seed,
      layer: row.layer,
      probe_split: row.probe_split,
      N: "",
      T: "",
      D: row.width,
      coordinate_system: "neuron",
      offline_only: false,
      family: "trained_gate_control",
      condition: row.method,
      alpha: "",
      scale_mode: "",
      mask_seed: "",
      target_zero_fraction: "",
      achieved_zero_fraction: "",
      mask_replicate: "",
      marginal_preservation_check: "",
      r50: row.r50,
      r80: row.r80,
      r90: row.r90,
      r95: row.r95,
      r99: row.r99,
      stable_rank: row.stable_rank,
      spectral_entropy: row.spectral_entropy,
      entropy_effective_rank: row.entropy_effective_rank,
      normalized_entropy_effective_rank: row.entropy_rank_over_D,
      mean_abs_g: gate?.mean_abs_g ?? "",
      rms_g: gate?.rms_g ?? "",
      exact_zero_fraction: gate?.exact_zero_fraction ?? "",
      near_zero_fraction: gate?.near_zero_fraction ?? "",
      temporal_gate_cosine: gate?.temporal_gate_cosine ?? "",
      adjacent_timestep_cosine: gate?.adjacent_timestep_cosine ?? "",
      checkpoint_sha256: row.checkpoint_sha256,
      method: row.method,
      epoch100_test_accuracy: finalRun?.test_accuracy ?? "",
      selected_test_accuracy: selectedRun?.test_accuracy ?? "",
      source_file: [sourceRef(sourcePaths.spectral, row), gate ? sourceRef(sourcePaths.gates, gate) : "", finalRun ? sourceRef(sourcePaths.runs, finalRun) : "", selectedRun ? sourceRef(sourcePaths.runs, selectedRun) : ""].filter(Boolean).join(";"),
    });
  }
  return output;
}

function buildMemoryless(rows) {
  const columns = ["experiment_id", "dataset", "method", "seed", "layer", "checkpoint", "checkpoint_sha256", "model_sha256", "probe_split", "N", "T", "D", "best_validation_epoch", "best_validation_test_accuracy", "final_test_accuracy", "q_l2", "r50", "r80", "r90", "r95", "r99", "stable_rank", "spectral_entropy", "entropy_effective_rank", "normalized_entropy_effective_rank", "mean_abs_g", "rms_g", "exact_zero_fraction", "near_zero_fraction", "temporal_gate_cosine", "adjacent_timestep_cosine", "norm_cv_over_time", "gate_temporal_variance", "replaced_cells", "production_source_modified"];
  return rows.map((row) => ({ ...pick(row, columns), source_file: sourceRef(sourcePaths.memoryless, row) }));
}

function buildDvs(gateRows, testRows, causalRows, covarianceRows) {
  const actualCausal = causalRows.filter((row) => row.Method === "Actual");
  const causalMap = mapBy(actualCausal, ["Seed"]);
  const testsMap = mapBy(testRows, ["method", "seed", "checkpoint"]);
  const gateMap = mapBy(gateRows, ["seed", "layer", "gate_space"]);
  const dvsCov = covarianceRows.filter((row) => row.experiment === "dvs" && row.condition === "actual");
  const covMap = mapBy(dvsCov, ["seed", "layer", "signal", "component"]);
  const layers = ["H1", "H2", "H3"];
  const out = [];
  for (const causal of actualCausal) {
    for (const layer of layers) {
      const seed = causal.Seed;
      const projected = gateMap.get(key(seed, layer, "canonical_spatial_mean_channels"));
      const raw = gateMap.get(key(seed, layer, "raw_neuron_and_spatial_features"));
      const testBest = testsMap.get(key("dvs_dfa", seed, "best_validation"));
      const testFinal = testsMap.get(key("dvs_dfa", seed, "final"));
      const deltaTotal = covMap.get(key(seed, layer, "delta", "total"));
      const deltaWithin = covMap.get(key(seed, layer, "delta", "within_time"));
      const deltaBetween = covMap.get(key(seed, layer, "delta", "between_sample"));
      const gateTotal = covMap.get(key(seed, layer, "gate", "total"));
      const gateWithin = covMap.get(key(seed, layer, "gate", "within_time"));
      const gateBetween = covMap.get(key(seed, layer, "gate", "between_sample"));
      const carrier = covMap.get(key(seed, layer, "q_carrier", "carrier_total"));
      out.push({
        dataset: "DVS-Gesture",
        method: "SNN-DFA",
        seed,
        layer,
        final_epoch_test_accuracy: testFinal?.test_accuracy ?? causal["Test accuracy"],
        best_validation_test_accuracy: testBest?.test_accuracy ?? causal["Best-val test accuracy"],
        best_validation_epoch: testBest?.best_validation_epoch ?? causal["Best validation epoch"],
        N: projected?.N ?? raw?.N ?? deltaTotal?.N ?? "",
        T: projected?.T ?? raw?.T ?? deltaTotal?.T ?? "",
        projected_D: projected?.width ?? deltaTotal?.D ?? "",
        raw_capture_shape_last_batch: raw?.source_raw_gate_batch_shape ?? projected?.source_raw_gate_batch_shape ?? "",
        projected_analysis_shape: projected ? `[${projected.T},${projected.N},${projected.width}]` : "",
        raw_gate_cosine: raw?.temporal_gate_cosine ?? "",
        projected_gate_cosine: projected?.temporal_gate_cosine ?? "",
        raw_adjacent_timestep_cosine: raw?.adjacent_timestep_cosine ?? "",
        projected_adjacent_timestep_cosine: projected?.adjacent_timestep_cosine ?? "",
        carrier_q_r95: causal[`${layer} q r95`] ?? carrier?.r95 ?? projected?.q_r95 ?? "",
        post_gate_delta_timestep_r95: causal[`${layer} delta timestep r95`] ?? deltaTotal?.r95 ?? projected?.credit_r95 ?? "",
        post_gate_delta_aggregated_r95: causal[`${layer} delta aggregated r95`] ?? "",
        gate_expansion_ratio: causal[`${layer} GE`] ?? projected?.GE ?? "",
        delta_within_time_trace_fraction: deltaWithin?.trace_fraction_of_total ?? "",
        delta_between_sample_trace_fraction: deltaBetween?.trace_fraction_of_total ?? "",
        gate_total_r95: gateTotal?.r95 ?? "",
        gate_within_time_trace_fraction: gateWithin?.trace_fraction_of_total ?? "",
        gate_between_sample_trace_fraction: gateBetween?.trace_fraction_of_total ?? "",
        carrier_total_r95_covariance_replay: carrier?.r95 ?? "",
        covariance_identity_relative_frobenius: deltaTotal?.identity_relative_frobenius ?? "",
        max_production_delta_relative_error: causal["max production delta relative error"] ?? "",
        checkpoint_sha256: projected?.checkpoint_sha256 ?? deltaTotal?.checkpoint_sha256 ?? "",
        source_file: [
          sourceRef(sourcePaths.dvsCausal, causal),
          projected ? sourceRef(sourcePaths.dvsGates, projected) : "",
          raw ? sourceRef(sourcePaths.dvsGates, raw) : "",
          testBest ? sourceRef(sourcePaths.dvsTests, testBest) : "",
          testFinal ? sourceRef(sourcePaths.dvsTests, testFinal) : "",
          deltaTotal ? sourceRef(sourcePaths.covariance, deltaTotal) : "",
        ].filter(Boolean).join(";"),
      });
    }
  }
  return out;
}

function buildProbeSensitivity(rows) {
  const columns = ["dataset", "method", "seed", "layer", "checkpoint_rule", "width", "probe_split", "N", "T", "observations", "repeat", "subsample_seed", "selected_positions", "r95", "stable_rank", "entropy_effective_rank", "full_sample_repetition", "checkpoint_sha256"];
  return rows.map((row) => ({ ...pick(row, columns), source_file: sourceRef(sourcePaths.samples, row) }));
}

function buildWidth(rows) {
  return rows.map((row) => {
    const { __source_row: _sourceRow, ...published } = row;
    return { ...published, provenance_file: sourceRef(sourcePaths.width, row) };
  });
}

async function main() {
  const loaded = {};
  for (const [name, relativePath] of Object.entries(sourcePaths)) loaded[name] = await loadRows(relativePath);

  const tables = [
    ["nmnist_main", "nmnist_main_per_seed.csv", buildNmnistMain(loaded.spectral, loaded.runs)],
    ["cutoff_metrics", "cutoff_metrics_per_seed.csv", buildCutoffs(loaded.spectral)],
    ["threshold_free", "threshold_free_metrics_per_seed.csv", buildThresholdFree(loaded.spectral)],
    ["homogenization", "homogenization_per_seed_layer_alpha.csv", buildHomogenization(loaded.homogenization)],
    ["controls", "controls_per_seed.csv", buildControls(loaded.spectral, loaded.gates, loaded.runs, loaded.magnitude, loaded.random, loaded.marginal)],
    ["memoryless", "memoryless_per_seed.csv", buildMemoryless(loaded.memoryless)],
    ["dvs", "dvs_per_seed.csv", buildDvs(loaded.dvsGates, loaded.dvsTests, loaded.dvsCausal, loaded.covariance)],
    ["probe_sensitivity", "probe_size_sensitivity.csv", buildProbeSensitivity(loaded.samples)],
    ["width_robustness", "width_robustness.csv", buildWidth(loaded.width)],
  ];

  await fs.mkdir(resultsDir, { recursive: true });
  await fs.mkdir(supplementDir, { recursive: true });
  await fs.mkdir(qaDir, { recursive: true });

  const workbook = Workbook.create();
  for (const [sheetName, fileName, rows] of tables) {
    if (!rows.length) throw new Error(`${sheetName} unexpectedly has zero rows`);
    const csv = toCsv(rows);
    await fs.writeFile(path.join(resultsDir, fileName), csv, "utf8");

    const matrix = matrixFor(rows);
    const sheet = workbook.worksheets.add(sheetName);
    sheet.getRange("A1").write(matrix);
    sheet.freezePanes.freezeRows(1);
    sheet.showGridLines = false;
    const used = sheet.getUsedRange(true);
    used.format.font = { name: "Aptos", size: 9, color: "#1F2937" };
    const header = sheet.getRangeByIndexes(0, 0, 1, matrix[0].length);
    header.format.fill = "#17365D";
    header.format.font = { name: "Aptos", size: 9, bold: true, color: "#FFFFFF" };
    header.format.wrapText = true;
    header.format.rowHeight = 34;
    for (let column = 0; column < matrix[0].length; column += 1) {
      const observed = matrix.slice(0, Math.min(matrix.length, 101)).map((row) => String(row[column] ?? "").length);
      const width = Math.max(10, Math.min(30, Math.max(...observed) + 1));
      sheet.getRangeByIndexes(0, column, matrix.length, 1).format.columnWidth = width;
    }
  }

  workbook.recalculate();
  const overview = await workbook.inspect({ kind: "sheet", include: "id,name", maxChars: 6000 });
  await fs.writeFile(path.join(qaDir, "workbook_sheet_inspection.txt"), overview.ndjson ?? String(overview), "utf8");

  for (const [sheetName, , rows] of tables) {
    const columns = Object.keys(rows[0]).length;
    const previewColumns = Math.min(columns, 12);
    const endColumn = columnName(previewColumns);
    const endRow = Math.min(rows.length + 1, 16);
    const inspection = await workbook.inspect({ kind: "region", sheetId: sheetName, range: `A1:${endColumn}${endRow}`, maxChars: 5000 });
    await fs.writeFile(path.join(qaDir, `${sheetName}_inspection.txt`), inspection.ndjson ?? String(inspection), "utf8");
    const preview = await workbook.render({ sheetName, range: `A1:${endColumn}${endRow}`, scale: 1, format: "png" });
    await fs.writeFile(path.join(qaDir, `${sheetName}.png`), new Uint8Array(await preview.arrayBuffer()));
  }

  const xlsx = await SpreadsheetFile.exportXlsx(workbook);
  await xlsx.save(path.join(supplementDir, "appendix_tables.xlsx"));

  const summary = tables.map(([sheet, file, rows]) => ({ sheet, file, rows: rows.length, columns: Object.keys(rows[0]).length }));
  await fs.writeFile(path.join(qaDir, "table_build_summary.json"), `${JSON.stringify(summary, null, 2)}\n`, "utf8");
  process.stdout.write(`${JSON.stringify(summary, null, 2)}\n`);
}

function columnName(oneBasedColumn) {
  let value = oneBasedColumn;
  let out = "";
  while (value > 0) {
    value -= 1;
    out = String.fromCharCode(65 + (value % 26)) + out;
    value = Math.floor(value / 26);
  }
  return out;
}

await main();
