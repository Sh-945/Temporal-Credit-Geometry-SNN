# Reproducibility Audit

## Verdict

The frozen ICASSP 2027 claims are traceable to real per-seed result files, exact
split/probe indices, source code, checkpoint hashes, and sanitized checkpoint
configuration records. No main experiment was retrained and no scientific value
was inferred from manuscript prose. One path-sanitized release copy of a
SHA-256-identified canonical N-MNIST SNN-DFA checkpoint is bundled so the
reviewer smoke test performs a real replay without exposing machine paths.

Audit date: 2026-09-13. The source archive had no recoverable Git commit; the
release therefore uses byte-level source and artifact manifests instead.

| Area | Status |
|---|---|
| A. Training/model/feedback configuration | Confirmed from code and 28 sanitized checkpoint records |
| B. Splits and probes | Exact indices included and regeneration rule confirmed |
| C. DFA local gradient | Confirmed from production code plus two numerical checks |
| D. Spectral diagnostics | Exact definitions and decomposition confirmed |
| E. Homogenization/controls | Preserve/change semantics confirmed from code |
| F. DVS-Gesture | Three-seed values, shapes and cosine definitions recovered |
| G. Numerical tables | Nine per-seed/source tables generated from frozen CSVs |
| H. Public repository | Structured, path-sanitized and hash-manifested |
| I. One-click replay | Scripts provided; canonical checkpoint smoke passed |

## Evidence hierarchy

1. `results/frozen_sources/` contains byte-preserved source CSVs.
2. `results/*.csv` contains reviewer-facing tables built only from those files.
3. `metadata/` contains exact indices, 28 sanitized checkpoint records, feedback
   tensor hashes, source-code hashes, and smoke outputs.
4. `src/` and `diagnostics/archived/` contain the production and replay logic.
5. `ARTIFACT_MANIFEST_SHA256.csv` covers every published file byte-for-byte.

## A–C. Training, data, and local-gradient implementation

Scope: read-only recovery from the frozen source/result archive. No training or canonical-result regeneration was performed. Code and result references use public repository paths, except entries explicitly labeled `archived path`; those are provenance identifiers for final checkpoint binaries that are not redistributed.

## A. Training, model, and feedback configuration

### A.1 Canonical N-MNIST configuration

The core configuration is identical in the archived final checkpoints and in `configs/nmnist/paper1_experiment03.yaml:4-43`:

| Item | Recovered value | Evidence |
|---|---|---|
| input / architecture | time-major input `[T,B,2,34,34]`; flatten to 2312; spiking FC hidden widths `800,800,800`; spiking 10-class readout | `configs/nmnist/paper1_experiment03.yaml:4-16`; `src/models/factory.py:22-30`; `src/models/fc/network.py:29-72` |
| temporal length | `T=30` | `configs/nmnist/paper1_experiment03.yaml:7` |
| epochs | 100 completed epochs (loop epochs 0--99; `epoch_100.pt` stores `epoch=99`) | `configs/nmnist/paper1_experiment03.yaml:30`; `src/analysis/paper1_geometry/training.py:355-415`; `metadata/all_checkpoint_metadata_sanitized.json` |
| batch size | 128; no `drop_last`, so the last 54,000-sample training minibatch has 112 examples | `configs/nmnist/paper1_experiment03.yaml:8`; `src/analysis/paper1_geometry/data.py:107-120` |
| optimizer | PyTorch `Adam`, not AdamW; `lr=1e-3`, `betas=(0.9,0.999)`, `eps=1e-8`, `weight_decay=0`, `amsgrad=False` | construction: `src/training/engine.py:39-46`; observed fields in all checkpoint records: `metadata/all_checkpoint_metadata_sanitized.json` |
| LR schedule | `StepLR(step_size=60,gamma=0.1)` stepped after each completed epoch; LR is `1e-3` for epochs 1--60 and `1e-4` for epochs 61--100 | `src/training/engine.py:47-56`; scheduler step: `src/analysis/paper1_geometry/training.py:385-386`; observed final scheduler state in `metadata/all_checkpoint_metadata_sanitized.json` |
| gradient clipping | one global L2-norm clip, max norm 1.0, after `backward()` and before Adam; the argument list is model plus feedback parameters | `configs/nmnist/paper1_experiment03.yaml:33`; `src/training/engine.py:161-173`; `torch.nn.utils.clip_grad_norm_` default `norm_type=2` |
| neuron | LIF decay 0.5, threshold 1.0; carried-state update `u_t=0.5*u_(t-1)*(1-s_(t-1))+I_t` | `configs/nmnist/paper1_experiment03.yaml:17-21`; `src/models/neurons/lif.py:29-57` |
| reset | reset-to-zero of the carried membrane contribution after a preceding spike, implemented multiplicatively by `(1-previous_spike)` before adding the current input; there is no separate configurable reset parameter | `src/models/neurons/lif.py:45-56` |
| surrogate derivative | fast-sigmoid derivative `g(u)=1/(1+10*abs(u-threshold))^2`; the forward spike is the hard indicator `1[u-threshold>=0]` | `src/models/neurons/lif.py:9-26`; beta in `configs/nmnist/paper1_experiment03.yaml:21` |
| task loss | `L_task=(1/(2B))*sum_n ||mean_t(o[t,n])-onehot(y_n)||_2^2`, not cross entropy | `src/training/engine.py:67-70` |
| local loss weight | 1.0 | `configs/nmnist/paper1_experiment03.yaml:23-28`; `src/training/engine.py:125-128` |
| checkpoint epochs | initialization plus completed epochs 10, 25, 50, 75, 100 in the formal workflow | config: `configs/nmnist/paper1_experiment03.yaml:36,43`; checkpoint naming: `src/analysis/paper1_geometry/training.py:339-415` |

All linear layers include PyTorch's default bias.  The source contains no call to `torch.nn.init` and does not override `reset_parameters`: construction is by bare `nn.Linear(...)` (`src/models/fc/network.py:13-17,43-47`).  Therefore the archived PyTorch 2.8.0 initialization is the framework default: `kaiming_uniform_(weight,a=sqrt(5))`, algebraically uniform on `[-1/sqrt(fan_in),+1/sqrt(fan_in)]`, and bias uniform on the same interval.  Model construction happens after `seed_everything(seed)` (`src/analysis/soft_spectral/training.py:205-217`).  The archived runtime was Python 3.12.3, PyTorch 2.8.0+cu128, NumPy 2.3.2 and pandas 3.0.5 (`metadata/runtime_environment.txt`). The public `requirements.txt` and `environment.yml` pin PyTorch 2.8.0; the exact CUDA 12.8 build remains host-specific and is recorded in the runtime metadata.

`seed_everything` seeds Python, NumPy, PyTorch CPU and all CUDA devices; it requests deterministic algorithms with `warn_only=True` and disables cuDNN benchmarking (`src/training/seed.py:9-17`).  `warn_only=True` should be disclosed: it requests deterministic kernels but does not turn an unsupported nondeterministic operation into a hard error.

### A.2 Five N-MNIST seeds and checkpoint provenance

The exact final seed set is:

```text
20260830, 20260831, 20260901, 20260908, 20260909
```

Evidence: original set in `diagnostics/archived/v9_replay/replay_v9.py:23`, two preregistered extensions in `diagnostics/archived/v9_replay/train_new_seeds.py:29`, and the 28-record recovered checkpoint audit in `metadata/all_checkpoint_metadata_sanitized.json`.  The five SNN-DFA epoch-100 checkpoint identities are:

| seed | archived path | SHA-256 |
|---:|---|---|
| 20260830 | `results/experiment03_soft_spectral/experiment03_20260831_server/checkpoints/full_dense/seed_20260830/epoch_100.pt` | `b8f2c8c95c5944358405d35f2d38dcb293461fa1e8641e9e6cb29ab98e39a534` |
| 20260831 | `results/experiment03_soft_spectral/experiment03_20260831_server/checkpoints/full_dense/seed_20260831/epoch_100.pt` | `b486ef7efe077a5e11dbfa4148cfa5a8ec90514c3508227a9bf491bde631d58a` |
| 20260901 | `results/experiment03_soft_spectral/experiment03_20260831_server/checkpoints/full_dense/seed_20260901/epoch_100.pt` | `554990a77b2d2e8f6498d66d6182e5b9f7b8a61fd2c63c09d3ded066f44b5563` |
| 20260908 | `results/paper_v9_signal_processing/03_five_seed/checkpoints/dfa_trained/seed_20260908/epoch_100.pt` | `9aad88592135096912c2e9971ced9f74b63d546bdd69208e989abf481efd833e` |
| 20260909 | `results/paper_v9_signal_processing/03_five_seed/checkpoints/dfa_trained/seed_20260909/epoch_100.pt` | `0800c9e1ee7b4cf64c5099e83f65ae2c8adb1bf072ccdaa45be9f1911ae9bad7` |

The first three checkpoints were retained from the compatible `full_dense` experiment and the last two were trained by the v9 extension; all have the same numerical data/model/neuron/optimizer recipe.  `metadata/nmnist_checkpoint_pairing_audit.json` records `matched_recipe=true` and matched paired initialization for the original six DFA/BPTT runs.  `diagnostics/archived/v9_replay/train_new_seeds.py:23-57` records the new-seed dispatch.

### A.3 Dense DFA feedback matrices

For hidden layer `l`, the code creates a distinct tensor

```text
B_l = scale * randn(D_l, C) / sqrt(C)
q_l = e @ B_l.T
```

where `C=10`, `D_l=800`, and `scale=1.0`.  Thus every element is sampled from `Normal(0,1/10)` (standard deviation `1/sqrt(10)`, approximately 0.316227766) in default float32.  Each hidden layer is a separate `DenseFeedback` constructed by a separate `torch.randn` call (`src/methods/feedback_bank.py:42-62`; `src/methods/dfa/feedback.py:9-27`).

Each seed is re-seeded and a new model followed by a new feedback bank is constructed (`src/analysis/soft_spectral/training.py:205-217`).  Within a seed, paired DFA/control/ANN variants reload the same saved feedback state (`src/analysis/soft_spectral/training.py:226-244`; `src/analysis/paper1_geometry/training.py:312-323`), so comparisons are initialization matched.  The 15 layer/seed tensor hashes in `metadata/feedback_matrix_manifest.csv` are all distinct and every measured matrix rank is 10; this directly confirms per-layer and per-seed resampling rather than merely relying on the constructor.

`B_l` is registered as a buffer, not a `Parameter` (`src/methods/dfa/feedback.py:12-16`).  It is therefore checkpointed but absent from optimizer parameter groups and is frozen for the full run.  The projected teaching signal is also explicitly detached before the local proxy (`src/training/engine.py:119-124`).

## B. Data splits and diagnostic probes

### B.1 Runtime data semantics shared by N-MNIST and DVS-Gesture

The clean code consumes already-generated NPZ frame files.  Numeric class directories and filenames are sorted; `dataset_index` means position in that deterministic concatenation (`src/training/data.py:99-111`).  `NPZFrameDataset` loads the `frames` array, converts it to float32, applies no normalization or augmentation, and returns the stored class label (`src/training/data.py:35-65`).  If a file does not already have `T` frames, it is deterministically reindexed with rounded `torch.linspace(0,T_original-1,T)`; neither canonical audit sample shape needed that fallback.  The training loader alone shuffles, using a dedicated `torch.Generator` seeded by the experiment seed (`src/analysis/paper1_geometry/data.py:107-140`).

### B.2 N-MNIST split and exact indices

The official 60,000-example training folder is split once with NumPy `default_rng(20260830)`:

1. `permutation = rng.permutation(60000)`;
2. `validation_count = round(60000*0.10) = 6000`;
3. the first 6,000 indices are sorted into validation;
4. the remaining 54,000 are sorted into formal training;
5. the official 10,000-example test folder remains untouched.

This is implemented by `build_paper1_split` at `src/analysis/paper1_geometry/data.py:58-95`.  The immutable exact list is `metadata/splits/nmnist_train_val_split.csv` (60,000 rows; SHA-256 `1e191fa64750098d0f5c091efdd03ade885795cb18ef041f53c543fae84de85e`).

The 1,024-example training-only probe is a second independent permutation:

```text
probe = default_rng(20260831).permutation(training_indices)[:1024]
basis_fit  = probe[0:512]
basis_eval = probe[512:1024]
```

The random order is deliberately retained and is not sorted before the half split (`src/analysis/paper1_geometry/data.py:66-95`; regression test `tests/test_paper1_experiment03.py:39-44`).  The exact positions, dataset indices, split labels and class labels are in `metadata/splits/nmnist_diagnostic_probe.csv` (1,024 rows; SHA-256 `b3bc4fdf3d3bbe28be7530b656b7db0f2c59fec0eea4c16247418c30bf4794bb`).  The first row is `(probe_position=0,dataset_index=29430,basis_fit,label=4)` and the last is `(1023,52902,basis_eval,label=8)`.  `diagnostics/archived/v9_replay/replay_v9.py:87-90` reconstructs this object and hard-fails if it differs from the archived probe CSV.

### B.3 Probe-size sensitivity rule

Probe-size sensitivity is evaluated only inside the fixed 512-example `basis_eval` half.  For each `N in {128,256,512}` and each repeat `r in {0,...,19}`:

```text
subsample_seed = 2026090800 + r
positions = default_rng(subsample_seed).choice(512,N,replace=False)
positions.sort()
```

The code is `diagnostics/archived/v9_replay/replay_v9.py:139-147`.  `positions` are zero-based positions inside `basis_eval`; the corresponding original dataset index is row `probe_position=512+position` in `diagnostic_probe.csv`.  Every exact selection is retained as JSON in column `selected_positions` of `results/frozen_sources/nmnist/sample_metrics.csv` (2,700 rows = 3 methods x 5 seeds x 3 layers x 3 N values x 20 repeats).  The same selection is reused across method, seed and layer for a given `(N,repeat)`.  At `N=512`, sorting necessarily gives all positions `0,...,511`, so the 20 rows are repeated full-set calculations rather than 20 distinct subsets.

### B.4 DVS-Gesture preprocessing, split, and probe

Confirmed frozen protocol:

| Item | Value | Evidence |
|---|---|---|
| frame tensor | `[30,2,128,128]`, float32 at load time | `configs/dvs_gesture/paper1_experiment03.yaml:4-17`; `metadata/dvs_dataset_audit.json` |
| model | Conv(2->32,3x3,pad1), average pool 2; Conv(32->64,3x3,pad1), average pool 2; flatten `64x32x32`; FC256; 11-class spiking readout | `src/models/conv/network.py:15-82`; DVS config lines 11-17 |
| T / batch | `T=30`, batch size 16 | DVS config lines 7-9 |
| seeds | `20260830,20260831,20260901` | DVS config line 45; `scripts/diagnostics/paper1_stageD_prepare.py:27,74-83` |
| official split | 1,176 train recordings, 288 test recordings | `metadata/dvs_dataset_audit.json` |
| formal train/validation | 1,058/118 from official train, 10% validation, split seed 20260830 | DVS config lines 39-45; `paper1_stageD_prepare.py:45-59` |
| probe | 896 training-only examples, probe seed 20260831; first 448 basis fit and last 448 basis eval | `paper1_stageD_prepare.py:60-71`; DVS config lines 40-44 |

Exact split file: `metadata/splits/dvs_train_val_split.csv` (1,176 rows; SHA-256 `73dbe59fa28f522b0c9126ba086534bbbf16c4c6f33f1a7ebad28689bef19ae7`).  Exact probe file: `metadata/splits/dvs_diagnostic_probe.csv` (896 rows; SHA-256 `0a17e3a8a87714f5f8809b6a0edc0d47cd1adc950289fcce5a5be21360e7db3d`).  Both exactly regenerate with `build_paper1_split(1176,split_seed=20260830,probe_seed=20260831,validation_fraction=.1,probe_size=896)`.

The clean repository does **not** contain the raw-event-to-NPZ converter.  It only proves that training consumed `dvs/frames_number_30_split_by_number/{train,test}/{class}/*.npz` (`src/training/data.py:120-152`; `data/README.md`).  An adjacent historical source tree records the generating API call as SpikingJelly `DVS128Gesture(..., train=True/False, data_type="frame", frames_number=30, split_by="number")`, i.e. 30 event-count-based frames, but the archived requirements do not pin a SpikingJelly version.  Therefore the runtime preprocessing is fully known after NPZ creation, whereas bit-for-bit raw AEDAT-to-NPZ regeneration remains an unresolved packaging item.  The reviewer repository should either (a) publish/checksum the exact NPZ files, or (b) pin and vendor/document the precise SpikingJelly converter version; it must not silently claim that the present clean code generates those NPZs.

## C. DFA local-gradient implementation

### C.1 Stop-gradient boundaries and temporal differentiation

For pointwise DFA (`method.temporal_mode=pointwise`):

1. The provisional full-network forward, hidden inputs, readout input and output error are obtained inside `torch.no_grad()` (`src/training/engine.py:86-95`).
2. The readout is replayed from `readout_input.detach()` so `L_task` updates the readout but not hidden layers (`src/training/engine.py:97-100`).
3. Each hidden layer is replayed independently from `layer_input.detach()` (`src/training/engine.py:101-118`).  Consequently the proxy for layer `l` does not update any earlier layer.
4. `q_l=B_l e` is detached (`src/training/engine.py:119-124`), so the local proxy updates the forward layer, not the error-producing path or `B_l`.
5. The replay uses `detach_temporal=True`.  After every timestep both `membrane` and `previous_spike` are detached (`src/models/neurons/lif.py:38-56`).  The current timestep still differentiates through the surrogate gate, but there is no cross-timestep derivative through either the carried membrane state or the spike/reset path.

This is a no-trace/pointwise local rule, not local BPTT.  In contrast, the matched BPTT path calls `model(...,detach_temporal=False)` and differentiates only the task MSE (`src/training/engine.py:72-84`).

### C.2 Exact local proxy and reduction

Let `e_n = mean_t(o[t,n])-onehot(y_n)`, `q_l[n]=B_l e_n`, pre-spike current `I_l[t,n]`, spike `h_l[t,n]=H(I/state-threshold)`, and local surrogate gate `g_l[t,n]=dh_l/dI_l` under the pointwise temporal detach.

For an FC hidden layer of width `D_l`, production uses:

```text
L_local,l = mean_{t,n,d}( h_l[t,n,d] * q_l[n,d] )
L_local   = sum_l L_local,l
L_total   = L_task + 1.0 * L_local
```

This is exactly `src/training/engine.py:101-128`.  Therefore, defining the diagnostic unscaled post-gate signal

```text
delta_l[t,n,d] = q_l[n,d] * g_l[t,n,d],
```

the production current gradient is

```text
dL_local,l/dI_l = delta_l / M_l,
M_l = T * B_current * D_l.
```

For a full N-MNIST minibatch, `M_l=30*128*800=3,072,000`; on the final 112-example minibatch it is `2,688,000`.  The diagnostic deliberately multiplies the autograd current gradient by `numel()` to recover `delta` (`src/analysis/paper1_geometry/diagnostics.py:137-156`).

For the convolutional DVS hidden layers, channel-level `q_l[B,C]` is broadcast over space **after an additional division by `H*W`**, then the product is reduced with `.mean()` over `[T,B,C,H,W]` (`src/training/engine.py:121-124`).  Thus the per-neuron diagnostic signal is `(q_l/(H*W))*g_l`; the FC formula applies without that spatial factor.  This implementation detail must not be omitted from an exact DVS description.

### C.3 Relation to the actual weight gradient

For an FC layer with detached input `x_l[t,n]`, before global clipping and before Adam:

```text
dL_total/dW_l
  = (1/M_l) * sum_{t,n} delta_l[t,n]^T x_l[t,n]
dL_total/db_l
  = (1/M_l) * sum_{t,n} delta_l[t,n].
```

`L_task` cannot contribute to hidden `W_l` because the readout input is detached, and other layers' proxies cannot contribute because every replay input is detached.  This makes the expression the complete hidden-layer pre-clipping/pre-Adam gradient, not merely one additive term.  `_weight_from_delta` encodes the same contraction as `einsum("tbo,tbi->oi",delta/delta.numel(),layer_input.detach())` (`src/analysis/paper1_geometry/diagnostics.py:39-42`).  For Conv2d, the analogous contraction is the standard cross-correlation of unfolded detached input patches with `(q/(H*W))*g/M`; the implementation obtains it through autograd rather than materializing im2col (`src/analysis/paper1_geometry/diagnostics.py:104-148`).

After this gradient is formed, `clip_grad_norm_(...,1.0)` jointly rescales all trainable gradients if their global L2 norm exceeds one, and Adam consumes the clipped values (`src/training/engine.py:161-173`).  `delta` is therefore explicitly a pre-clipping diagnostic.

Gate-intervention training changes only the injected current derivative: it constructs `desired_delta=production_teaching*applied_gate`, uses `(current*desired_delta).mean()`, and adds a detached correction so the forward scalar remains equal to the production proxy (`src/analysis/paper1_geometry/training.py:128-155`).


### C.4 Numerical checks (no training)

`diagnostics/local_gradient_parity.py` compares the explicit global-reduction
contraction `sum(delta*x^T)/(T*B*D)` with both the diagnostic autograd capture
and `Trainer._local_batch`. The saved source-only check passed with maximum
relative error `1.121796849321369e-07` at tolerance
`1.0e-06` (`metadata/smoke_results/local_gradient_parity.json`).

The stronger checkpoint smoke loaded the bundled epoch-100 seed-20260830
scientific state, verified the sanitized release SHA-256 and all 1,024 probe
indices, replayed the fixed 512-sample `basis_eval` half, and performed no
optimizer step. Its explicit weight-gradient contraction error was
`2.7921030762554063e-07`;
the production `q*g` delta parity error was
`5.3263978117001898e-08`. All checks passed.

## D–F. Spectral diagnostics, controls, and DVS-Gesture

This document is a read-only audit of the frozen ICASSP 2027 result archive. No optimizer step, checkpoint mutation, or scientific-result regeneration was performed for this audit. Line references below point to the repository copies of the source code and frozen CSV files.

## D. Spectral diagnostics

### Diagnostic observations and centering

For a signal tensor with shape `[T,N,D]`, the canonical timestep diagnostic treats every time/sample pair as one row, so the observation matrix is `X in R^(TN x D)`. The time-collapsed diagnostic first computes `Xbar[n]=mean_t X[t,n]`, producing `N` rows. For convolutional H1/H2 signals, the canonical feature map first averages the last two spatial axes and retains channels as features; a 3-D signal is left unchanged. Evidence: `src/analysis/paper1_geometry/metrics.py:17-25` (`feature_signal`) and `src/analysis/paper1_geometry/metrics.py:47-60` (`GeometryAccumulator.update`).

Each split and each temporal mode has its own `OnlineCovariance` accumulator. Thus a reported `basis_eval` rank is centered at the `basis_eval` mean; it does not reuse the `basis_fit` mean. PCA bases used for fit/eval generalization are fitted from the centered `basis_fit` scatter, while held-out energy is evaluated by forming the evaluation scatter about the fit mean. Evidence: `src/analysis/paper1_geometry/metrics.py:37-45,73-114`; `src/analysis/paper1_geometry/diagnostics.py:228-248`; `src/analysis/feedback_expansion/core.py:98-117`.

`OnlineCovariance.m2` is a centered scatter, not a normalized covariance. For rows `x_i`,

```text
mu = (1/M) sum_i x_i
S  = sum_i (x_i-mu)(x_i-mu)^T = X_c^T X_c.
```

Within each input chunk, rows are converted to float32, the chunk mean is subtracted, and `centered.T @ centered` is computed. Chunk summaries are merged on CPU in float64 with the parallel/Welford between-chunk mean correction. The N-MNIST v9 replay intentionally uses chunks of 128 samples; DVS accumulates the same statistics from its 16-sample evaluation minibatches. No division by `M`, `M-1`, `TN`, or `TN-1` occurs before eigendecomposition; dividing by any common population/sample scalar would not change the scale-free ranks. Evidence: `src/analysis/feedback_subspace/metrics.py:21-64`; `diagnostics/archived/v9_replay/replay_v9.py:45-50`; `diagnostics/archived/v9_replay/dvs_gate_v9.py:23-31`.

### Eigendecomposition and clipping

The spectrum path explicitly symmetrizes `S` as `(S+S.T)/2`, calls the full `torch.linalg.eigvalsh` on the requested device in float32, and falls back after `RuntimeError` to CPU float64. Eigenvalues are clipped with `clamp_min_(0)`, reversed to descending order, and converted to NumPy float64. The corresponding singular values are `sigma_i=sqrt(lambda_i)`. Evidence: `src/analysis/feedback_subspace/metrics.py:104-113,138-147`.

When eigenvectors are required, `pca_from_covariance` performs the full `torch.linalg.eigh` of the same symmetrized scatter, applies the same descending-order reversal and nonnegative eigenvalue clipping, and retains the requested leading columns. Evidence: `src/analysis/feedback_expansion/core.py:58-95`.

The run archive does not record whether each historical eigensolve used the normal float32-device branch or the float64 fallback. This is an operational detail that cannot be recovered after the fact; the algorithm and fallback are fully specified.

### Rank definitions

Let the descending clipped eigenvalues be `lambda_1 >= ... >= lambda_D >= 0`, let `E=sum_i lambda_i`, and, for `E>0`, let `p_i=lambda_i/E`.

| Quantity | Exact archived definition | Implementation |
|---|---|---|
| `r50`, `r80`, `r90`, `r95`, `r99` | `min{k: sum_(i<=k) p_i >= tau}` for `tau` in `{.50,.80,.90,.95,.99}` | `np.searchsorted(cumulative, tau, side="left") + 1` |
| stable rank | `E/lambda_1`, equivalently `||X_c||_F^2 / ||X_c||_2^2` | `total / eigenvalues[0]` |
| spectral entropy | `H=-sum_(p_i>0) p_i log(p_i)` | positive entries only |
| entropy effective rank | `exp(-sum_(p_i>0) p_i log(p_i+1e-30))` | `entropy_rank` in the core implementation |
| normalized entropy effective rank | entropy effective rank divided by `D` | output post-processing |

For zero total energy, all cutoff ranks, stable rank, participation ratio, and entropy effective rank are reported as zero. Negative eigensolver roundoff is clipped to zero without a separate cutoff-rank tolerance. The separate `numerical_rank` tolerance, `sqrt(max(rows,D)*eps_float32)*sigma_max`, is not used by `r95`, stable rank, or entropy rank. Evidence: `src/analysis/feedback_subspace/metrics.py:116-125,138-190`; `diagnostics/archived/v9_replay/replay_v9.py:45-52,131-136`.

### Two distinct meanings of “within” in the archive

They must not be conflated:

1. **Within-class residualization.** For each label `c`, the code accumulates a class-centered scatter and then sums the class scatters, `S_W,class=sum_c sum_(i in c)(x_i-mu_c)(x_i-mu_c)^T`. The main geometry table labels this `residualization=within_class`. That path does not emit an explicit between-class matrix. Evidence: `src/analysis/paper1_geometry/metrics.py:55-60,101-129`.
2. **Within-time / between-sample decomposition.** For `x[t,n] in R^D`, the covariance replay defines

```text
mu_n    = (1/T) sum_t x[t,n]
mu      = (1/(TN)) sum_n sum_t x[t,n]
S_total = sum_n sum_t (x[t,n]-mu)(x[t,n]-mu)^T
S_within_time = sum_n sum_t (x[t,n]-mu_n)(x[t,n]-mu_n)^T
S_between_sample = T sum_n (mu_n-mu)(mu_n-mu)^T.
```

The identity is `S_total=S_within_time+S_between_sample`. Population covariance is obtained by dividing every matrix by `N*T`. A component trace fraction is `tr(S_component)/tr(S_total)`. Evidence: `diagnostics/archived/covariance/replay_covariance_decomposition.py:315-364`.

The decomposition residual is

```text
||S_total-S_within_time-S_between_sample||_F / (||S_total||_F + 1e-30).
```

The replay acceptance threshold is `5e-6`. The maximum archived residual is `1.8767147887308863e-06`, in the N-MNIST/seed-20260831/H3/actual/delta record. Evidence: implementation at `diagnostics/archived/covariance/replay_covariance_decomposition.py:347-404,871-875`; frozen value at `results/frozen_sources/covariance/covariance_decomposition_per_seed.csv:67`, column `identity_relative_frobenius`.

For a non-total component whose trace fraction is at most `1e-12`, the raw trace is retained but all scale-invariant ranks are explicitly reported as zero; this handles the analytically zero within-time term of a time-constant tensor. Evidence: `diagnostics/archived/covariance/replay_covariance_decomposition.py:365-388`.

## E. Homogenization and controls

All offline controls below start from the same frozen epoch-100 SNN-DFA checkpoint, fixed `basis_eval` probe, fixed output-error carrier `q`, and fixed trained neuron coordinate system. They are signal manipulations and have no task-accuracy interpretation. The archived offline tensor is `[T,N,D]=[30,512,800]` for each seed/layer. Evidence: `diagnostics/archived/mechanisms/offline_gate_manipulations.py:68-87`.

### Progressive temporal homogenization (offline)

The exact coefficients are `alpha in {0,0.25,0.5,0.75,1}`. For actual gate `G[t,n,d]`,

```text
Gbar[t,n,d] = (1/T) sum_s G[s,n,d]     # broadcast over t
R_alpha     = (1-alpha) G + alpha Gbar
c_alpha     = ||G||_F / (||R_alpha||_F + 1e-12)
G_alpha     = c_alpha R_alpha
delta_alpha = G_alpha * q[None,:,:].
```

The Frobenius norm covers all `T*N*D` entries of one complete 512-sample evaluation tensor for one seed and layer; it is not per sample, timestep, neuron, or minibatch. Interpolation precedes scaling. The operation preserves the checkpoint, `q`, sample order, time labels, and neuron coordinates; after epsilon-stabilized scaling it preserves the one global gate Frobenius norm. It changes temporal deviations from each sample/neuron’s time mean and recomputes `delta`. At `alpha=1`, the gate is constant across time for each sample/neuron. Evidence: `diagnostics/archived/mechanisms/offline_gate_manipulations.py:52-53,86-93` and frozen rows/columns `alpha`, `scale_mode`, `N`, `T`, `D` in `results/frozen_sources/mechanisms/temporal_homogenization_per_seed.csv`.

This operation is distinct from training MeanGate: offline homogenization concatenates the full evaluation probe before applying a single scalar, while training MeanGate applies one scalar to each current layer/minibatch tensor.

### Magnitude mask (offline)

For each seed/layer, the code flattens the entire actual `[T,N,D]` gate, performs one stable ascending sort of `abs(G)`, and zeros exactly `k=round(f*T*N*D)` entries at the front of that ordering. Target fractions are `f in {0,0.25,0.50,0.75,f_ANN}`, where `f_ANN` is the actual temporal-ANN exact-zero fraction for the same seed/layer on `basis_eval`. Because the ordering is computed once, target supports are nested. Evidence: `diagnostics/archived/mechanisms/offline_gate_manipulations.py:64-67,94-102`.

The archive contains both unscaled and global-Frobenius-matched variants. The norm-matched copy uses the same one-scalar `rescale` function; normalized spectral metrics are copied because a global nonzero scale cannot change them. This control preserves `q`, all unmasked gate locations and values up to the final scalar, and the trained SNN coordinate system. It changes global support by deleting the smallest-magnitude entries. “ANN-matched” preserves only the ANN’s **global exact-zero fraction**; it does not use ANN support positions, ANN gate values, ANN `q`, per-neuron sparsity, or per-timestep sparsity. Evidence: `diagnostics/archived/mechanisms/offline_gate_manipulations.py:95-102`; summary-generation rule at `diagnostics/archived/mechanisms/analyze_offline_results.py:16-21,35-40`.

### Random mask (offline)

For each seed/layer and each of ten deterministic replicates, NumPy `default_rng(mask_seed)` creates one permutation of all `T*N*D` locations. The first `round(f*T*N*D)` locations are zeroed for the same five target fractions; using one permutation makes fractions nested within a replicate. The exact mask seed is `202609100000 + seed*100 + layer_index*10 + replicate`. Every masked tensor is then globally Frobenius-rescaled before `delta=G_masked*q`. Evidence: `diagnostics/archived/mechanisms/offline_gate_manipulations.py:46-50,103-109`.

The operation preserves `q`, the original value at each surviving location up to one scalar, the target global masked-entry count, and the global Frobenius norm after scaling. It changes which entries remain active and does not preserve per-neuron, per-sample, or per-time support. For reported ANN-matched ratios, the metric is first averaged over the ten masks within each seed/layer; seed/layer units are summarized only afterward. Evidence: `diagnostics/archived/mechanisms/analyze_offline_results.py:16-21,35-40`.

### Per-neuron marginal shuffle (offline)

The gate is reshaped from `[T,N,D]` to `[N*T,D]`. For each fixed neuron coordinate `d`, an independent RNG permutation is applied across the combined sample/time observations. The exact shuffle seed is `202609190000 + seed*10 + layer_index`. An assertion checks equality of the sorted before/after values for every neuron. No scale correction is needed (`scale_mode=value_multiset_exact`). Evidence: `diagnostics/archived/mechanisms/offline_gate_manipulations.py:55-60,110-112`.

This exactly preserves each neuron’s complete value multiset, hence every one-dimensional per-neuron marginal, the global Frobenius norm, and global zero counts. It changes sample identity, timestep identity, cross-neuron joint gate vectors, and alignment with fixed `q[n,d]`; it does not preserve each sample’s sequence of complete gate vectors.

### MeanGate (retrained gradient intervention)

On every training minibatch and every intervened hidden layer, the actual gate is detached, averaged over time (`dim=0`) for each sample and remaining feature/spatial coordinate, broadcast to all timesteps, and multiplied by one scalar

```text
||G_actual||_F / (||G_temporal_mean||_F + 1e-12)
```

covering the full current layer/minibatch tensor. Evidence: `src/methods/gate_intervention/gates.py:71-94`. The main N-MNIST MeanGate run leaves `intervention_layers=None`, so all hidden layers are intervened; evidence: `diagnostics/archived/v9_replay/train_new_seeds.py:46` and `src/analysis/paper1_geometry/training.py:46-55,101-105`.

The forward pass remains actual. The trainer constructs `desired_delta=q*G_applied`, injects it through a current-level proxy, and uses `injected_proxy + (production_proxy-injected_proxy).detach()`, which preserves the production proxy’s scalar value while replacing only its current gradient. Evidence: `src/analysis/paper1_geometry/training.py:118-155`. Per step, this preserves the forward activations/output, detached `q`, and the gate’s one global minibatch Frobenius norm, while eliminating within-sample temporal gate variation in the local gradient. It is a 100-epoch retrained intervention, so final weights are not held fixed.

### ShuffledGate (retrained gradient intervention)

For each sample, a deterministic affine time permutation `pi(t)=(a*t+b) mod T`, with `gcd(a,T)=1`, is keyed by run seed, completed epoch, and immutable source-sample index. The same permutation is applied to all feature/spatial coordinates of that sample. Evidence: `src/methods/gate_intervention/gates.py:20-68,95-112`; call-site creation at `src/analysis/paper1_geometry/training.py:91-100,135-142`.

This preserves each sample’s multiset of complete gate vectors over time, all gate values and norms, the actual forward pass, and the production proxy scalar. It changes the timestep assignment of the gate and therefore its alignment with time-dependent layer inputs in the local weight-gradient contraction. The main N-MNIST run applies it to all hidden layers. This is not the per-neuron marginal shuffle above: ShuffledGate uses one coordinated time permutation per sample, whereas marginal shuffle independently permutes each neuron across sample and time.

The archived DVS causal intervention is a separate H3-only variant (`intervention_layers={2}`), not an all-layer DVS intervention. Evidence: `scripts/diagnostics/paper1_experiment04_stageB_run.py:190-192`.

### Memoryless surrogate (retrained control)

Every parameterless `LIFCell` attached to hidden layers and the readout is replaced with

```text
s_t = surrogate_spike(I_t - threshold, surrogate_beta),
```

so `u_t=I_t`; there is no carried membrane, leak term, or reset path, and the `detach_temporal` argument has no effect. The forward spike remains the same hard threshold and the backward derivative remains the production surrogate. Evidence: `diagnostics/archived/mechanisms/stateless_surrogate.py:6-18`; production hard threshold/surrogate at `src/models/neurons/lif.py:9-18`; stateful reference dynamics at `src/models/neurons/lif.py:39-50`.

The control is initialized from the same paired initial model/feedback state; replacing cells leaves the parameter-state hash unchanged. It preserves the architecture and widths, `T=30`, hard threshold, surrogate beta, DFA feedback initialization, data split, optimizer, learning-rate schedule, gradient clipping, and 100-epoch budget, and changes only the neuron recurrence before retraining. Evidence: the released source launch procedure `diagnostics/archived/mechanisms/train_stateless_control.py:23-43`, plus the frozen three-seed results and `replaced_cells` field at `results/frozen_sources/mechanisms/stateless_surrogate_per_seed.csv:1-10`.

## F. DVS-Gesture

### Protocol and tensor spaces

The frozen DFA protocol is Conv32 / Conv64 / FC256 with an 11-class readout, `T=30`, batch size 16, and seeds `{20260830,20260831,20260901}`. The fixed training-only probe has 896 examples split into 448 fit and 448 evaluation examples. Spectra use completed epoch-100 checkpoints; accuracy is also separately available at the validation-selected checkpoint. Evidence: `configs/dvs_gesture/paper1_experiment03.yaml:4-17,30-45`; `diagnostics/archived/v9_replay/dvs_gate_v9.py:10-19,44-45`.

The raw captured gate and the representation used for the two cosine variants are:

| Layer | Raw captured shape per evaluation minibatch | Raw-cosine feature shape per minibatch | Projected feature shape per minibatch | Projected full-evaluation shape |
|---|---:|---:|---:|---:|
| H1 | `[30,16,32,128,128]` | `[30,16,524288]` | `[30,16,32]` | `[30,448,32]` |
| H2 | `[30,16,64,64,64]` | `[30,16,262144]` | `[30,16,64]` | `[30,448,64]` |
| H3 | `[30,16,256]` | `[30,16,256]` | `[30,16,256]` | `[30,448,256]` |

Raw shapes are frozen in `results/frozen_sources/dvs/gate_per_seed.csv:2-19`, column `source_raw_gate_batch_shape`. The flattening and batch aggregation are implemented at `diagnostics/archived/v9_replay/dvs_gate_v9.py:29-34,46-53`. Spatial projection is `mean(dim=(-2,-1))` in `src/analysis/paper1_geometry/metrics.py:17-25`; projected batches are concatenated over the sample axis at `diagnostics/archived/v9_replay/dvs_gate_v9.py:32,45`.

For convolutional local error, production first broadcasts `q/(H*W)` over space and multiplies it by the raw gate. The canonical projection then averages delta over `H,W`; the production `1/(H*W)` carrier scale is retained. This scalar affects absolute covariance energy but not `r95`. Evidence: `src/analysis/paper1_geometry/diagnostics.py:132-148`; `diagnostics/archived/covariance/replay_covariance_decomposition.py:553-576,774-782,936-939`.

### Exact raw and projected gate-cosine definition

For either representation `g[t,n] in R^D`, define

```text
u[t,n] = g[t,n] / (||g[t,n]||_2 + 1e-12)
c[n] = (||sum_t u[t,n]||_2^2 - T) / (T*(T-1))
reported cosine = mean_n c[n].
```

Evidence: `src/analysis/feedback_expansion/core.py:206-219` and `diagnostics/archived/v9_replay/replay_v9.py:53-68`. With no zero-norm DVS gate vectors (all frozen rows have `zero_norm_fraction=0`), this equals the mean cosine over all `t != s` pairs for each sample—870 ordered pairs, or equivalently 435 unordered pairs at `T=30`—then the arithmetic mean over 448 evaluation samples. Epsilon handling and the zero-vector convention are recorded in column `cosine_definition` of `results/frozen_sources/dvs/gate_per_seed.csv:2-19`.

For **raw cosine**, H1/H2 flatten channel and all spatial coordinates, compute the per-sample statistic in each 16-example minibatch, and combine minibatch results with sample-count weights. All 28 canonical evaluation minibatches are equal-sized. For **projected cosine**, H1/H2 first average each channel over space, concatenate all 448 samples, and compute the same statistic once. H3 has no spatial axes; raw and projected values differ only at tiny floating-point aggregation-order scale.

### Per-seed frozen DFA values

`q/post/aggregated` denotes carrier `q r95`, projected post-gate timestep `delta r95`, and projected temporal-mean `delta r95`, respectively. `delta W/B` and `gate W/B` are within-time/between-sample trace fractions. Accuracy is repeated across layers only to keep every layer row self-contained.

| Seed | Layer | Validation-selected test accuracy | Epoch-100 test accuracy | q/post/aggregated r95 | Raw gate cosine | Projected gate cosine | delta W/B | gate W/B |
|---:|:---:|---:|---:|:---:|---:|---:|:---:|:---:|
| 20260830 | H1 | 0.8020833333333334 | 0.7743055555555556 | 6/6/6 | 0.16219390662653105 | 0.9962298274040222 | 0.008377371577554664/0.9916227053931894 | 0.19575471037726083/0.8042454003438617 |
| 20260830 | H2 | 0.8020833333333334 | 0.7743055555555556 | 7/7/7 | 0.3637241763728006 | 0.9954853653907776 | 0.014077244364807595/0.9859228419319896 | 0.17687678638785773/0.8231233113099833 |
| 20260830 | H3 | 0.8020833333333334 | 0.7743055555555556 | 7/197/102 | 0.09249298168080193 | 0.0924929827451706 | 0.908126639647208/0.09187336200577298 | 0.9061345400472458/0.0938654585673899 |
| 20260831 | H1 | 0.7361111111111112 | 0.7395833333333334 | 6/5/5 | 0.19203997563038552 | 0.9967893958091736 | 0.007563233170822222/0.9924368532676837 | 0.22347459790742283/0.7765254873103224 |
| 20260831 | H2 | 0.7361111111111112 | 0.7395833333333334 | 6/6/6 | 0.3871825156467302 | 0.9943318367004395 | 0.012551991288842773/0.9874480992270033 | 0.20177435478854264/0.7982257233411882 |
| 20260831 | H3 | 0.7361111111111112 | 0.7395833333333334 | 7/189/95 | 0.11473665865404266 | 0.1147366613149643 | 0.8943898300104057/0.10561017535456983 | 0.8951403698384158/0.10485962875656793 |
| 20260901 | H1 | 0.7916666666666666 | 0.7847222222222222 | 7/6/6 | 0.18485830404928752 | 0.9969858527183533 | 0.008623320750551205/0.9913767855178522 | 0.2396116135068217/0.7603884484677104 |
| 20260901 | H2 | 0.7916666666666666 | 0.7847222222222222 | 7/8/8 | 0.34842227186475483 | 0.9928364157676697 | 0.01697983360008104/0.9830202584406263 | 0.15943753616260134/0.8405625358116334 |
| 20260901 | H3 | 0.7916666666666666 | 0.7847222222222222 | 7/189/107 | 0.07830606321138996 | 0.07830606400966644 | 0.9266177867031096/0.07338221522497985 | 0.922703163596204/0.0772968293561943 |

Source-row map for the table:

| Seed/layer | Accuracy rows in `results/frozen_sources/dvs/test_results.csv` | Rank/cosine rows in `results/frozen_sources/dvs/gate_per_seed.csv` | Aggregated-rank row in `results/frozen_sources/dvs/dvs_h3_causal_results.csv` | Decomposition rows in `results/frozen_sources/covariance/covariance_decomposition_per_seed.csv` |
|---|---:|---:|---:|---:|
| 20260830/H1 | 2-3 | 2-3 | 2 | 365-371 |
| 20260830/H2 | 2-3 | 4-5 | 2 | 372-378 |
| 20260830/H3 | 2-3 | 6-7 | 2 | 379-385 |
| 20260831/H1 | 4-5 | 8-9 | 3 | 386-392 |
| 20260831/H2 | 4-5 | 10-11 | 3 | 393-399 |
| 20260831/H3 | 4-5 | 12-13 | 3 | 400-406 |
| 20260901/H1 | 6-7 | 14-15 | 4 | 407-413 |
| 20260901/H2 | 6-7 | 16-17 | 4 | 414-420 |
| 20260901/H3 | 6-7 | 18-19 | 4 | 421-427 |

The relevant columns are `test_accuracy`, `q_r95`, `credit_r95`, `temporal_gate_cosine`, `H* delta aggregated r95`, `signal`, `component`, `r95`, and `trace_fraction_of_total`.

Across the three seeds, validation-selected DFA test accuracy is `0.7766203703703703 +/- 0.035466558902880904` (sample SD, `ddof=1`); epoch-100 accuracy is `0.7662037037037037 +/- 0.023634928074840415`. These are computed directly from `results/frozen_sources/dvs/test_results.csv:2-7` by selecting `method=dvs_dfa` and the corresponding `checkpoint` value.

### Three-seed layer summaries

All entries are arithmetic mean `+/-` sample SD over the three frozen seeds (`ddof=1`), computed from the source rows mapped above.

| Layer | Carrier q r95 | Post-gate timestep r95 | Temporally aggregated r95 | Raw gate cosine | Projected gate cosine | delta within fraction | delta between fraction |
|:---:|---:|---:|---:|---:|---:|---:|---:|
| H1 | 6.333333333 +/- 0.577350269 | 5.666666667 +/- 0.577350269 | 5.666666667 +/- 0.577350269 | 0.179697395 +/- 0.015577971 | 0.996668359 +/- 0.000392277 | 0.008187975 +/- 0.000554842 | 0.991812115 +/- 0.000554836 |
| H2 | 6.666666667 +/- 0.577350269 | 7.000000000 +/- 1.000000000 | 7.000000000 +/- 1.000000000 | 0.366442988 +/- 0.019522630 | 0.994217873 +/- 0.001328147 | 0.014536356 +/- 0.002249341 | 0.985463733 +/- 0.002249340 |
| H3 | 7.000000000 +/- 0.000000000 | 191.666666667 +/- 4.618802154 | 101.333333333 +/- 6.027713773 | 0.095178568 +/- 0.018363179 | 0.095178569 +/- 0.018363180 | 0.909711419 +/- 0.016172320 | 0.090288584 +/- 0.016172322 |

The projected post/pre ratios, computed per seed as `credit_r95/q_r95` and then averaged, are H1 `0.896825397 +/- 0.090141402`, H2 `1.047619048 +/- 0.082478610`, and H3 `27.380952381 +/- 0.659828879`. Their inputs are columns `credit_r95`, `q_r95`, and `GE` in `results/frozen_sources/dvs/gate_per_seed.csv:2-19`.

### Existing DVS BPTT records (no new training)

The archive already contains matched DVS-BPTT task results. Validation-selected test accuracies are `0.8854166666666666`, `0.8993055555555556`, and `0.8993055555555556` for seeds 20260830/20260831/20260901, giving `0.8946759259259259 +/- 0.008018753738744838` (sample SD). Epoch-100 accuracies are `0.9097222222222222`, `0.9027777777777778`, and `0.8993055555555556`, giving `0.9039351851851851 +/- 0.005303907054347018`. Evidence: `results/frozen_sources/dvs/test_results.csv:8-13`, columns `method`, `seed`, `checkpoint`, and `test_accuracy`. No BPTT retraining is needed to report these archived task results.

### DVS facts that remain unavailable or must be qualified

- H1/H2 **raw high-dimensional credit spectra** were not computed or archived. The `raw_neuron_and_spatial_features` rows repeat `credit_r95`, `q_r95`, and `GE` as shared metadata from the projected diagnostic; they are not raw-spatial r95 values. Only raw gate-cosine summaries and projected channel-space credit spectra are supported.
- The raw H1/H2 gate tensors themselves were not saved in the curated repository. Exact raw-cosine summaries, shapes, code, probe indices, and checkpoint hashes are present, but replaying those summaries requires the referenced checkpoints.
- The eigensolver branch actually taken in each historical spectrum call (float32 device or float64 fallback) is not logged.
- Accuracy and geometry use different checkpoint semantics: headline task accuracy is validation-selected, whereas the reported DVS spectra/gate statistics use completed epoch 100. They must remain separately labeled.

## G. Full numerical tables

The release-authoring table builder is `reproduce/build_appendix_tables.mjs`.
It uses the frozen CSV inputs and emitted both the requested CSVs and one
formatted workbook. Every output row includes `source_file` or
`provenance_file`; values are selected or joined from real source rows, never
typed from the paper. The builder's `@oai/artifact-tool` dependency belongs to
the authoring environment and is not a reviewer runtime dependency.

| File | Rows | SHA-256 |
|---|---:|---|
| `results/nmnist_main_per_seed.csv` | 45 | `25ef72a2e96e77cd62e326f3eda5358092ae3b4f9f9a888d95258d2f9cfcf439` |
| `results/cutoff_metrics_per_seed.csv` | 75 | `5f9e450ebe4502cebc053388efc05213ee1485d7b3aaf43671dd0da023db0169` |
| `results/threshold_free_metrics_per_seed.csv` | 75 | `f597257cd15ae6f6584801f0664b452db596a173b6d86056e5b64be7fa36cd60` |
| `results/homogenization_per_seed_layer_alpha.csv` | 75 | `35a88b8c61ff2e27bae8f22481fa370e68b67bb7902091e5b0a8074825317f2d` |
| `results/controls_per_seed.csv` | 945 | `8d11692b6fd39e40d444de68667f81ec14d3dbaf83f9147c29bf713fa8e0c7bb` |
| `results/memoryless_per_seed.csv` | 9 | `69bdffa01234c93126e1fb1a11d23a34f6f3ce052851be189d501293ce439369` |
| `results/dvs_per_seed.csv` | 9 | `9bfc6816cc50ba121fb61d02ceae70c706cfdf93a60a1d8efb1b7cf7dae7b58b` |
| `results/probe_size_sensitivity.csv` | 2700 | `a1eebd730d56a2dfba9c20a11821e54f1c895ac7c8350a0dc1fab4e4bd0ab304` |
| `results/width_robustness.csv` | 27 | `813402d906af41e789826cd1b189aefe28e31d310f91eb1738e677ad4192a174` |

Workbook: `supplement/appendix_tables.xlsx` (954109 bytes,
SHA-256 `33f441187f3cc39e5fce71abbf77a2cce9eae77779da03ac3d89e9c8d46979c4`).

`dvs_per_seed.csv` has one row per seed/layer. Accuracy uses both explicitly
labeled checkpoint rules; raw/projected cosine, q/post/aggregated ranks, shapes,
and delta/gate within/between fractions remain distinct columns.

## H. Repository and privacy audit

The public layout is `configs/`, `src/`, `diagnostics/`, `reproduce/`,
`metadata/`, `results/`, and `supplement/`. The release excludes datasets,
training logs, submission PDFs, private repository addresses, host/IP details,
credentials, usernames, and machine-local paths. In the bundled release copy,
location-only config fields were removed/replaced with `DATA_ROOT` and
`OUTPUT_DIR`; no scientific state was changed. The original and release hashes,
changed keys, tensor count, and state-equivalence check are recorded in
`metadata/checkpoints/checkpoint_release_provenance.json`.
`metadata/source_code_provenance.csv` confirms
that all eight high-impact production files checked against the completed-run
server copy are byte-identical.

`metadata/checkpoints/checkpoint_manifest.csv` lists all 28 final checkpoints
(25 N-MNIST and three DVS-Gesture) by method/seed and frozen SHA-256. The bundled
file is a path-sanitized release copy of the canonical SNN-DFA seed-20260830
epoch-100 scientific state. The other binaries are represented by
path/hash/config metadata, not silently replaced with older checkpoints.

## I. Running the repository

```bash
python -m pip install -e .
bash reproduce/smoke_test.sh --source-only
DATA_ROOT=/path/containing/nmnist bash reproduce/smoke_test.sh
bash reproduce/make_figures.sh
DATA_ROOT=/path/containing/nmnist bash reproduce/reproduce_main_diagnostics.sh
```

The actual release smoke used seed 20260830/H1 with a separately supplied
matching pre-framed N-MNIST NPZ tree and passed every assertion:

- checkpoint SHA-256 and stored completed epoch: passed;
- exact probe ordering and fit/eval assignment: passed;
- samples/time/features: `[30,512,800]`;
- total `r95`: expected 546, measured 546;
- within-time fraction: expected 0.892891085269444,
  measured 0.8928910990951194;
- covariance identity residual: 6.9905749826120049e-07;
- no optimizer step and no training.

The portable smoke output is
`metadata/smoke_results/checkpoint_smoke_seed_20260830_H1.json`.

## J. Confirmed and not-confirmed boundaries

### Spearman checkpoint-count audit

The exact checkpoint-count audit was read from `results/frozen_sources/nmnist/correlation_breakdown.csv`. Pooled correlations use `n=105` and each layer uses `n=35`. The frozen checkpoint rule is `scheduled_0_10_25_50_75_100_plus_validation_best_deduplicated_by_model_identity`. The archived p-value field is `not_reported`; no p value is inferred.

| temporal mode | group | rho | n | p value |
|---|---|---:|---:|---|
| timestep | all layers | -0.6028121473827908 | 105 | not_reported |
| timestep | H1 | -0.8039825585146007 | 35 | not_reported |
| timestep | H2 | -0.4871899577397523 | 35 | not_reported |
| timestep | H3 | -0.0703208906828761 | 35 | not_reported |
| time-collapsed | all layers | -0.8233528648708845 | 105 | not_reported |
| time-collapsed | H1 | -0.9371937780762201 | 35 | not_reported |
| time-collapsed | H2 | -0.9168236719706306 | 35 | not_reported |
| time-collapsed | H3 | -0.6173192424347538 | 35 | not_reported |

### Fig.2D dominant-subspace audit

**Verdict C.** The archive retains real overlap/principal-angle summaries but
does not retain the matching independently trained DFA and BPTT basis matrices
or the underlying teaching-signal matrices needed to refit them. Principal angles provide singular values of
`Q_DFA.T @ Q_BPTT`; they do not determine its 1,024 individual entries.
Consequently a real top-32 matrix `abs(Q_DFA.T @ Q_BPTT)` cannot be uniquely
reconstructed, and this repository does not generate a 32x32 heatmap. The safe
paper display is the archived layerwise normalized-overlap and mean-angle
summary only. See `supplement/fig2d_data_audit.md` and
`results/frozen_sources/nmnist/fig2_subspace_overlap.csv`.

### Main-text versus appendix scope

Confirmed facts suitable for the main text are the frozen five-seed N-MNIST
accuracy/rank comparisons, the specified intervention/control summaries, the
within-time decomposition, and the explicitly qualified three-seed DVS boundary
result. Per-seed cutoffs, threshold-free ranks, probe-size repeats, width results,
full alpha trajectories, configuration records, and DVS row-level diagnostics
are best placed in the appendix/supplement.

The following facts are not recoverable and are not claimed:

1. The clean archive consumes pre-generated NPZ frames but lacks the exact raw
   event-to-NPZ converter and pinned SpikingJelly version. Post-NPZ loading,
   shapes, splits and indices are exact for a matching supplied tree;
   bit-for-bit raw conversion and NPZ dataset identity are not verified.
2. One path-sanitized release copy of 28 final checkpoint binaries is bundled.
   Its canonical identity and unchanged scientific state are verified. The other 27 have
   sanitized configuration/path/hash records, but cannot be replayed from this
   repository alone.
3. The historical eigensolver branch (normal float32-device branch versus its
   documented CPU-float64 fallback) was not logged per call.
4. The original unpacked experiment tree had no usable Git commit. Source SHA-256
   manifests are the provenance mechanism.
5. DVS H1/H2 raw-spatial credit spectra were not archived. Their raw gate cosine
   values must not be described as raw credit ranks.
6. The archive's dataset checksum includes source path strings and is not
   portable across directory roots; the exact split/probe file hashes are
   portable and published.

Existing DVS-BPTT task accuracies are archived in
`results/frozen_sources/dvs/test_results.csv`; this audit did not launch new BPTT
training. These boundaries do not alter the frozen scientific conclusions.
