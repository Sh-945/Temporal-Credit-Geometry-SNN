# Training, data, and local-gradient audit notes

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

### C.4 Numerical contraction check (no training)

A deterministic source-only check executes the actual production
`Trainer._local_batch` path on the released `2312-800-800-800-10`, `T=30`
architecture. It compares

```text
explicit = einsum("tbo,tbi->oi", delta/delta.numel(), detached_input)
autograd = production local-proxy weight gradient
relative_error = ||explicit-autograd||_F / (||autograd||_F+1e-30).
```

The released evidence is `metadata/smoke_results/local_gradient_parity.json`.
Every assertion passed; the maximum relative error between the explicit
contraction and the production trainer was `1.121796849321369e-07`, below the
preregistered `1e-6` tolerance. Run it with
`python diagnostics/local_gradient_parity.py` or
`bash reproduce/smoke_test.sh --source-only`.

The stronger release-time checkpoint smoke loaded the canonical scientific
state, replayed the fixed 512-example H1 evaluation probe, and reported maximum
relative errors `5.32639781170019e-08` for `q*g` parity and
`2.7921030762554063e-07` for the explicit weight contraction. It also matched
the frozen H1 `r95=546` and covariance decomposition. Evidence:
`metadata/smoke_results/checkpoint_smoke_seed_20260830_H1.json` and
`diagnostics/checkpoint_smoke.py`.

Independent archived PyTorch-2.8/CUDA evidence is published at
`results/frozen_sources/nmnist/production_parity.csv`: 432 rows across three
seeds, six checkpoint epochs, two probe halves, four minibatches and three
layers. Its `production_delta_relative_error` spans
`4.86716764669382e-08` to `5.38465378951969e-08`.

## Unresolved or packaging-critical facts

1. One path-sanitized release copy of the hash-verified canonical N-MNIST SNN-DFA checkpoint is bundled at `metadata/checkpoints/nmnist_snn_dfa_seed_20260830_epoch_100_release.pt`. `metadata/checkpoints/checkpoint_release_provenance.json` records both hashes and verifies that all non-config scientific state is unchanged. The other 27 final binaries are represented by sanitized configuration records and exact canonical SHA-256 identities but are not redistributed.
2. The raw DVS/N-MNIST event-to-NPZ conversion script and exact SpikingJelly version are absent from the source archive. The current loader reproduces post-NPZ behavior; bit-exact raw AEDAT conversion is therefore not claimed.
3. The archived dataset hash named `sha256_path_size_and_content` includes each source path string (`scripts/diagnostics/paper1_prepare.py:85-100`), so it is not portable across directory roots. Preserve it as provenance; if NPZ data are distributed separately, publish a path-independent per-file content manifest.
4. Weight initialization is confirmed as an unmodified PyTorch default, not encoded as a repository-owned initializer. Pinning PyTorch 2.8.0 is necessary if exact initialization semantics are claimed.
5. The original unpacked experiment tree had no usable Git commit; byte-level source and artifact manifests are the available provenance mechanism.

## Audit status

- A: configuration, seeds, feedback sampling/frozen status — confirmed from code plus checkpoint metadata.
- B: formal train/validation/test semantics and exact split/probe/subsample indices — confirmed; upstream raw-event conversion version remains unresolved.
- C: detach semantics, proxy/reduction and pre-clipping gradient relationship — confirmed; the production-path source check and real checkpoint smoke both pass below `1e-6`.
