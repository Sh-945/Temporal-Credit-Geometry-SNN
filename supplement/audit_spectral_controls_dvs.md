# Spectral diagnostics, controls, and DVS-Gesture audit notes

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
