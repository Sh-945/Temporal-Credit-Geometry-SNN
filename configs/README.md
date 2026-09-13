# Configuration scope

The frozen manuscript uses these configuration files:

- `nmnist/paper1_experiment03.yaml`: canonical N-MNIST training and diagnostics.
- `nmnist/paper1_experiment04_width400.yaml` and
  `nmnist/paper1_experiment04_width1200.yaml`: frozen width robustness.
- `dvs_gesture/paper1_experiment03.yaml`: canonical DVS-Gesture protocol.

The remaining YAML files are retained verbatim from the same production source
tree for code provenance. They are legacy development configurations, are not
invoked by the reviewer one-click scripts, and carry no frozen-result claim in
this manuscript. In particular, the N-Caltech101 and SHD configurations are
outside the frozen ICASSP 2027 evidence scope.
