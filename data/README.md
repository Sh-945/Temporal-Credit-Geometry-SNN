# Dataset layout

Datasets are not included in this repository, and this checkout cannot fetch or
reconstruct them from raw event files on its own. The experiment loaders expect
pre-binned event frames stored as NumPy `.npz` files with a `frames` array of
shape `[T, C, H, W]`.

The default relative layout is:

```text
data/
├── nmnist/frames_number_30_split_by_number/
│   ├── train/{0..9}/*.npz
│   └── test/{0..9}/*.npz
└── dvs/frames_number_30_split_by_number/
    ├── train/{0..10}/*.npz
    └── test/{0..10}/*.npz
```

Check a prepared dataset with:

```bash
python -m pip install -e .
python scripts/check_data.py --config configs/nmnist/paper1_experiment03.yaml
python scripts/check_data.py --config configs/dvs_gesture/paper1_experiment03.yaml
```

To use a different location, update `data.root` in a local copy of the relevant
YAML file. Do not commit the event datasets or local cache files.

The original archive did not retain the exact raw-event-to-NPZ conversion
script or a pinned SpikingJelly version. It does confirm 30 event-count-based
frames for DVS-Gesture and the exact consumed NPZ shapes. Therefore this release
supports bit-exact loading, split generation, and probe selection for a supplied
matching NPZ tree, but does not claim bit-exact raw AEDAT conversion or verify
NPZ dataset identity. The historical aggregate dataset hash included path
strings and is not portable; no path-independent NPZ content manifest was
recoverable. See `../REPRODUCIBILITY_AUDIT.md`.
