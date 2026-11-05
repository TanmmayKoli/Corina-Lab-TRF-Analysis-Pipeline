# TRF Pipeline — Technical Reference

Forward encoding model: continuous motion features → EEG (Pearson r per channel, leave-one-presentation-out CV).

---

## Directory structure

```
sign_language_trf/          # Python package
├── config.py               # Config loader (YAML → dataclass)
├── pipeline.py             # End-to-end convenience wrapper
├── preprocessing/
│   ├── eeg_loader.py       # MNE read + bandpass + drop channels
│   ├── eeg_markers.py      # BioSemi trigger extraction
│   ├── motion_loader.py    # Rokoko hands-only CSV → MotionData
│   ├── raw_rokoko_loader.py# Full Rokoko CSV → joint positions
│   ├── markers_loader.py   # Processed BioSemi markers CSV
│   ├── elan_loader.py      # ELAN .eaf → sign timings/glosses
│   └── synchronization.py  # Segment extraction per presentation
├── regressors/
│   ├── manager.py          # RegressorManager: builds + caches feature matrices
│   ├── velocity.py         # Left/right hand velocity (L2 norm)
│   ├── motion_energy.py    # Combined bilateral motion energy
│   ├── stasis.py           # Bilateral stasis state + per-hand variants
│   ├── peaks.py            # Global motion acceleration peaks
│   ├── acceleration.py     # Acceleration + direction reversals + position
│   ├── video.py            # IVC from video (pixel motion energy, spatial regions)
│   └── linguistic.py       # ELAN-derived sign onset/offset regressors
├── models/
│   ├── forward_trf.py      # mTRFpy wrapper: ridge TRF (tmin, tmax, alpha)
│   ├── cross_validation.py # Leave-one-presentation-out CV loop
│   ├── evaluation.py       # per_channel_r (Pearson r per EEG channel)
│   └── permutation.py      # Circular time-shift null distribution
├── data/
│   ├── eeg/                # .set + .fdt files (MNE EEGLAB format)
│   ├── rokoko/             # Motion CSVs
│   ├── elan/               # .eaf annotation files
│   ├── markers/            # Processed BioSemi trigger CSVs
│   └── video/              # Stimulus video (Tristan5.mp4)
config/
├── session.yaml            # MC participant config
├── session_haggerty.yaml   # EH participant config
└── experiments/
    ├── ivc_bridge.yaml     # IVC + Rokoko bridge model
    └── motion_state_v3.yaml# Best kinematic posture model (8 sig channels)
run_analysis.py             # Entry point: analysis experiments
run_permutation.py          # Entry point: permutation significance test
plot_results.py             # Entry point: all plots
results/                    # .npz output files (auto-created)
```

---

## How to run

### Analysis

```bash
# Single participant, single experiment
python run_analysis.py --participant MC --experiment ivc
python run_analysis.py --participant EH --experiment ivc
python run_analysis.py --participant MC --experiment high_priority
python run_analysis.py --participant MC --experiment accel

# All participants × all experiments
python run_analysis.py --participant all --experiment all
```

Outputs:
- `results/ivc_results.npz` (MC) or `results/haggerty_results.npz` (EH)
- `results/high_priority_results.npz`, `results/accel_results.npz` (MC only)

### Permutation test

```bash
# Full run (200 permutations), MC
python run_permutation.py --participant MC --n-perm 200

# Batch mode for parallelizing (e.g., 0–50 on one machine, 50–100 on another)
python run_permutation.py --participant MC --batch-start 0 --batch-end 50
python run_permutation.py --participant MC --batch-start 50 --batch-end 100
```

Batches use RNG fast-forward (seed=42) so results are reproducible and consistent with the full run. Outputs: `results/permutation_test.npz` or `results/perm_batch_{start}_{end}.npz`.

### Plots

```bash
# Single participant
python plot_results.py --participant MC --plot results
python plot_results.py --participant MC --plot kernels
python plot_results.py --participant MC --plot bands
python plot_results.py --participant EH --plot all

# Comparison and group average (requires both participants)
python plot_results.py --participant both --plot comparison
python plot_results.py --participant both --plot group
python plot_results.py --participant both --plot all
```

Plot outputs (saved to `results/`):
- `MC_results_summary.png`, `EH_results_summary.png` — per-channel r bar charts
- `MC_kernels_plot.png`, `EH_kernels_plot.png` — TRF kernel waveforms per model
- `MC_band_plot.png`, `EH_band_plot.png` — broadband/delta/theta/alpha/beta
- `comparison_MC_EH.png` — side-by-side topographies
- `group_avg_results.png`, `group_avg_kernels.png` — mean ± range across N=2

Kernels and bands are computed from scratch if `.npz` not found, then saved for reuse.

---

## Participants

| Participant | Config file | EEG file | end_code | Dropped channels |
|---|---|---|---|---|
| MC | `config/session.yaml` | `MC.ICA.reject.set` | 254 | LMastoid, RMastoid, Eye, Oz, FC5, O2 |
| EH | `config/session_haggerty.yaml` | `E.Haggerty.8.25.26_filt_chop_elist_bins_ica_rmic_copy.set` | 253 | same + EXG4–EXG8 |

Both: 512 Hz acquisition → 64 Hz common rate, 5 presentations (~173s each), onset_code=255.

---

## Experiments

### IVC suite (`--experiment ivc`, runs for MC + EH)

Uses `config/experiments/ivc_bridge.yaml` (ridge, α=1e7, tmin=0s, tmax=0.5s).

| Model | Regressors | Scientific question |
|---|---|---|
| IVC alone | ivc | Does pixel motion predict EEG? |
| Rokoko base | motion_energy + velocity_stasis | Does 3D kinematics predict EEG? |
| Bridge | ivc + motion_energy + velocity_stasis | Double dissociation: IVC→occipital, kinematics→frontal? |
| IVC face | ivc_face | Face region contribution |
| IVC dominant hand | ivc_dominant_hand | Dominant hand spatial region |
| IVC non-dominant hand | ivc_nondominant_hand | Non-dominant hand spatial region |
| IVC all regions | ivc_face + ivc_dominant_hand + ivc_nondominant_hand | Full spatial decomposition |
| log(IVC+1) | log-scaled ivc | Nonlinear IVC scaling |
| Bridge log-IVC | log(IVC+1) + motion_energy + stasis | Log-scaled bridge |

Key finding: double dissociation — 3D kinematics (Rokoko) predict F7 (frontal), IVC dominant hand predicts O1 (occipital).

### High priority (`--experiment high_priority`, MC only)

Uses `config/experiments/motion_state_v3.yaml`. Requires full Rokoko CSV + ELAN.

Tests whether neural activity reflects transitions into stable signing postures (stasis). Key model: `motion_state_v3` with motion_energy + velocity_stasis + stasis onset/offset + per-hand stops + motion peaks → F7 r=0.1078, 8 channels significant (permutation p<0.05).

### Acceleration (`--experiment accel`, MC only)

Tests acceleration-derived regressors: direction reversals, per-hand stasis, signing-space position (height, lateral). Requires full Rokoko CSV + ELAN.

---

## Regressors

### Continuous

| Name | Source | Description |
|---|---|---|
| `motion_energy` | Rokoko hands-only | L2 velocity magnitude, both hands combined |
| `velocity_stasis` | Rokoko hands-only | Binary: both hands < 0.05 m/s simultaneously |
| `left_hand_stasis` | Rokoko raw | Left hand stasis (per-hand threshold) |
| `right_hand_stasis` | Rokoko raw | Right hand stasis (per-hand threshold) |
| `ivc` | Video | IVC(t) = Σ[xᵢ(t)−xᵢ(t-1)]² across all pixels |
| `ivc_face` | Video | IVC restricted to face bounding region |
| `ivc_dominant_hand` | Video | IVC restricted to dominant hand region |
| `ivc_nondominant_hand` | Video | IVC restricted to non-dominant hand region |
| `right_hand_height` | Rokoko raw | Vertical signing space position |
| `right_hand_lateral` | Rokoko raw | Lateral signing space position |

### Event-derived

| Name | Source | Description |
|---|---|---|
| `velocity_stasis_onset_10` | Rokoko | Stasis onset events at 0.10 m/s threshold |
| `velocity_stasis_offset_10` | Rokoko | Stasis offset events at 0.10 m/s threshold |
| `left_hand_stop_onset` | Rokoko hands-only CSV | Left hand stop events (~83 events) |
| `right_hand_stop_onset` | Rokoko hands-only CSV | Right hand stop events (~33 events) |
| `global_motion_peak` | Rokoko | Acceleration peaks (~101 events) |
| `left_hand_reversal` | Rokoko raw | Left hand direction reversal events |
| `right_hand_reversal` | Rokoko raw | Right hand direction reversal events |

### ELAN-derived (future)

| Name | Source | Description |
|---|---|---|
| `sign_onset` | ELAN .eaf | Sign boundary onsets |
| `sign_offset` | ELAN .eaf | Sign boundary offsets |

---

## Data files

| File | Format | Description |
|---|---|---|
| `data/rokoko/Tristan_Story_1_hands_only.csv` | 10 cols, ~10490 rows @ ~59Hz | time_s, time_ms, L/RHand vx/vy/vz/stopped_rule |
| `data/rokoko/Tristan_Story_1_raw.csv` | Full Rokoko joint CSV | All joints; needed for reversals, position, per-hand stasis |
| `data/video/Tristan5.mp4` | MP4 | Stimulus video used for IVC extraction |
| `data/eeg/MC.ICA.reject.set` | EEGLAB .set | MC EEG, ICA cleaned |
| `data/eeg/E.Haggerty.*.set` | EEGLAB .set | EH EEG, ICA cleaned |
| `data/markers/Tristan_Story_1_processed_biosemi_markers.csv` | CSV | Processed BioSemi triggers (shared for both participants) |
| `data/elan/*.eaf` | ELAN XML | Sign timing and gloss annotations |

---

## Model config

Ridge TRF via mTRFpy. Key parameters in `config/experiments/*.yaml`:

- `tmin / tmax`: TRF window in seconds (currently 0.0–0.5s)
- `alpha`: Ridge regularization (1e7, optimized via sweep 100→1e8; F7 plateaus at 1e7–1e8)
- `folds: leave_one_presentation_out`: 5-fold CV across 5 video presentations

---

## Adding a new participant

1. Create `config/session_<name>.yaml` — copy `session.yaml`, update: `participant`, `eeg_path`, `end_code`, `drop_channels`, `n_presentations`.
2. Add the EEG `.set`/`.fdt` files to `data/eeg/`.
3. Run: `python run_analysis.py --participant <NAME> --experiment ivc`
4. Run: `python plot_results.py --participant <NAME> --plot all`

For group averages across >2 participants, update `plot_group()` in `plot_results.py` (currently hardcoded for MC + EH).

## Adding a new experiment

1. Create `config/experiments/<name>.yaml` — define `regressors`, `trf` params, `evaluation`.
2. Add a `run_<name>_experiment()` function in `run_analysis.py` following the existing pattern.
3. Wire it into `main()` in `run_analysis.py` under `--experiment <name>`.
4. Add a corresponding plot function in `plot_results.py` if needed.

## Adding a new regressor

1. Implement the computation in the appropriate `regressors/` module (or create a new one).
2. Register it in `RegressorManager.get()` in `regressors/manager.py`.
3. Reference it by name in an experiment YAML under `regressors:`.
