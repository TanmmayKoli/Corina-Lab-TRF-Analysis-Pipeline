# Corina Lab — Sign Language TRF Pipeline

Forward encoding model (TRF) predicting EEG from continuous motion features recorded during ASL narrative viewing. Full technical reference is in `PIPELINE.md`.

## Setup

```bash
pip install -r requirements.txt
```

## Running the pipeline

```bash
# Run IVC experiment for both participants
python run_analysis.py --participant all --experiment ivc

# Run high-priority kinematic experiments (MC only)
python run_analysis.py --participant MC --experiment high_priority

# Permutation significance test
python run_permutation.py --participant MC --n-perm 200

# Generate all plots
python plot_results.py --participant both --plot all
```

All outputs save to `results/`.

## Project structure

```
run_analysis.py          # Entry point: analysis experiments
run_permutation.py       # Entry point: permutation test
plot_results.py          # Entry point: all plots
sign_language_trf/       # Python package (preprocessing, regressors, models)
config/                  # Session configs (session.yaml = MC, session_haggerty.yaml = EH)
config/experiments/      # Experiment configs (ivc_bridge.yaml, motion_state_v3.yaml, ...)
sign_language_trf/data/  # EEG (.set/.fdt), Rokoko CSVs, ELAN .eaf, video, markers
results/                 # .npz result files and .png plots
notebooks/               # Jupyter notebooks (01–04)
PIPELINE.md              # Full technical documentation
```

## Participants

- **MC** — `config/session.yaml`, EEG: `data/eeg/MC.ICA.reject.set`, end_code=254
- **EH** — `config/session_haggerty.yaml`, EEG: `data/eeg/E.Haggerty.*.set`, end_code=253
- **Jenna** (pending) — add `config/session_jenna.yaml` when EEG is ready

## Key scientific context

- **Forward model**: stimulus features → EEG (Pearson r per channel)
- **Cross-validation**: leave-one-presentation-out (5 presentations per participant)
- **Key finding**: double dissociation — 3D kinematics (Rokoko) predict F7 frontal; IVC dominant hand predicts O1 occipital
- **Best kinematic model**: `motion_state_v3` — F7 r=0.1078, 8/29 channels significant (permutation p<0.05)
- **TRF window**: 0–500 ms, ridge α=1e7

## Adding a new participant

1. Copy `config/session.yaml` → `config/session_<name>.yaml`, update `participant`, `eeg_path`, `end_code`, `drop_channels`
2. Add EEG `.set`/`.fdt` files to `sign_language_trf/data/eeg/`
3. Run `python run_analysis.py --participant <NAME> --experiment ivc`

See `PIPELINE.md` for full details on regressors, experiments, and extending the pipeline.
