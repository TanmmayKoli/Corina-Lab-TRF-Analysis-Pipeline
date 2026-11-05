# run_permutation.py — TRF Permutation Significance Test

import sys
import glob
import time
import argparse
import numpy as np

sys.path.insert(0, '.')

from sign_language_trf.config import load_session_config, load_experiment_config
from sign_language_trf.preprocessing.eeg_loader import load_eeg
from sign_language_trf.preprocessing.motion_loader import load_motion
from sign_language_trf.preprocessing.markers_loader import load_markers
from sign_language_trf.preprocessing.elan_loader import load_elan
from sign_language_trf.preprocessing.synchronization import (
    extract_presentation_segments, extract_segment_eeg
)
from sign_language_trf.regressors.manager import RegressorManager, register_elan_regressors
from sign_language_trf.models.forward_trf import make_trf_model
from sign_language_trf.models.evaluation import per_channel_r


def load_data(participant):
    """Load all data and return (Xt, pres, n_pres, n_trim, ch_names, cfg, exp_cfg)."""
    if participant == 'MC':
        session_yaml = 'config/session.yaml'
    else:
        session_yaml = 'config/session_haggerty.yaml'

    cfg     = load_session_config(session_yaml)
    exp_cfg = load_experiment_config('config/experiments/motion_state_v3.yaml')

    print(f"Participant: {cfg.participant}", flush=True)
    motion  = load_motion(cfg.motion_path)
    markers = load_markers(cfg.markers_path)

    elan = load_elan(glob.glob('sign_language_trf/data/elan/*.eaf')[0])
    register_elan_regressors(elan)

    mgr = RegressorManager(motion, markers, target_fs=float(cfg.common_rate))
    X, _ = mgr.get_matrix(['motion_energy', 'velocity_stasis'])

    print("Loading EEG...", flush=True)
    eeg  = load_eeg(cfg.eeg_path, cfg, bandpass=True, verbose=False)
    segs = extract_presentation_segments(
        eeg, onset_code=cfg.onset_code, offset_code=cfg.end_code
    )
    pres = [extract_segment_eeg(eeg, s) for s in segs]

    n_trim   = min(P.shape[0] for P in pres)
    pres     = [P[:n_trim] for P in pres]
    Xt       = X[:n_trim]
    n_pres   = len(pres)
    ch_names = list(eeg.ch_names)

    print(f"  {n_pres} presentations, {n_trim} samples, {len(ch_names)} channels\n")
    return Xt, pres, n_pres, n_trim, ch_names, cfg, exp_cfg


def run_cv(Xin, pres, n_pres, cfg, exp_cfg):
    """Leave-one-out CV, returns mean per-channel r."""
    all_r = []
    for k in range(n_pres):
        train  = [pres[i] for i in range(n_pres) if i != k]
        X_3d   = np.stack([Xin] * len(train), axis=1)
        y_3d   = np.stack(train, axis=1)
        m      = make_trf_model(exp_cfg.trf, cfg)
        m.fit(X_3d, y_3d)
        all_r.append(per_channel_r(pres[k], m.predict(Xin)))
    return np.mean(all_r, axis=0)


def main():
    parser = argparse.ArgumentParser(
        description="Permutation test for TRF base model significance"
    )
    parser.add_argument('--participant', '-p', choices=['MC', 'EH'], default='MC')
    parser.add_argument('--n-perm', type=int, default=200,
                        help="Total number of permutations (full run only)")
    parser.add_argument('--batch-start', type=int, default=None,
                        help="Start index for batch mode (inclusive)")
    parser.add_argument('--batch-end',   type=int, default=None,
                        help="End index for batch mode (exclusive)")
    args = parser.parse_args()

    batch_mode = (args.batch_start is not None and args.batch_end is not None)
    start  = args.batch_start if batch_mode else 0
    end    = args.batch_end   if batch_mode else args.n_perm
    n_perm = end - start

    Xt, pres, n_pres, n_trim, ch_names, cfg, exp_cfg = load_data(args.participant)
    n_ch      = len(ch_names)
    f7        = ch_names.index('F7')
    MIN_SHIFT = int(10 * float(cfg.common_rate))   # 10-second minimum shift

    # Compute observed r (always)
    print("Computing observed r...", flush=True)
    r_obs = run_cv(Xt, pres, n_pres, cfg, exp_cfg)
    print(f"Observed F7 = {r_obs[f7]:.4f}\n", flush=True)

    # Advance RNG to correct position so batches are reproducible and consistent
    rng = np.random.default_rng(42)
    if start > 0:
        _ = rng.integers(MIN_SHIFT, n_trim - MIN_SHIFT, size=start)

    null_dist = np.zeros((n_perm, n_ch))
    t0        = time.time()

    for idx, i in enumerate(range(start, end)):
        shift          = int(rng.integers(MIN_SHIFT, n_trim - MIN_SHIFT))
        null_dist[idx] = run_cv(np.roll(Xt, shift, axis=0), pres, n_pres, cfg, exp_cfg)
        if (idx + 1) % 10 == 0 or idx == 0:
            elapsed = time.time() - t0
            eta     = elapsed / (idx + 1) * (n_perm - idx - 1)
            print(f"  [{i+1:>3d}]  null F7={null_dist[idx, f7]:.4f}  "
                  f"ETA {eta:.0f}s", flush=True)

    if batch_mode:
        out = f'results/perm_batch_{start}_{end}.npz'
        np.savez(out, r_obs=r_obs, null_batch=null_dist,
                 ch_names=np.array(ch_names))
        print(f"\nSaved → {out}", flush=True)
    else:
        p_vals  = np.mean(null_dist >= r_obs[np.newaxis, :], axis=0)
        n_sig05 = (p_vals < 0.05).sum()
        n_sig01 = (p_vals < 0.01).sum()
        print(f"\nF7 p={p_vals[f7]:.4f}  |  p<0.05: {n_sig05}/{n_ch}  "
              f"|  p<0.01: {n_sig01}/{n_ch}", flush=True)

        out = 'results/permutation_test.npz'
        np.savez(out,
                 ch_names  = np.array(ch_names),
                 r_obs     = r_obs,
                 null_dist = null_dist,
                 p_vals    = p_vals,
                 n_perm    = args.n_perm)
        print(f"Saved → {out}", flush=True)


if __name__ == '__main__':
    main()
