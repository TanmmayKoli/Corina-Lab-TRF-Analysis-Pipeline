# run_analysis.py — Unified TRF Experiment Runner

import sys
import glob
import argparse
import numpy as np

sys.path.insert(0, '.')

from sign_language_trf.config import load_session_config, load_experiment_config
from sign_language_trf.preprocessing.eeg_loader import load_eeg
from sign_language_trf.preprocessing.motion_loader import load_motion
from sign_language_trf.preprocessing.markers_loader import load_markers
from sign_language_trf.preprocessing.synchronization import (
    extract_presentation_segments, extract_segment_eeg
)
from sign_language_trf.regressors.manager import (
    RegressorManager, register_video_regressors
)
from sign_language_trf.models.forward_trf import make_trf_model
from sign_language_trf.models.evaluation import per_channel_r

VIDEO_PATH = 'sign_language_trf/data/video/Tristan5.mp4'

# ── Shared CV function ─────────────────────────────────────────────────────────

def run_cv(X, pres, n_pres, n_trim, exp_cfg, cfg, ch_names, f7, o1, label=""):
    """Leave-one-presentation-out CV. Returns mean per-channel r."""
    Xt    = X[:n_trim]
    all_r = []
    for k in range(n_pres):
        train  = [pres[i] for i in range(n_pres) if i != k]
        X_3d   = np.stack([Xt] * len(train), axis=1)
        y_3d   = np.stack(train, axis=1)
        m      = make_trf_model(exp_cfg.trf, cfg)
        m.fit(X_3d, y_3d)
        all_r.append(per_channel_r(pres[k], m.predict(Xt)))
    r = np.mean(all_r, axis=0)
    o1_str = f"  O1={r[o1]:.4f}" if o1 is not None else ""
    print(f"  [{label:<42s}]  F7={r[f7]:.4f}{o1_str}  mean={r.mean():.4f}  max={r.max():.4f}",
          flush=True)
    return r


# ── Shared data loader ─────────────────────────────────────────────────────────

def load_participant_data(session_yaml):
    """Load EEG, motion, markers and build regressor manager for one participant."""
    cfg     = load_session_config(session_yaml)
    exp_cfg = load_experiment_config('config/experiments/ivc_bridge.yaml')

    print(f"\nParticipant: {cfg.participant}")
    print(f"EEG:         {cfg.eeg_path}")
    print(f"end_code:    {cfg.end_code}\n")

    print("Loading motion + markers...", flush=True)
    motion  = load_motion(cfg.motion_path)
    markers = load_markers(cfg.markers_path)

    print("Registering video regressors...", flush=True)
    register_video_regressors(VIDEO_PATH, motion.time_s)
    mgr = RegressorManager(motion, markers, target_fs=float(cfg.common_rate))

    print("Loading EEG...", flush=True)
    eeg  = load_eeg(cfg.eeg_path, cfg, bandpass=True, verbose=False)
    segs = extract_presentation_segments(
        eeg, onset_code=cfg.onset_code, offset_code=cfg.end_code
    )
    pres = [extract_segment_eeg(eeg, s) for s in segs]

    n_trim   = min(P.shape[0] for P in pres)
    pres     = [P[:n_trim] for P in pres]
    n_pres   = len(pres)
    ch_names = list(eeg.ch_names)
    f7       = ch_names.index('F7')
    o1       = ch_names.index('O1') if 'O1' in ch_names else None

    print(f"  {n_pres} presentations, {n_trim} samples "
          f"({n_trim/float(cfg.common_rate):.1f}s), {len(ch_names)} channels\n")

    return cfg, exp_cfg, mgr, pres, n_pres, n_trim, ch_names, f7, o1


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT: IVC
# Runs for both MC and EH. Saves to results/{participant}_results.npz.
# ══════════════════════════════════════════════════════════════════════════════

def run_ivc_experiment(session_yaml, out_file):
    """IVC experiment suite:"""
    cfg, exp_cfg, mgr, pres, n_pres, n_trim, ch_names, f7, o1 = \
        load_participant_data(session_yaml)

    print("Building feature matrices...", flush=True)

    X_ivc,     _ = mgr.get_matrix(['ivc'])
    X_base,    _ = mgr.get_matrix(['motion_energy', 'velocity_stasis'])
    X_bridge,  _ = mgr.get_matrix(['ivc', 'motion_energy', 'velocity_stasis'])
    X_face,    _ = mgr.get_matrix(['ivc_face'])
    X_dom,     _ = mgr.get_matrix(['ivc_dominant_hand'])
    X_nondom,  _ = mgr.get_matrix(['ivc_nondominant_hand'])
    X_regions, _ = mgr.get_matrix(['ivc_face', 'ivc_dominant_hand', 'ivc_nondominant_hand'])

    raw_ivc      = mgr.get('ivc')
    log_ivc      = np.log1p(raw_ivc)
    log_ivc_z    = (log_ivc - log_ivc.mean()) / log_ivc.std()
    X_ivc_log    = log_ivc_z.reshape(-1, 1)
    X_bridge_log = np.column_stack([log_ivc_z, X_base])

    print(f"  IVC={X_ivc.shape}  bridge={X_bridge.shape}\n")

    # Shared kwargs for run_cv
    cv_kw = dict(pres=pres, n_pres=n_pres, n_trim=n_trim,
                 exp_cfg=exp_cfg, cfg=cfg, ch_names=ch_names, f7=f7, o1=o1)

    print("=== EXPERIMENT 1: IVC ALONE ===")
    r_ivc = run_cv(X_ivc, label="IVC (pixel motion energy)", **cv_kw)

    print("\n=== EXPERIMENT 2: ROKOKO BASE ===")
    r_base = run_cv(X_base, label="motion_energy + velocity_stasis", **cv_kw)

    print("\n=== EXPERIMENT 3: BRIDGE (IVC + Rokoko) ===")
    r_bridge = run_cv(X_bridge, label="IVC + motion_energy + velocity_stasis", **cv_kw)

    print("\n=== EXPERIMENT 4: SPATIAL REGIONS ===")
    r_face    = run_cv(X_face,    label="IVC face region", **cv_kw)
    r_dom     = run_cv(X_dom,     label="IVC dominant hand region", **cv_kw)
    r_nondom  = run_cv(X_nondom,  label="IVC non-dominant hand region", **cv_kw)
    r_regions = run_cv(X_regions, label="IVC all regions combined", **cv_kw)

    print("\n=== EXPERIMENT 5: LOG-SCALED IVC ===")
    r_ivc_log    = run_cv(X_ivc_log,    label="log(IVC+1)", **cv_kw)
    r_bridge_log = run_cv(X_bridge_log, label="log(IVC+1) + motion_energy + stasis", **cv_kw)

    # ── Summary table ──────────────────────────────────────────────────────────
    rows = [
        ("IVC alone",               r_ivc),
        ("Rokoko base",             r_base),
        ("Bridge (IVC + Rokoko)",   r_bridge),
        ("IVC face",                r_face),
        ("IVC dominant hand",       r_dom),
        ("IVC non-dominant hand",   r_nondom),
        ("IVC all regions",         r_regions),
        ("log(IVC+1)",              r_ivc_log),
        ("Bridge log-IVC + Rokoko", r_bridge_log),
    ]
    o1_hdr = f"{'O1':>8}" if o1 is not None else ""
    print(f"\n── Summary at F7 and O1 ──────────────────────────────────────────────")
    print(f"  {'Model':<42s}  {'F7':>8}  {o1_hdr}  {'mean':>8}  {'max':>8}")
    print(f"  {'-'*78}")
    for lbl, r in rows:
        o_val = f"{r[o1]:8.4f}" if o1 is not None else ""
        print(f"  {lbl:<42s}  {r[f7]:8.4f}  {o_val}  {r.mean():8.4f}  {r.max():8.4f}")

    print("\n── Unique Variance (Δr vs Rokoko base) ──────────────────────────────")
    for lbl, r in rows:
        if lbl == "Rokoko base":
            continue
        d_f7 = r[f7] - r_base[f7]
        d_o1 = (r[o1] - r_base[o1]) if o1 is not None else float('nan')
        print(f"  {lbl:<42s}  ΔF7={d_f7:+.4f}  ΔO1={d_o1:+.4f}")

    np.savez(out_file,
             ch_names     = np.array(ch_names),
             r_ivc        = r_ivc,
             r_base       = r_base,
             r_bridge     = r_bridge,
             r_face       = r_face,
             r_dom        = r_dom,
             r_nondom     = r_nondom,
             r_regions    = r_regions,
             r_ivc_log    = r_ivc_log,
             r_bridge_log = r_bridge_log)
    print(f"\nSaved → {out_file}")


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT: HIGH PRIORITY  (MC only — requires raw Rokoko + ELAN)
# ══════════════════════════════════════════════════════════════════════════════

def run_high_priority_experiment():
    from sign_language_trf.preprocessing.elan_loader import load_elan
    from sign_language_trf.preprocessing.raw_rokoko_loader import load_raw_rokoko
    from sign_language_trf.regressors.manager import register_elan_regressors, register_raw_regressors

    cfg     = load_session_config('config/session.yaml')
    exp_cfg = load_experiment_config('config/experiments/motion_state_v3.yaml')

    print(f"\nParticipant: {cfg.participant} — High Priority Experiments\n")

    print("Loading motion + markers...", flush=True)
    motion  = load_motion(cfg.motion_path)
    markers = load_markers(cfg.markers_path)

    print("Loading ELAN...", flush=True)
    elan = load_elan(glob.glob('sign_language_trf/data/elan/*.eaf')[0])
    register_elan_regressors(elan)

    print("Loading raw Rokoko (reversals + position)...", flush=True)
    raw = load_raw_rokoko('sign_language_trf/data/rokoko/Tristan_Story_1_raw.csv')
    register_raw_regressors(raw)

    mgr = RegressorManager(motion, markers, target_fs=float(cfg.common_rate))

    print("Loading EEG...", flush=True)
    eeg  = load_eeg(cfg.eeg_path, cfg, bandpass=True, verbose=False)
    segs = extract_presentation_segments(eeg, onset_code=255, offset_code=254)
    pres = [extract_segment_eeg(eeg, s) for s in segs]
    n_trim   = min(P.shape[0] for P in pres)
    pres     = [P[:n_trim] for P in pres]
    n_pres   = len(pres)
    ch_names = list(eeg.ch_names)
    f7       = ch_names.index('F7')
    print(f"  {n_pres} presentations, {n_trim} samples, {len(ch_names)} channels\n")

    print("Building feature matrices...", flush=True)

    X_base,        _ = mgr.get_matrix(['motion_energy', 'velocity_stasis'])
    X_rev,         _ = mgr.get_matrix(['motion_energy', 'velocity_stasis',
                                       'left_hand_reversal', 'right_hand_reversal'])
    X_rev_only,    _ = mgr.get_matrix(['left_hand_reversal', 'right_hand_reversal'])
    X_left_stasis, _ = mgr.get_matrix(['motion_energy', 'velocity_stasis', 'left_hand_stasis'])
    X_right_stasis,_ = mgr.get_matrix(['motion_energy', 'velocity_stasis', 'right_hand_stasis'])
    X_both_stasis, _ = mgr.get_matrix(['motion_energy', 'velocity_stasis',
                                       'left_hand_stasis', 'right_hand_stasis'])
    X_pos,         _ = mgr.get_matrix(['motion_energy', 'velocity_stasis',
                                       'right_hand_height', 'right_hand_lateral'])
    X_height,      _ = mgr.get_matrix(['motion_energy', 'velocity_stasis', 'right_hand_height'])
    X_lateral,     _ = mgr.get_matrix(['motion_energy', 'velocity_stasis', 'right_hand_lateral'])

    rev_l = mgr.get('left_hand_reversal').sum()
    rev_r = mgr.get('right_hand_reversal').sum()
    ls    = (mgr.get('left_hand_stasis') > 0).mean() * 100
    rs    = (mgr.get('right_hand_stasis') > 0).mean() * 100
    print(f"  Direction reversals: L={rev_l:.0f}  R={rev_r:.0f}")
    print(f"  Per-hand stasis:     L={ls:.1f}%  R={rs:.1f}%\n")

    cv_kw = dict(pres=pres, n_pres=n_pres, n_trim=n_trim,
                 exp_cfg=exp_cfg, cfg=cfg, ch_names=ch_names, f7=f7, o1=None)

    print("=== BASE MODEL ===")
    r_base = run_cv(X_base, label="motion_energy + velocity_stasis", **cv_kw)

    print("\n=== EXPERIMENT A: DIRECTION REVERSALS ===")
    r_rev      = run_cv(X_rev,      label="base + L/R reversals", **cv_kw)
    r_rev_only = run_cv(X_rev_only, label="reversals only",        **cv_kw)

    print("\n=== EXPERIMENT B: PER-HAND STASIS ===")
    r_left_stasis  = run_cv(X_left_stasis,  label="base + left_hand_stasis",  **cv_kw)
    r_right_stasis = run_cv(X_right_stasis, label="base + right_hand_stasis", **cv_kw)
    r_both_stasis  = run_cv(X_both_stasis,  label="base + L+R stasis",        **cv_kw)

    print("\n=== EXPERIMENT C: SIGNING-SPACE POSITION ===")
    r_pos     = run_cv(X_pos,     label="base + height + lateral",      **cv_kw)
    r_height  = run_cv(X_height,  label="base + right_hand_height",     **cv_kw)
    r_lateral = run_cv(X_lateral, label="base + right_hand_lateral",    **cv_kw)

    print(f"\n── Unique Variance at F7 ─────────────────────────────────────────────")
    print(f"  Base:                        r  = {r_base[f7]:.4f}")
    for lbl, r in [
        ("Reversals (L+R)",     r_rev),
        ("Left stasis",         r_left_stasis),
        ("Right stasis",        r_right_stasis),
        ("Both unilateral",     r_both_stasis),
        ("Height + Lateral",    r_pos),
        ("Height only",         r_height),
        ("Lateral only",        r_lateral),
    ]:
        print(f"  {lbl:<28s}  Δr = {r[f7] - r_base[f7]:+.4f}")

    np.savez('results/high_priority_results.npz',
             ch_names       = np.array(ch_names),
             r_base         = r_base,
             r_rev          = r_rev,
             r_rev_only     = r_rev_only,
             r_left_stasis  = r_left_stasis,
             r_right_stasis = r_right_stasis,
             r_both_stasis  = r_both_stasis,
             r_pos          = r_pos,
             r_height       = r_height,
             r_lateral      = r_lateral,
             n_rev_left       = rev_l,
             n_rev_right      = rev_r,
             pct_left_stasis  = ls,
             pct_right_stasis = rs)
    print("\nSaved → results/high_priority_results.npz")


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT: ACCELERATION  (MC only — requires raw Rokoko + ELAN)
# ══════════════════════════════════════════════════════════════════════════════

def run_accel_experiment():
    from sign_language_trf.preprocessing.elan_loader import load_elan
    from sign_language_trf.preprocessing.raw_rokoko_loader import load_raw_rokoko
    from sign_language_trf.regressors.manager import register_elan_regressors, register_raw_regressors

    cfg     = load_session_config('config/session.yaml')
    exp_cfg = load_experiment_config('config/experiments/motion_state_v3.yaml')

    print(f"\nParticipant: {cfg.participant} — Acceleration Experiment\n")

    motion  = load_motion(cfg.motion_path)
    markers = load_markers(cfg.markers_path)
    elan    = load_elan(glob.glob('sign_language_trf/data/elan/*.eaf')[0])
    register_elan_regressors(elan)
    raw = load_raw_rokoko('sign_language_trf/data/rokoko/Tristan_Story_1_raw.csv')
    register_raw_regressors(raw)
    mgr = RegressorManager(motion, markers, target_fs=float(cfg.common_rate))

    print("Building feature matrices...", flush=True)
    X_base,  _ = mgr.get_matrix(['motion_energy', 'velocity_stasis'])
    X_full,  _ = mgr.get_matrix(['motion_energy', 'velocity_stasis', 'combined_hand_accel'])
    X_accel, _ = mgr.get_matrix(['combined_hand_accel'])
    X_peaks, _ = mgr.get_matrix(['motion_energy', 'velocity_stasis',
                                  'left_hand_accel_peak', 'right_hand_accel_peak'])
    print(f"  base={X_base.shape}  +accel={X_full.shape}  accel={X_accel.shape}  +peaks={X_peaks.shape}")

    print("\nLoading EEG...", flush=True)
    eeg  = load_eeg(cfg.eeg_path, cfg, bandpass=True, verbose=False)
    segs = extract_presentation_segments(eeg, onset_code=255, offset_code=254)
    pres = [extract_segment_eeg(eeg, s) for s in segs]
    n_trim   = min(P.shape[0] for P in pres)
    pres     = [P[:n_trim] for P in pres]
    n_pres   = len(pres)
    ch_names = list(eeg.ch_names)
    f7       = ch_names.index('F7')
    print(f"  {n_pres} presentations, {n_trim} samples, {len(ch_names)} channels")

    cv_kw = dict(pres=pres, n_pres=n_pres, n_trim=n_trim,
                 exp_cfg=exp_cfg, cfg=cfg, ch_names=ch_names, f7=f7, o1=None)

    print("\nRunning CV...", flush=True)
    r_base  = run_cv(X_base,  label="base: motion_energy + velocity_stasis", **cv_kw)
    r_full  = run_cv(X_full,  label="+accel: base + combined_hand_accel",    **cv_kw)
    r_accel = run_cv(X_accel, label="accel_only",                             **cv_kw)
    r_peaks = run_cv(X_peaks, label="+peaks: base + L/R accel peak events",  **cv_kw)

    uv_accel = r_full  - r_base
    uv_peaks = r_peaks - r_base

    print(f"\n── Unique Variance at F7 ─────────────────────────────────────────────")
    print(f"  Accel UV:   Δr = {uv_accel[f7]:+.4f}")
    print(f"  Peaks UV:   Δr = {uv_peaks[f7]:+.4f}")
    print(f"  Accel only: r  = {r_accel[f7]:.4f}")

    np.savez('results/accel_uv.npz',
             ch_names = np.array(ch_names),
             r_base   = r_base,
             r_full   = r_full,
             r_accel  = r_accel,
             r_peaks  = r_peaks,
             uv_accel = uv_accel,
             uv_peaks = uv_peaks)
    print("\nSaved → results/accel_uv.npz")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="TRF experiment runner — sign language EEG pipeline"
    )
    parser.add_argument(
        '--participant', '-p',
        choices=['MC', 'EH', 'all'],
        default='MC',
        help="Which participant to run. 'all' runs both MC and EH for IVC experiment."
    )
    parser.add_argument(
        '--experiment', '-e',
        choices=['ivc', 'high_priority', 'accel', 'all'],
        default='ivc',
        help=(
            "Which experiment(s) to run.\n"
            "  ivc           — IVC + Rokoko + bridge + spatial regions (MC and EH)\n"
            "  high_priority — reversals / per-hand stasis / signing position (MC only)\n"
            "  accel         — hand acceleration unique variance (MC only)\n"
            "  all           — all of the above"
        )
    )
    args = parser.parse_args()

    want_ivc = args.experiment in ('ivc', 'all')
    want_hp  = args.experiment in ('high_priority', 'all')
    want_acc = args.experiment in ('accel', 'all')

    participants = ['MC', 'EH'] if args.participant == 'all' else [args.participant]

    if want_ivc:
        for p in participants:
            print(f"\n{'='*70}")
            print(f"  IVC EXPERIMENT — {p}")
            print(f"{'='*70}")
            if p == 'MC':
                run_ivc_experiment('config/session.yaml', 'results/ivc_results.npz')
            else:
                run_ivc_experiment('config/session_haggerty.yaml', 'results/haggerty_results.npz')

    if want_hp:
        if args.participant == 'EH':
            print("\nNOTE: high_priority experiment requires raw Rokoko + ELAN, only available for MC.")
        else:
            print(f"\n{'='*70}")
            print(f"  HIGH PRIORITY EXPERIMENTS — MC")
            print(f"{'='*70}")
            run_high_priority_experiment()

    if want_acc:
        if args.participant == 'EH':
            print("\nNOTE: accel experiment requires raw Rokoko + ELAN, only available for MC.")
        else:
            print(f"\n{'='*70}")
            print(f"  ACCELERATION EXPERIMENT — MC")
            print(f"{'='*70}")
            run_accel_experiment()


if __name__ == '__main__':
    main()
