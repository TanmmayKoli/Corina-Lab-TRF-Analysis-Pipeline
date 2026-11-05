# plot_results.py — Unified TRF Results Plotter

import sys
import glob
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.signal import butter, filtfilt

sys.path.insert(0, '.')

from sign_language_trf.config import load_session_config, load_experiment_config
from sign_language_trf.preprocessing.eeg_loader import load_eeg
from sign_language_trf.preprocessing.motion_loader import load_motion
from sign_language_trf.preprocessing.markers_loader import load_markers
from sign_language_trf.preprocessing.synchronization import (
    extract_presentation_segments, extract_segment_eeg
)
from sign_language_trf.regressors.manager import RegressorManager, register_video_regressors
from sign_language_trf.models.forward_trf import RidgeTRF
from sign_language_trf.models.evaluation import per_channel_r
from mne.decoding import ReceptiveField

# ── Shared constants ───────────────────────────────────────────────────────────

VIDEO_PATH = 'sign_language_trf/data/video/Tristan5.mp4'
ALPHA      = 10_000_000.0

BLUE   = '#2166ac'
ORANGE = '#d6604d'
GREEN  = '#4dac26'
PURPLE = '#762a83'
MC_C   = '#1a6faf'
EH_C   = '#b2182b'

CH_ORDER = ['Fp1','Fp2','AF3','AF4','F7','F3','Fz','F4','F8',
            'FC1','FC2','FC6','T7','C3','Cz','C4','T8',
            'CP1','CP2','CP5','CP6','P7','P3','Pz','P4','P8',
            'PO3','PO4','O1']

BANDS = {
    'Broadband\n(0.1–30 Hz)': (None, None),
    'Delta\n(1–4 Hz)':        (1.0,  4.0),
    'Theta\n(4–8 Hz)':        (4.0,  8.0),
    'Alpha\n(8–13 Hz)':       (8.0, 13.0),
    'Beta\n(13–30 Hz)':       (13.0, 30.0),
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def sort_channels(ch_names):
    """Return (ordered_names, ordered_indices) filtered to available channels."""
    ordered = [c for c in CH_ORDER if c in ch_names]
    idx     = [ch_names.index(c) for c in ordered]
    return ordered, idx


def bandpass_filter(data, lo, hi, fs=64.0, order=4):
    """data: (n_times, n_channels)"""
    nyq = fs / 2.0
    if lo is None:
        b, a = butter(order, hi / nyq, btype='low')
    elif hi is None:
        b, a = butter(order, lo / nyq, btype='high')
    else:
        b, a = butter(order, [lo / nyq, hi / nyq], btype='band')
    return filtfilt(b, a, data, axis=0)


def load_eeg_segments(cfg):
    """Load EEG and split into per-presentation arrays."""
    eeg  = load_eeg(cfg.eeg_path, cfg, bandpass=True, verbose=False)
    segs = extract_presentation_segments(
        eeg, onset_code=cfg.onset_code, offset_code=cfg.end_code
    )
    pres = [extract_segment_eeg(eeg, s) for s in segs]
    n_trim = min(P.shape[0] for P in pres)
    pres   = [P[:n_trim] for P in pres]
    return eeg, pres, n_trim


def run_cv_band(X, pres, n_pres, sfreq):
    """Leave-one-out CV on a set of (potentially band-filtered) presentations."""
    Xt    = X
    all_r = []
    for k in range(n_pres):
        train  = [pres[i] for i in range(n_pres) if i != k]
        X_3d   = np.stack([Xt] * len(train), axis=1)
        y_3d   = np.stack(train, axis=1)
        m = RidgeTRF(tmin=0.0, tmax=0.5, sfreq=sfreq, alpha=ALPHA)
        m.fit(X_3d, y_3d)
        all_r.append(per_channel_r(pres[k], m.predict(Xt)))
    return np.mean(all_r, axis=0)


def fit_kernel(X, pres, n_pres, sfreq, tmin=0.0, tmax=0.5, label=""):
    """Fit TRF kernel on all presentations simultaneously (no CV)."""
    X_3d = np.stack([X] * n_pres, axis=1)
    y_3d = np.stack(pres, axis=1)
    rf   = ReceptiveField(tmin=tmin, tmax=tmax, sfreq=sfreq,
                          estimator=ALPHA, scoring="corrcoef",
                          edge_correction=True)
    rf.fit(X_3d, y_3d)
    delays_ms = rf.delays_ / sfreq * 1000
    print(f"  [{label}] kernel shape: {rf.coef_.shape}")
    return rf.coef_, delays_ms


def session_yaml_for(participant):
    return 'config/session.yaml' if participant == 'MC' else 'config/session_haggerty.yaml'


def results_file_for(participant):
    return 'results/ivc_results.npz' if participant == 'MC' else 'results/haggerty_results.npz'


def kernels_file_for(participant):
    return 'results/ivc_kernels.npz' if participant == 'MC' else 'results/haggerty_kernels.npz'


def band_file_for(participant):
    return 'results/ivc_band_trf.npz' if participant == 'MC' else 'results/haggerty_band_trf.npz'


# ══════════════════════════════════════════════════════════════════════════════
# PLOT: RESULTS SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

def plot_results(participant):
    """Summary figure: double dissociation + spatial regions + channel profiles."""
    tag = participant
    d   = np.load(results_file_for(participant), allow_pickle=True)
    ch_names = [str(c) for c in d['ch_names']]
    f7  = ch_names.index('F7')
    o1  = ch_names.index('O1')

    r_base    = d['r_base']
    r_ivc     = d['r_ivc']
    r_bridge  = d['r_bridge']
    r_face    = d['r_face']
    r_dom     = d['r_dom']
    r_nondom  = d['r_nondom']
    r_regions = d['r_regions']

    ch_ord, idx_ord = sort_channels(ch_names)
    x_c = np.arange(len(ch_ord))

    fig = plt.figure(figsize=(16, 12), facecolor='#f8f8f8')
    gs  = GridSpec(3, 3, figure=fig, hspace=0.52, wspace=0.38,
                   left=0.07, right=0.97, top=0.91, bottom=0.07)

    # Panel A: Double dissociation bar chart
    ax_a = fig.add_subplot(gs[0, :2])
    models  = ['Rokoko\n(kinematics)', 'IVC global\n(pixel motion)',
                'IVC dominant\nhand only', 'Bridge\n(IVC + Rokoko)']
    f7_vals = [r_base[f7], r_ivc[f7], r_dom[f7], r_bridge[f7]]
    o1_vals = [r_base[o1], r_ivc[o1], r_dom[o1], r_bridge[o1]]
    x   = np.arange(len(models))
    w   = 0.35
    b_f7 = ax_a.bar(x - w/2, f7_vals, w, color=BLUE,   label='F7 (left frontal-temporal)', alpha=0.88, zorder=3)
    b_o1 = ax_a.bar(x + w/2, o1_vals, w, color=ORANGE, label='O1 (left occipital)',        alpha=0.88, zorder=3)
    ax_a.axhline(0, color='black', lw=0.8, zorder=2)
    ax_a.set_xticks(x); ax_a.set_xticklabels(models, fontsize=10)
    ax_a.set_ylabel("Pearson r", fontsize=10)
    ax_a.set_title(f"A  |  Double dissociation — {tag}: kinematics → frontal, IVC → occipital",
                   fontsize=11, fontweight='bold', loc='left', pad=8)
    ax_a.set_ylim(-0.04, 0.14); ax_a.legend(fontsize=9, loc='upper right')
    ax_a.grid(axis='y', alpha=0.3, zorder=0); ax_a.set_facecolor('white')
    for bar, col in [(b_f7, BLUE), (b_o1, ORANGE)]:
        for b in bar:
            h = b.get_height(); yp = h + 0.003 if h >= 0 else h - 0.007
            ax_a.text(b.get_x() + b.get_width()/2, yp, f'{h:.3f}', ha='center',
                      va='bottom' if h >= 0 else 'top', fontsize=8, color=col, fontweight='bold')

    # Panel B: Spatial regions
    ax_b = fig.add_subplot(gs[0, 2])
    regs  = ['Face\nregion', 'Dominant\nhand', 'Non-dominant\nhand', 'All\nregions']
    rf7   = [r_face[f7], r_dom[f7], r_nondom[f7], r_regions[f7]]
    ro1   = [r_face[o1], r_dom[o1], r_nondom[o1], r_regions[o1]]
    xr    = np.arange(4)
    ax_b.bar(xr - 0.2, rf7, 0.38, color=BLUE,   alpha=0.85, label='F7', zorder=3)
    ax_b.bar(xr + 0.2, ro1, 0.38, color=ORANGE, alpha=0.85, label='O1', zorder=3)
    ax_b.axhline(0, color='black', lw=0.8, zorder=2)
    ax_b.set_xticks(xr); ax_b.set_xticklabels(regs, fontsize=8.5)
    ax_b.set_ylabel("Pearson r", fontsize=9)
    ax_b.set_title("B  |  Spatial IVC regions", fontsize=11, fontweight='bold', loc='left', pad=8)
    ax_b.legend(fontsize=8, loc='upper left')
    ax_b.grid(axis='y', alpha=0.3, zorder=0); ax_b.set_facecolor('white')
    ax_b.set_ylim(-0.03, 0.06)

    # Panel C: All-channel profile
    ax_c = fig.add_subplot(gs[1, :])
    dom_ord  = r_dom[idx_ord]
    base_ord = r_base[idx_ord]
    ivc_ord  = r_ivc[idx_ord]
    ax_c.fill_between(x_c, 0, base_ord, alpha=0.25, color=BLUE)
    ax_c.plot(x_c, base_ord, 'o-', color=BLUE, ms=4, lw=1.5, label='Rokoko base', zorder=4)
    ax_c.fill_between(x_c, 0, dom_ord, alpha=0.25, color=ORANGE)
    ax_c.plot(x_c, dom_ord, 's-', color=ORANGE, ms=4, lw=1.5, label='IVC dominant hand', zorder=4)
    ax_c.fill_between(x_c, 0, ivc_ord, alpha=0.15, color=GREEN)
    ax_c.plot(x_c, ivc_ord, '^--', color=GREEN, ms=3, lw=1.2, alpha=0.7, label='IVC global', zorder=3)
    ax_c.axhline(0, color='black', lw=0.8)
    ax_c.set_xticks(x_c); ax_c.set_xticklabels(ch_ord, rotation=45, ha='right', fontsize=8)
    ax_c.set_ylabel("Pearson r", fontsize=10)
    ax_c.set_title("C  |  Prediction accuracy across all channels (frontal → occipital)",
                   fontsize=11, fontweight='bold', loc='left', pad=8)
    ax_c.legend(fontsize=9, loc='upper right')
    ax_c.grid(axis='y', alpha=0.25); ax_c.set_facecolor('white')
    for xband, x0, x1, col, lbl in [
        ('', 0,  8,  '#ddeeff', 'Frontal'),
        ('', 9,  16, '#ffeedd', 'Temporal/Central'),
        ('', 17, 25, '#eeffee', 'Parietal'),
        ('', 26, len(ch_ord)-1, '#fff0dd', 'Occipital'),
    ]:
        ax_c.axvspan(x0-0.5, x1+0.5, alpha=0.12, color=col, zorder=0)
        ax_c.text((x0+x1)/2, ax_c.get_ylim()[1]*0.92, lbl,
                  ha='center', fontsize=7.5, color='#555555', style='italic')

    # Panel D: Unique IVC variance
    ax_d = fig.add_subplot(gs[2, :2])
    delta      = (r_bridge - r_base)[idx_ord]
    colors_d   = [ORANGE if v > 0 else '#888888' for v in delta]
    ax_d.bar(x_c, delta, color=colors_d, alpha=0.85, zorder=3)
    ax_d.axhline(0, color='black', lw=0.8)
    ax_d.set_xticks(x_c); ax_d.set_xticklabels(ch_ord, rotation=45, ha='right', fontsize=8)
    ax_d.set_ylabel("Δr (bridge − base)", fontsize=10)
    ax_d.set_title("D  |  Unique variance added by IVC on top of kinematics",
                   fontsize=11, fontweight='bold', loc='left', pad=8)
    ax_d.grid(axis='y', alpha=0.25); ax_d.set_facecolor('white')

    # Panel E: Key numbers text box
    ax_e = fig.add_subplot(gs[2, 2])
    ax_e.axis('off')
    lines = [
        ("KEY FINDINGS", '#222222', 11, True),
        (f"Kinematics (Rokoko)", BLUE, 9.5, True),
        (f"  F7 = {r_base[f7]:.3f}  ← best channel", BLUE, 9, False),
        (f"  O1 = {r_base[o1]:.3f}", BLUE, 9, False),
        (f"IVC dominant hand only", ORANGE, 9.5, True),
        (f"  F7 = {r_dom[f7]:.3f}", ORANGE, 9, False),
        (f"  O1 = {r_dom[o1]:.3f}  ← occipital signal", ORANGE, 9, False),
        (f"Bridge (IVC + Rokoko)", GREEN, 9.5, True),
        (f"  F7 = {r_bridge[f7]:.3f}", GREEN, 9, False),
        (f"  O1 = {r_bridge[o1]:.3f}", GREEN, 9, False),
    ]
    y = 0.97
    for txt, col, fs, bold in lines:
        ax_e.text(0.04, y, txt, transform=ax_e.transAxes,
                  fontsize=fs, color=col, fontweight='bold' if bold else 'normal',
                  va='top', ha='left')
        y -= 0.10 if bold else 0.090
    for spine in ax_e.spines.values():
        spine.set_visible(True); spine.set_color('#cccccc'); spine.set_linewidth(1.5)

    fig.suptitle(
        f"IVC Experiment Results — {tag}\n"
        "Dominant hand pixel motion → occipital  |  Rokoko kinematics → frontal-temporal",
        fontsize=12, fontweight='bold', y=0.975
    )
    out = f'results/{tag.lower()}_results_summary.png'
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#f8f8f8')
    print(f"Saved → {out}")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# PLOT: KERNELS
# ══════════════════════════════════════════════════════════════════════════════

def compute_and_save_kernels(participant):
    """Train kernels from scratch if .npz doesn't exist, then save."""
    kfile = kernels_file_for(participant)
    try:
        data = np.load(kfile, allow_pickle=True)
        print(f"  Loaded pre-computed kernels from {kfile}")
        return data
    except FileNotFoundError:
        pass

    print(f"  Computing kernels for {participant}...", flush=True)
    cfg  = load_session_config(session_yaml_for(participant))
    sfreq = float(cfg.common_rate)

    motion  = load_motion(cfg.motion_path)
    markers = load_markers(cfg.markers_path)
    register_video_regressors(VIDEO_PATH, motion.time_s)
    mgr = RegressorManager(motion, markers, target_fs=sfreq)

    eeg, pres, n_trim = load_eeg_segments(cfg)
    n_pres = len(pres)

    X_dom,    _ = mgr.get_matrix(['ivc_dominant_hand'])
    X_base,   _ = mgr.get_matrix(['motion_energy', 'velocity_stasis'])
    X_bridge, _ = mgr.get_matrix(['ivc_dominant_hand', 'motion_energy', 'velocity_stasis'])
    X_dom    = X_dom[:n_trim]
    X_base   = X_base[:n_trim]
    X_bridge = X_bridge[:n_trim]

    k_dom,    d_dom    = fit_kernel(X_dom,    pres, n_pres, sfreq, label="IVC dominant hand")
    k_base,   d_base   = fit_kernel(X_base,   pres, n_pres, sfreq, label="Rokoko base")
    k_bridge, d_bridge = fit_kernel(X_bridge, pres, n_pres, sfreq, label="Bridge")

    ch_names = list(eeg.ch_names)
    np.savez(kfile,
             ch_names=np.array(ch_names),
             k_dom=k_dom,       d_dom=d_dom,
             k_base=k_base,     d_base=d_base,
             k_bridge=k_bridge, d_bridge=d_bridge)
    print(f"  Saved → {kfile}")
    return np.load(kfile, allow_pickle=True)


def plot_kernels(participant):
    data     = compute_and_save_kernels(participant)
    ch_names = [str(c) for c in data['ch_names']]
    f7       = ch_names.index('F7')
    o1       = ch_names.index('O1')
    k_dom    = data['k_dom']
    k_base   = data['k_base']
    k_bridge = data['k_bridge']
    d_dom    = data['d_dom'] * 1000
    d_base   = data['d_base'] * 1000
    d_bridge = data['d_bridge'] * 1000

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), facecolor='#f8f8f8')
    fig.suptitle(f"TRF Kernels — {participant} (IVC and Bridge Models)\n"
                 "EEG response to each feature over time (0–500 ms post-stimulus)",
                 fontsize=12, fontweight='bold', y=0.99)

    configs = [
        (0, 0, k_dom, d_dom, 0, f7, ORANGE, "IVC dominant hand → F7"),
        (1, 0, k_dom, d_dom, 0, o1, ORANGE, "IVC dominant hand → O1"),
        (0, 1, k_base, d_base, 0, f7, BLUE,   "motion_energy → F7 (Rokoko base)"),
        (1, 1, k_base, d_base, 1, f7, BLUE,   "velocity_stasis → F7 (Rokoko base)"),
        (0, 2, k_bridge, d_bridge, 0, f7, PURPLE, "Bridge: IVC dom → F7"),
        (1, 2, k_bridge, d_bridge, 1, f7, GREEN,  "Bridge: motion_energy → F7"),
    ]

    for row, col, kernel, delays, feat, ch, color, title in configs:
        ax = axes[row, col]
        k  = kernel[ch, feat, :]
        ax.plot(delays, k, color=color, lw=2.0)
        ax.fill_between(delays, 0, k, alpha=0.18, color=color)
        ax.axhline(0, color='black', lw=0.8, linestyle='--', alpha=0.5)
        ax.axvline(0, color='grey',  lw=0.8, linestyle=':', alpha=0.5)
        ax.set_title(title, fontsize=9.5, fontweight='bold', pad=5)
        ax.set_xlabel("Time lag (ms)", fontsize=8.5)
        ax.set_ylabel("Weight (a.u.)", fontsize=8.5)
        ax.set_facecolor('white'); ax.grid(alpha=0.2)
        peak_idx = np.argmax(np.abs(k))
        ax.annotate(f'peak\n{delays[peak_idx]:.0f}ms',
                    xy=(delays[peak_idx], k[peak_idx]),
                    xytext=(delays[peak_idx] + 40, k[peak_idx] * 0.85),
                    fontsize=7.5, color=color,
                    arrowprops=dict(arrowstyle='->', color=color, lw=0.8))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = f'results/{participant.lower()}_kernels_plot.png'
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#f8f8f8')
    print(f"Saved → {out}")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# PLOT: BAND-FILTERED EEG
# ══════════════════════════════════════════════════════════════════════════════

def compute_and_save_bands(participant):
    """Run band-filtered CV if .npz doesn't exist, then save."""
    bfile = band_file_for(participant)
    try:
        data = np.load(bfile, allow_pickle=True)
        print(f"  Loaded pre-computed band results from {bfile}")
        return data
    except FileNotFoundError:
        pass

    print(f"  Computing band analysis for {participant}...", flush=True)
    cfg  = load_session_config(session_yaml_for(participant))
    sfreq = float(cfg.common_rate)

    motion  = load_motion(cfg.motion_path)
    markers = load_markers(cfg.markers_path)
    register_video_regressors(VIDEO_PATH, motion.time_s)
    mgr = RegressorManager(motion, markers, target_fs=sfreq)

    eeg, pres, n_trim = load_eeg_segments(cfg)
    n_pres = len(pres)
    ch_names = list(eeg.ch_names)

    X_dom,  _ = mgr.get_matrix(['ivc_dominant_hand'])
    X_base, _ = mgr.get_matrix(['motion_energy', 'velocity_stasis'])
    X_dom  = X_dom[:n_trim]
    X_base = X_base[:n_trim]

    band_names    = list(BANDS.keys())
    r_ivc_mat     = np.zeros((len(BANDS), len(ch_names)))
    r_base_mat    = np.zeros((len(BANDS), len(ch_names)))

    for bi, (band_name, (lo, hi)) in enumerate(BANDS.items()):
        print(f"  Band: {band_name.replace(chr(10), ' ')}", flush=True)
        if lo is None and hi is None:
            pres_band = pres
        else:
            pres_band = [bandpass_filter(P, lo, hi) for P in pres]
        r_ivc_mat[bi]  = run_cv_band(X_dom,  pres_band, n_pres, sfreq)
        r_base_mat[bi] = run_cv_band(X_base, pres_band, n_pres, sfreq)

    np.savez(bfile,
             ch_names=np.array(ch_names),
             band_names=np.array(band_names),
             r_ivc=r_ivc_mat,
             r_base=r_base_mat)
    print(f"  Saved → {bfile}")
    return np.load(bfile, allow_pickle=True)


def plot_bands(participant):
    data     = compute_and_save_bands(participant)
    ch_names = [str(c) for c in data['ch_names']]
    ch_ord, idx_ord = sort_channels(ch_names)
    f7_pos   = ch_ord.index('F7')
    o1_pos   = ch_ord.index('O1')
    band_names = list(BANDS.keys())
    r_ivc_mat  = data['r_ivc']
    r_base_mat = data['r_base']
    n_bands    = len(band_names)

    fig, axes = plt.subplots(2, n_bands, figsize=(16, 8), facecolor='#f8f8f8')
    fig.suptitle(f"Band-Filtered TRF: IVC Dominant Hand vs. Rokoko Kinematics — {participant}\n"
                 "Top row = IVC dominant hand  |  Bottom row = motion_energy + velocity_stasis",
                 fontsize=12, fontweight='bold', y=0.99)

    for bi, band_name in enumerate(band_names):
        r_ivc_b  = r_ivc_mat[bi][idx_ord]
        r_base_b = r_base_mat[bi][idx_ord]
        y_max    = max(r_ivc_b.max(), r_base_b.max(), 0.02) * 1.3
        y_min    = min(r_ivc_b.min(), r_base_b.min(), -0.01)
        x        = np.arange(len(ch_ord))

        for row, (r_vals, color) in enumerate([(r_ivc_b, ORANGE), (r_base_b, BLUE)]):
            ax  = axes[row, bi]
            bar_colors = [color if v >= 0 else '#aaaaaa' for v in r_vals]
            ax.bar(x, r_vals, color=bar_colors, alpha=0.85, zorder=3)
            ax.axhline(0, color='black', lw=0.7)
            for pi, patch in enumerate(ax.patches):
                if pi == f7_pos:
                    patch.set_edgecolor(BLUE);   patch.set_linewidth(2.0)
                elif pi == o1_pos:
                    patch.set_edgecolor(ORANGE); patch.set_linewidth(2.0)
            ax.set_ylim(y_min * 1.3, y_max)
            ax.set_xticks(x[::3])
            ax.set_xticklabels([ch_ord[i] for i in range(0, len(ch_ord), 3)],
                               rotation=45, ha='right', fontsize=7)
            ax.set_facecolor('white'); ax.grid(axis='y', alpha=0.2, zorder=0)
            if bi == 0:
                ax.set_ylabel(f"{'IVC dominant hand' if row==0 else 'Rokoko base'}\nPearson r",
                              fontsize=8)
            if row == 0:
                ax.set_title(band_name, fontsize=9, fontweight='bold')
            ax.text(f7_pos, r_vals[f7_pos] + y_max * 0.05, f'F7\n{r_vals[f7_pos]:.3f}',
                    ha='center', fontsize=6.5, color=BLUE, fontweight='bold')
            ax.text(o1_pos, r_vals[o1_pos] + y_max * 0.05, f'O1\n{r_vals[o1_pos]:.3f}',
                    ha='center', fontsize=6.5, color=ORANGE, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = f'results/{participant.lower()}_band_plot.png'
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#f8f8f8')
    print(f"Saved → {out}")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# PLOT: CROSS-PARTICIPANT COMPARISON
# ══════════════════════════════════════════════════════════════════════════════

def plot_comparison():
    mc = np.load(results_file_for('MC'), allow_pickle=True)
    eh = np.load(results_file_for('EH'), allow_pickle=True)
    mc_ch = [str(c) for c in mc['ch_names']]
    eh_ch = [str(c) for c in eh['ch_names']]

    mc_ord, mc_idx = sort_channels(mc_ch)
    eh_ord, eh_idx = sort_channels(eh_ch)
    mc_f7 = mc_ord.index('F7'); mc_o1 = mc_ord.index('O1')
    eh_f7 = eh_ord.index('F7'); eh_o1 = eh_ord.index('O1')

    fig = plt.figure(figsize=(18, 14), facecolor='#f8f8f8')
    gs  = GridSpec(3, 2, figure=fig, hspace=0.55, wspace=0.38,
                   left=0.07, right=0.97, top=0.92, bottom=0.07)

    # Panel A: 4-model × 4-series bar chart
    ax_a = fig.add_subplot(gs[0, :])
    models = ['Rokoko base', 'IVC global', 'IVC dominant hand', 'Bridge (IVC+Rokoko)']
    keys   = ['r_base', 'r_ivc', 'r_dom', 'r_bridge']
    x      = np.arange(len(models))
    w      = 0.18
    offsets = [-1.5, -0.5, 0.5, 1.5]
    series = [
        ('MC', 'F7', MC_C, '', mc, mc_ch),
        ('EH', 'F7', EH_C, '///', eh, eh_ch),
        ('MC', 'O1', ORANGE, '', mc, mc_ch),
        ('EH', 'O1', '#f4a582', '///', eh, eh_ch),
    ]
    for i, (pname, ch, color, hatch, d, chs) in enumerate(series):
        vals = [d[k][chs.index(ch)] for k in keys]
        label = f"{pname} — {ch}"
        ax_a.bar(x + offsets[i]*w, vals, w*0.9,
                 color=color, alpha=0.85, hatch=hatch, label=label, zorder=3)
        for j, v in enumerate(vals):
            yp = v + 0.003 if v >= 0 else v - 0.008
            ax_a.text(x[j] + offsets[i]*w, yp, f'{v:.3f}', ha='center',
                      va='bottom' if v >= 0 else 'top', fontsize=7, color=color, fontweight='bold')
    ax_a.axhline(0, color='black', lw=0.8, zorder=2)
    ax_a.set_xticks(x); ax_a.set_xticklabels(models, fontsize=11)
    ax_a.set_ylabel("Pearson r", fontsize=11)
    ax_a.set_title("A  |  MC vs E. Haggerty — F7 (frontal) and O1 (occipital)",
                   fontsize=12, fontweight='bold', loc='left', pad=8)
    ax_a.legend(fontsize=9, loc='upper right', ncol=2)
    ax_a.grid(axis='y', alpha=0.3, zorder=0); ax_a.set_facecolor('white')
    ax_a.set_ylim(-0.05, 0.15)

    # Panels B/C: Full channel profiles per participant
    for ax_idx, (participant, ch_ord, ch_idx, f7_i, o1_i, d, color) in enumerate([
        ('MC', mc_ord, mc_idx, mc_f7, mc_o1, mc, MC_C),
        ('EH', eh_ord, eh_idx, eh_f7, eh_o1, eh, EH_C),
    ]):
        ax = fig.add_subplot(gs[1, ax_idx])
        x_p = np.arange(len(ch_ord))
        base_ord = d['r_base'][ch_idx]
        dom_ord  = d['r_dom'][ch_idx]
        ax.fill_between(x_p, 0, base_ord, alpha=0.20, color=BLUE)
        ax.plot(x_p, base_ord, 'o-', color=BLUE, ms=4, lw=1.8, label='Rokoko base', zorder=4)
        ax.fill_between(x_p, 0, dom_ord, alpha=0.20, color=ORANGE)
        ax.plot(x_p, dom_ord, 's-', color=ORANGE, ms=4, lw=1.8, label='IVC dominant hand', zorder=4)
        ax.axhline(0, color='black', lw=0.8)
        ax.set_xticks(x_p); ax.set_xticklabels(ch_ord, rotation=45, ha='right', fontsize=7.5)
        ax.set_ylabel("Pearson r", fontsize=10)
        ax.set_title(f"{'B' if ax_idx==0 else 'C'}  |  {participant} — full channel profile",
                     fontsize=11, fontweight='bold', loc='left', pad=8)
        ax.legend(fontsize=8.5, loc='upper right')
        ax.grid(axis='y', alpha=0.25); ax.set_facecolor('white')
        ax.annotate(f'F7={base_ord[f7_i]:.3f}', xy=(f7_i, base_ord[f7_i]),
                    xytext=(f7_i-2, base_ord[f7_i]+0.012), fontsize=7.5, color=BLUE,
                    arrowprops=dict(arrowstyle='->', color=BLUE, lw=0.8))
        ax.annotate(f'O1={dom_ord[o1_i]:.3f}', xy=(o1_i, dom_ord[o1_i]),
                    xytext=(o1_i+1, dom_ord[o1_i]+0.015), fontsize=7.5, color=ORANGE,
                    arrowprops=dict(arrowstyle='->', color=ORANGE, lw=0.8))

    # Panel D: Unique IVC variance, both participants
    common = [c for c in CH_ORDER if c in mc_ord and c in eh_ord]
    mc_delta = np.array([mc['r_bridge'][mc_ch.index(c)] - mc['r_base'][mc_ch.index(c)] for c in common])
    eh_delta = np.array([eh['r_bridge'][eh_ch.index(c)] - eh['r_base'][eh_ch.index(c)] for c in common])
    x_d  = np.arange(len(common))
    ax_d = fig.add_subplot(gs[2, :])
    ax_d.bar(x_d - 0.2, mc_delta, 0.38,
             color=[MC_C if v >= 0 else '#aaaaaa' for v in mc_delta],
             alpha=0.85, label='MC', zorder=3)
    ax_d.bar(x_d + 0.2, eh_delta, 0.38,
             color=[EH_C if v >= 0 else '#cccccc' for v in eh_delta],
             alpha=0.85, hatch='///', label='E. Haggerty', zorder=3)
    ax_d.axhline(0, color='black', lw=0.8, zorder=2)
    ax_d.set_xticks(x_d); ax_d.set_xticklabels(common, rotation=45, ha='right', fontsize=8)
    ax_d.set_ylabel("Δr (bridge − base)", fontsize=10)
    ax_d.set_title("D  |  Unique variance added by IVC (bridge − Rokoko base)",
                   fontsize=11, fontweight='bold', loc='left', pad=8)
    ax_d.legend(fontsize=9, loc='upper left')
    ax_d.grid(axis='y', alpha=0.25); ax_d.set_facecolor('white')
    o1_pos = common.index('O1')
    ax_d.annotate(f'O1: MC Δr={mc_delta[o1_pos]:+.3f}',
                  xy=(o1_pos - 0.2, mc_delta[o1_pos]),
                  xytext=(o1_pos - 2.5, mc_delta[o1_pos] + 0.015),
                  fontsize=8, color=MC_C,
                  arrowprops=dict(arrowstyle='->', color=MC_C, lw=0.9))
    ax_d.annotate(f'O1: EH Δr={eh_delta[o1_pos]:+.3f}',
                  xy=(o1_pos + 0.2, eh_delta[o1_pos]),
                  xytext=(o1_pos + 1.5, eh_delta[o1_pos] + 0.018),
                  fontsize=8, color=EH_C,
                  arrowprops=dict(arrowstyle='->', color=EH_C, lw=0.9))

    fig.suptitle(
        "TRF Results: MC vs E. Haggerty — Same ASL Stimulus, Leave-One-Out CV\n"
        "Both participants: kinematics → F7 (frontal)  |  IVC dominant hand → O1 (occipital)",
        fontsize=13, fontweight='bold', y=0.975
    )
    plt.savefig('results/comparison_MC_EH.png', dpi=150, bbox_inches='tight', facecolor='#f8f8f8')
    print("Saved → results/comparison_MC_EH.png")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# PLOT: GROUP AVERAGE
# ══════════════════════════════════════════════════════════════════════════════

def plot_group():
    mc  = np.load(results_file_for('MC'), allow_pickle=True)
    eh  = np.load(results_file_for('EH'), allow_pickle=True)
    mc_ch = [str(c) for c in mc['ch_names']]
    eh_ch = [str(c) for c in eh['ch_names']]

    common = [c for c in CH_ORDER if c in mc_ch and c in eh_ch]
    mc_idx = [mc_ch.index(c) for c in common]
    eh_idx = [eh_ch.index(c) for c in common]
    f7_pos = common.index('F7')
    o1_pos = common.index('O1')

    def align(d, idx, key):
        return d[key][idx]

    parts = {
        'MC': {k: align(mc, mc_idx, k) for k in ['r_base','r_ivc','r_dom','r_bridge']},
        'EH': {k: align(eh, eh_idx, k) for k in ['r_base','r_ivc','r_dom','r_bridge']},
    }

    def gmean(key): return np.mean([parts['MC'][key], parts['EH'][key]], axis=0)
    def grange(key):
        arr = np.stack([parts['MC'][key], parts['EH'][key]])
        return arr.min(axis=0), arr.max(axis=0)

    # Figure 1: Accuracy
    fig1 = plt.figure(figsize=(18, 13), facecolor='#f8f8f8')
    gs   = GridSpec(3, 2, figure=fig1, hspace=0.55, wspace=0.38,
                    left=0.07, right=0.97, top=0.92, bottom=0.07)

    # Panel A: key model × channel, mean ± range + individual dots
    ax_a = fig1.add_subplot(gs[0, :])
    models_info = [('Rokoko base','r_base'),('IVC global','r_ivc'),
                   ('IVC dominant hand','r_dom'),('Bridge (IVC+Rokoko)','r_bridge')]
    x = np.arange(len(models_info))
    for i, (label, key) in enumerate(models_info):
        mc_f7 = parts['MC'][key][f7_pos]; eh_f7 = parts['EH'][key][f7_pos]
        mc_o1 = parts['MC'][key][o1_pos]; eh_o1 = parts['EH'][key][o1_pos]
        mn_f7 = (mc_f7 + eh_f7) / 2;     mn_o1 = (mc_o1 + eh_o1) / 2

        ax_a.bar(x[i] - 0.28, mn_f7, 0.18, color=BLUE,   alpha=0.95, zorder=4,
                 label='Mean F7' if i == 0 else '')
        ax_a.errorbar(x[i] - 0.28, mn_f7,
                      yerr=[[mn_f7 - min(mc_f7,eh_f7)],[max(mc_f7,eh_f7) - mn_f7]],
                      fmt='none', color='black', capsize=4, lw=1.5, zorder=5)
        ax_a.bar(x[i], mn_o1, 0.18, color=ORANGE, alpha=0.95, zorder=4,
                 label='Mean O1' if i == 0 else '')
        ax_a.errorbar(x[i], mn_o1,
                      yerr=[[mn_o1 - min(mc_o1,eh_o1)],[max(mc_o1,eh_o1) - mn_o1]],
                      fmt='none', color='black', capsize=4, lw=1.5, zorder=5)
        ax_a.plot(x[i] - 0.28, mc_f7, 'o', color=MC_C, ms=6, zorder=6, label='MC' if i==0 else '')
        ax_a.plot(x[i] - 0.28, eh_f7, 's', color=EH_C, ms=6, zorder=6, label='EH' if i==0 else '')
        ax_a.plot(x[i], mc_o1, 'o', color=MC_C, ms=6, zorder=6)
        ax_a.plot(x[i], eh_o1, 's', color=EH_C, ms=6, zorder=6)
        for val, xp, col in [(mn_f7, x[i]-0.28, BLUE), (mn_o1, x[i], ORANGE)]:
            yp = val + 0.004 if val >= 0 else val - 0.009
            ax_a.text(xp, yp, f'{val:.3f}', ha='center',
                      va='bottom' if val >= 0 else 'top', fontsize=7.5, color=col, fontweight='bold')

    ax_a.axhline(0, color='black', lw=0.8, zorder=2)
    ax_a.set_xticks(x - 0.14); ax_a.set_xticklabels([m[0] for m in models_info], fontsize=11)
    ax_a.set_ylabel("Pearson r", fontsize=11)
    ax_a.set_title("A  |  Group average (N=2) — mean ± range — F7 and O1\n"
                   "Error bars = min–max across participants; dots = individuals",
                   fontsize=11, fontweight='bold', loc='left', pad=8)
    ax_a.legend(fontsize=9, loc='upper right', ncol=4)
    ax_a.grid(axis='y', alpha=0.3, zorder=0); ax_a.set_facecolor('white')
    ax_a.set_ylim(-0.04, 0.16)

    # Panel B: Rokoko base group profile
    xc = np.arange(len(common))
    for ax_sub, key, color, title in [
        (gs[1, 0], 'r_base', BLUE,   "B  |  Rokoko base — group profile"),
        (gs[1, 1], 'r_dom',  ORANGE, "C  |  IVC dominant hand — group profile"),
    ]:
        ax = fig1.add_subplot(ax_sub)
        mn = gmean(key); lo, hi = grange(key)
        ax.fill_between(xc, lo, hi, alpha=0.18, color=color)
        ax.plot(xc, mn,              '-',  color=color, lw=2.2, label='Mean', zorder=4)
        ax.plot(xc, parts['MC'][key], '--', color=MC_C, lw=1.2, alpha=0.7, label='MC', zorder=3)
        ax.plot(xc, parts['EH'][key], ':',  color=EH_C, lw=1.2, alpha=0.7, label='EH', zorder=3)
        ax.axhline(0, color='black', lw=0.8)
        ax.set_xticks(xc); ax.set_xticklabels(common, rotation=45, ha='right', fontsize=7.5)
        ax.set_ylabel("Pearson r", fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold', loc='left', pad=8)
        ax.legend(fontsize=8.5); ax.grid(axis='y', alpha=0.25); ax.set_facecolor('white')

    # Panel D: Group unique IVC variance
    ax_d = fig1.add_subplot(gs[2, :])
    mc_delta = parts['MC']['r_bridge'] - parts['MC']['r_base']
    eh_delta = parts['EH']['r_bridge'] - parts['EH']['r_base']
    mn_delta = (mc_delta + eh_delta) / 2
    bar_cols = [GREEN if v >= 0 else '#aaaaaa' for v in mn_delta]
    ax_d.bar(xc, mn_delta, color=bar_cols, alpha=0.80, zorder=3, label='Mean Δr')
    ax_d.fill_between(xc, np.minimum(mc_delta, eh_delta), np.maximum(mc_delta, eh_delta),
                      alpha=0.25, color='grey', step='mid', label='Range (MC–EH)')
    ax_d.plot(xc, mc_delta, 'o--', color=MC_C, ms=4, lw=1.0, alpha=0.8, label='MC')
    ax_d.plot(xc, eh_delta, 's:',  color=EH_C, ms=4, lw=1.0, alpha=0.8, label='EH')
    ax_d.axhline(0, color='black', lw=0.8)
    ax_d.set_xticks(xc); ax_d.set_xticklabels(common, rotation=45, ha='right', fontsize=8)
    ax_d.set_ylabel("Δr (bridge − base)", fontsize=10)
    ax_d.set_title("D  |  Unique variance added by IVC — group mean ± range",
                   fontsize=11, fontweight='bold', loc='left', pad=8)
    ax_d.legend(fontsize=9, loc='upper left', ncol=4)
    ax_d.grid(axis='y', alpha=0.25); ax_d.set_facecolor('white')
    ax_d.annotate(f'O1 mean Δr={mn_delta[o1_pos]:+.3f}',
                  xy=(o1_pos, mn_delta[o1_pos]),
                  xytext=(o1_pos+1.5, mn_delta[o1_pos]+0.012),
                  fontsize=8.5, color=GREEN, fontweight='bold',
                  arrowprops=dict(arrowstyle='->', color=GREEN, lw=1.0))

    fig1.suptitle(
        "Group-Averaged TRF Results — MC + E. Haggerty (N=2)\n"
        "Double dissociation: kinematics → F7  |  IVC dominant hand → O1",
        fontsize=13, fontweight='bold', y=0.975
    )
    plt.savefig('results/group_avg_results.png', dpi=150, bbox_inches='tight', facecolor='#f8f8f8')
    print("Saved → results/group_avg_results.png")
    plt.close()

    # Figure 2: Group kernels
    mc_k = np.load(kernels_file_for('MC'), allow_pickle=True)
    eh_k = np.load(kernels_file_for('EH'), allow_pickle=True)
    mc_kch = [str(c) for c in mc_k['ch_names']]
    eh_kch = [str(c) for c in eh_k['ch_names']]
    mc_f7k = mc_kch.index('F7'); mc_o1k = mc_kch.index('O1')
    eh_f7k = eh_kch.index('F7'); eh_o1k = eh_kch.index('O1')

    kpanels = [
        ('k_dom',    'd_dom',    0, mc_o1k, eh_o1k, ORANGE, "IVC dominant hand → O1"),
        ('k_dom',    'd_dom',    0, mc_f7k, eh_f7k, ORANGE, "IVC dominant hand → F7"),
        ('k_base',   'd_base',   0, mc_f7k, eh_f7k, BLUE,   "motion_energy → F7"),
        ('k_base',   'd_base',   1, mc_f7k, eh_f7k, BLUE,   "velocity_stasis → F7"),
        ('k_bridge', 'd_bridge', 0, mc_f7k, eh_f7k, PURPLE, "Bridge: IVC dom → F7"),
        ('k_bridge', 'd_bridge', 1, mc_f7k, eh_f7k, GREEN,  "Bridge: motion_energy → F7"),
    ]

    fig2, axes2 = plt.subplots(2, 3, figsize=(15, 9), facecolor='#f8f8f8')
    fig2.suptitle("Group-Averaged TRF Kernels (N=2: MC + E. Haggerty)\n"
                  "Solid = mean  |  Shaded = range  |  Dashed/dotted = individuals",
                  fontsize=12, fontweight='bold', y=0.99)

    for idx, (kkey, dkey, feat, ch_mc, ch_eh, color, title) in enumerate(kpanels):
        row, col = divmod(idx, 3)
        ax       = axes2[row, col]
        d_ms     = mc_k[dkey] * 1000
        mc_t     = mc_k[kkey][ch_mc, feat, :]
        eh_t     = eh_k[kkey][ch_eh, feat, :]
        mean_t   = (mc_t + eh_t) / 2

        ax.fill_between(d_ms, np.minimum(mc_t, eh_t), np.maximum(mc_t, eh_t),
                        alpha=0.22, color=color)
        ax.plot(d_ms, mc_t,  '--', color=MC_C,  lw=1.3, alpha=0.75, label='MC')
        ax.plot(d_ms, eh_t,  ':',  color=EH_C,  lw=1.3, alpha=0.75, label='EH')
        ax.plot(d_ms, mean_t, '-', color=color, lw=2.2, label='Mean', zorder=5)
        ax.axhline(0, color='black', lw=0.8, linestyle='--', alpha=0.5)
        ax.axvline(0, color='grey',  lw=0.8, linestyle=':', alpha=0.5)
        ax.set_title(title, fontsize=9.5, fontweight='bold', pad=5)
        ax.set_xlabel("Time lag (ms)", fontsize=8.5)
        ax.set_ylabel("Weight (a.u.)", fontsize=8.5)
        ax.set_facecolor('white'); ax.grid(alpha=0.2)
        peak_idx = np.argmax(np.abs(mean_t))
        ax.annotate(f'peak\n{d_ms[peak_idx]:.0f}ms',
                    xy=(d_ms[peak_idx], mean_t[peak_idx]),
                    xytext=(d_ms[peak_idx] + 40, mean_t[peak_idx] * 0.80),
                    fontsize=7.5, color=color,
                    arrowprops=dict(arrowstyle='->', color=color, lw=0.8))
        if idx == 0:
            ax.legend(fontsize=8, loc='upper right')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('results/group_avg_kernels.png', dpi=150, bbox_inches='tight', facecolor='#f8f8f8')
    print("Saved → results/group_avg_kernels.png")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="TRF results plotter — sign language EEG pipeline"
    )
    parser.add_argument(
        '--participant', '-p',
        choices=['MC', 'EH', 'both'],
        default='MC',
        help="Which participant(s) to plot. 'both' needed for comparison/group plots."
    )
    parser.add_argument(
        '--plot', '-t',
        choices=['results', 'kernels', 'bands', 'comparison', 'group', 'all'],
        default='all',
        help="Which figure(s) to generate."
    )
    args = parser.parse_args()

    participants = ['MC', 'EH'] if args.participant == 'both' else [args.participant]
    want_results    = args.plot in ('results',    'all')
    want_kernels    = args.plot in ('kernels',    'all')
    want_bands      = args.plot in ('bands',      'all')
    want_comparison = args.plot in ('comparison', 'all')
    want_group      = args.plot in ('group',      'all')

    for p in participants:
        if want_results:
            print(f"\n── Results summary: {p} ──")
            plot_results(p)
        if want_kernels:
            print(f"\n── Kernels: {p} ──")
            plot_kernels(p)
        if want_bands:
            print(f"\n── Band analysis: {p} ──")
            plot_bands(p)

    if want_comparison:
        if len(participants) < 2:
            print("NOTE: comparison plot requires --participant both")
        else:
            print("\n── Cross-participant comparison ──")
            plot_comparison()

    if want_group:
        if len(participants) < 2:
            print("NOTE: group plot requires --participant both")
        else:
            print("\n── Group average ──")
            plot_group()

    print("\nDone.")


if __name__ == '__main__':
    main()
