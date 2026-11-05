# Interview Plot Guide



## Plot 1 — ivc_results_summary.png

This is a 4-panel figure. Walk through it top to bottom:

**Panel A — Double dissociation (top left bar chart)**
- X-axis: the 4 models tested (Rokoko kinematics, IVC global pixel motion, IVC dominant hand only, Bridge combining both)
- Y-axis: Pearson r — how well the model predicted the EEG signal (higher = better)
- Two bars per model: blue = F7 (left frontal channel), red = O1 (left occipital channel)
- Key point to say: *"Rokoko kinematics predict F7 strongly (r=0.107) but barely touch O1 (r=0.004). IVC dominant hand flips this — O1 jumps to r=0.041 while F7 drops. That's the double dissociation — different feature types driving different brain regions."*

**Panel B — Spatial IVC regions (top right bar chart)**
- X-axis: which part of the video frame we computed motion from — face, dominant hand, non-dominant hand, all regions combined
- Y-axis: Pearson r at F7 and O1
- Key point: *"The dominant hand region alone drives the occipital response. Face and non-dominant hand contribute much less. This tells us the brain is specifically tracking the primary signing hand visually."*

**Panel C — All 29 channels (middle line chart)**
- X-axis: all 29 EEG electrode positions, arranged roughly front to back (frontal left → temporal/central → parietal → occipital)
- Y-axis: Pearson r per channel
- Three lines: blue = Rokoko, red = IVC dominant hand, green = IVC global
- Key point: *"Rokoko (blue) peaks at frontal-left channels and drops toward the back. IVC dominant hand (red) does the opposite — it's weak frontally but rises toward occipital. The two models have opposite spatial profiles, confirming the dissociation isn't just a fluke at two channels."*

**Panel D — Unique variance (bottom bar chart)**
- X-axis: all 29 channels again
- Y-axis: Δr — the improvement in prediction when you add IVC on top of Rokoko kinematics alone
- Red bars = IVC adds signal, grey bars = IVC hurts slightly
- Key point: *"Adding IVC to the kinematic model gives a unique boost specifically at occipital channels. That means IVC is capturing something about visual cortex that the motion capture data doesn't explain."*

---

## Plot 2 — topomap_prediction_accuracy.png
**Pull up when explaining where on the scalp the model works.**

This is a top-down view of the head (nose at top, left ear on the right side of the diagram — standard EEG convention).

- Color scale: red = high Pearson r (model predicts well), blue = low or negative r (noise)
- Left topomap: raw r values across all channels
- Right topomap: same but only showing positive predictions (clipped at zero)
- Key point to say: *"The strongest prediction is at F7 — left frontal-temporal — which is associated with language and motor planning for speech. The central and parietal regions are near zero, and some parietal channels actually go negative which means those electrodes are noisy. The spatial pattern matches what we'd expect from ASL processing literature."*

---

## Plot 3 — permutation_topomap.png
**Pull up if they ask "how do you know the results aren't just noise?"**

Same head topography view, now colored by prediction accuracy (red = high r) with white circles marking statistically significant channels.

- Left map: channels significant at p < 0.05 — 7 out of 29 channels marked
- Right map: channels significant at p < 0.01 — 5 out of 29 channels marked
- How significance was computed: 200 permutations where the feature matrix was time-shifted by a random amount (minimum 10 seconds) to break the true stimulus-EEG alignment. The real r has to beat 95% or 99% of those shuffled values.
- Key point: *"Circular time-shift permutation testing is the standard approach for continuous EEG — you can't just shuffle individual samples because the signal is correlated in time. By shifting the whole feature stream by 10+ seconds you destroy the real relationship while preserving the temporal autocorrelation structure. The frontal-left cluster survives at p<0.01, which is a strong result for continuous naturalistic EEG."*

---

## Plot 4 — ivc_kernels_plot.png
**Pull up if they ask "what does the TRF actually look like" or "what's the timing?"**

6 subplots showing the learned temporal filter (kernel) for each feature-to-channel pair.

- X-axis on every plot: time lag in milliseconds (0 to 500ms) — this is how long after a motion event the brain responds
- Y-axis: kernel weight in arbitrary units — the magnitude of the brain's response at each lag
- Top row: IVC dominant hand → F7, motion energy → F7, Bridge IVC → F7
- Bottom row: IVC dominant hand → O1, velocity stasis → F7, Bridge motion energy → F7

Key things to point out:
- *"Motion energy peaks at 250ms — a broad, sustained response to continuous movement at the frontal channel. That's consistent with higher-level motor/language processing."*
- *"IVC dominant hand peaks at ~200ms at both F7 and O1. The negative weight means the brain suppresses its response when the dominant hand is moving fast — which makes sense as a predictive coding signature."*
- *"Velocity stasis peaks immediately at 0ms and stays negative — the brain reacts to stillness almost instantly, possibly reflecting anticipation of an upcoming sign."*

---

## Plot 5 — comparison_MC_EH.png
**Pull up if they ask about generalizability or sample size.**

Results from both participants (MC and E. Haggerty) side by side.

**Panel A — top bar chart:**
- Same 4 models on the X-axis
- Blue/red solid bars = MC, blue/red hatched bars = E. Haggerty
- Key point: *"Both participants show the same pattern — kinematics dominate at F7, IVC dominant hand is stronger at O1. The effect replicates across individuals even with N=2."*

**Panel B & C — full channel profiles:**
- Left: MC's r values across all channels for Rokoko vs IVC dominant hand
- Right: E. Haggerty's same comparison
- Key point: *"The spatial profiles look similar — frontal peak for kinematics, occipital rise for IVC — though E. Haggerty shows more distributed IVC signal, which is consistent with some individual variability in cortical organization."*

**Panel D — unique variance by participant:**
- Grey bars = MC, hatched red = E. Haggerty
- Shows IVC's added value over kinematics at each channel
- Key point: *"E. Haggerty shows stronger unique IVC variance at O1 (Δr=+0.030 vs MC's +0.006), suggesting her visual cortex is more strongly tracking the dominant hand. This kind of individual difference is interesting and something we'd want to explore with more participants."*
