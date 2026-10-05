"""
MUAP-based surface-EMG decomposition.

Implementation of Chen, Li & Xia (2025), "A motor unit action potential-based
method for surface electromyography decomposition", J. NeuroEng. Rehabil.
22:60 (https://doi.org/10.1186/s12984-025-01595-y).

The method has two stages (their Algorithm 1):

  Stage 1 -- RECONSTRUCT MU FILTERS (from a training signal x1)
    1. Pre-process x1 (band-pass 20-500 Hz, 50 Hz comb, extend).
    2. Decompose x1 with gCKC -> MUSTs Θ = [θ1, ..., θJ].
    3. Estimate each multichannel MUAP by STA (62.5 ms / 128 samples/channel).
    4. Extract a K-sample window per channel (Centered Peak) and cascade ->
       reconstructed MU filter c_θjx̄.
    5. Stack the filters into C_Θx̄.

  Stage 2 -- DECOMPOSE NEW DATASET (test signal x2)
    6. Pre-process and extend x2 by K; build the (Current) covariance C_x̄x̄.
    7. (refine/iteration modes) re-optimise the filters on x2.
    8. LMMSE pulse-train estimate per filter (Eq. 3).
    9. K-means on θ̂j -> discharge times; remove duplicates / bad sources.

Defaults reproduce the settings the paper reported as most effective:
Centered-Peak reconstruction, Current C_xx, STA extraction, extension factor
K = 40, 128-sample MUAPs, and MUAP-refine MUST estimation.

The class mirrors ``CBSS``: ``__init__(config=None, **kwargs)``, ``set_param``,
and a ``decompose`` that returns the same kind of 6-tuple
``(sources, spikes, sil, mu_filters, muaps, centroids)`` (``muaps`` takes the
slot CBSS used for the whitening matrix Z, which is not needed here).
"""

import os

import numpy as np

from .core import (
    bandpass_signals,
    extension,
    notch_signals,
    est_spike_times,
    remove_bad_sources,
    remove_duplicates,
)
from .muap_core import (
    compute_cov_inverse,
    estimate_stim_template,
    gckc,
    least_squares_muap,
    lmmse_source,
    pca_smooth_muap,
    reconstruct_mu_filter,
    remove_stim_artifact,
    spike_triggered_average,
    _cov_isi,
    _spike_change,
)

_EPS = 1e-12


# --------------------------------------------------------------------------- #
# Diagnostics for blanked / stimulation data                                  #
# --------------------------------------------------------------------------- #
def _dbg_detect_blanks(sig, tol=1e-8):
    """Boolean valid-mask from an all-channel near-zero test (True=signal)."""
    energy = np.abs(np.asarray(sig)).sum(axis=0)
    return energy > tol


def _dbg_edges(valid):
    d = np.diff(valid.astype(int))
    starts = np.where(d == -1)[0] + 1          # signal -> blank (blank onset)
    ends = np.where(d == 1)[0] + 1             # blank -> signal
    edges = (np.sort(np.concatenate([starts, ends]))
             if (starts.size or ends.size) else np.array([], dtype=int))
    return edges, starts


def _dbg_edge_fraction(spk, edges, guard):
    spk = np.asarray(spk)
    if spk.size == 0 or edges.size == 0:
        return 0.0
    nearest = np.min(np.abs(spk[:, None] - edges[None, :]), axis=1)
    return float(np.mean(nearest <= guard))


def _dbg_isi_comb(spk, fsamp, stim_period, tol_frac=0.15):
    """Fraction of ISIs near an integer multiple of the stim period; ISIs (ms)."""
    spk = np.sort(np.asarray(spk))
    if spk.size < 3 or not stim_period:
        return 0.0, np.array([])
    isi = np.diff(spk)
    mult = isi / stim_period
    near = (np.abs(mult - np.round(mult)) < tol_frac) & (np.round(mult) >= 1)
    return float(np.mean(near)), isi / fsamp * 1000.0


def _dbg_pairwise_agreement(spikes, n_samp, fsamp, max_shift_s=0.1, tol_s=0.002):
    n = len(spikes)
    tol = max(1, int(tol_s * fsamp))
    max_shift = int(max_shift_s * fsamp)
    keys = sorted(spikes)
    sp = [np.asarray(spikes[k], int) for k in keys]
    M = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            si, sj = sp[i], sp[j]
            if si.size == 0 or sj.size == 0:
                continue
            best = 0.0
            for lag in range(-max_shift, max_shift + 1, tol):
                d = np.min(np.abs(si[:, None] - (sj[None, :] + lag)), axis=1)
                m = int(np.sum(d <= tol))
                roa = m / (si.size + sj.size - m)
                if roa > best:
                    best = roa
            M[i, j] = M[j, i] = best
    return M


class MUAPDecomposition:
    """MUAP-based convolution-kernel-compensation EMG decomposition."""

    def __init__(self, config=None, **kwargs):
        # --- Pre-processing ------------------------------------------------
        self.apply_bandpass = True
        self.bandpass = [20, 500]
        self.bandpass_order = 4
        self.apply_notch = True
        self.notch_frequency = 50
        self.notch_n_harmonics = 3
        self.notch_order = 2
        self.notch_width = 1

        # --- Extension / covariance ---------------------------------------
        self.ext_fact = 40           # K, paper-optimal (Fig. 7 peak)
        self.cov_reg = "auto"
        self.cxx_mode = "current"    # 'current' | 'raw' | 'global'

        # --- MUAP estimation & filter reconstruction ----------------------
        self.muap_duration = 128     # N samples = 62.5 ms @ 2048 Hz (Eq. 7)
        self.muap_extraction = "STA"  # 'STA' | 'LS'
        self.use_pca = False         # STA+PCA variant
        self.pca_var = 0.9
        self.recon_method = "centered_peak"  # paper-recommended

        # --- Stage 1 (gCKC) ------------------------------------------------
        self.stage1_max_mu = 100
        self.stage1_max_iter = 45
        self.stage1_min_spikes = 5
        self.contrast = "square"
        # seed for each MU search:
        #   'activity_idx' (argmax γ, paper) | 'biased_random' (γ-weighted draw)
        #   | 'random' (random unit vector, CBSS-style)
        self.opt_initialization = "activity_idx"
        # peel-off: subtract each accepted train from x̄ so a re-seed can't land
        # on an already-extracted unit (costs one Cinv@x̄ matmul per accepted MU)
        self.peel_off = False
        self.peel_win = 0.04          # s, STA window for the peeled component
        # peel-off refreshes Cx = Cinv@x̄ incrementally (Cx -= ΔCx); recompute it
        # exactly every N peels to bound float drift (0 = never, pure incremental)
        self.peel_recompute_every = 2
        # train-side dedup: reject a quality-passing source that duplicates an
        # already-accepted train (uses match_th/match_max_shift/match_tol below)
        self.train_dedup = True
        # dominating-attractor peel: a short (1-2 spike) reject location that the
        # search re-converges onto this many times is peeled from x̄ (periodic
        # artifact guard). inf disables; finite values require Cinv.
        self.recur_peel_th = float("inf")

        # --- Stage 2 (MUST estimation) ------------------------------------
        self.decomp_mode = "refine"   # 'direct' | 'iteration' | 'refine'
        self.refine_max_iter = 100
        self.iter_tol = 0.01          # 1% MUST change stopping criterion
        self.grad_lr = 0.005           # learning rate for 'iteration' (Eq. 6)

        # --- Spike extraction / quality -----------------------------------
        self.cluster_method = "kmeans"
        self.sil_th = 0.9
        self.cov_th = 0.35
        self.min_firing_rate = 0.5    # Hz

        # --- Post-processing ----------------------------------------------
        self.match_th = 0.3           # RoA duplicate threshold
        self.match_max_shift = 0.1
        self.match_tol = 0.001

        self.random_seed = 1909
        self.verbose_mode = True

        # --- Stimulation-artifact removal (test signal only) --------------
        # Experimental paradigm: the TEST signal has an electrical-stimulation
        # artifact superimposed on the EMG. Supply a separate recording that
        # contains ONLY the stimulation (the ``stim_sig`` arg of decompose) to
        # estimate its multichannel template, then detect (matched filter,
        # jitter-tolerant) and subtract it from the RAW test signal BEFORE
        # band-pass/notch (avoids filtfilt ringing across the sharp edges).
        # ``stim_hz`` (below) is the frequency prior; if None it is inferred.
        self.remove_stim_artifact = False
        self.stim_template_win = 0.015   # s, STA window for the stim waveform
        self.stim_scale_per_pulse = True  # LS-scale the template per pulse
        self.stim_align_jitter = 2       # +/- samples re-alignment per pulse
        self.stim_subsample_align = True  # parabolic sub-sample alignment
        self.stim_refractory_frac = 0.7  # min pulse spacing = frac * period
        self.stim_height_frac = 0.3      # keep peaks >= frac * max(score)
        self.stim_blank = False          # blank (replace) windows vs. subtract
        self.stim_blank_fill = "linear"  # "linear" | "zero" fill for blanking
        self.stim_blank_pad = 0          # widen each blanked window by N samples

        # --- Diagnostics (blanking / comb investigation) ------------------
        self.debug = True              # write debug plots + verbose stats
        self.debug_dir = "muap_debug"
        self.debug_max_units = 6        # units shown in detailed plots
        self.blank_detect_tol = 1e-8    # all-channel |.|-sum below this == blank
        self.stim_hz = None             # if known; else inferred from blanks

        # Merge config object + explicit kwargs (CBSS pattern) -------------
        # These keys are consumed by the decompose_muap wrapper, not the class:
        #   sampling_frequency -> fsamp arg of decompose()
        #   start_time/end_time -> time-window crop before decomposition
        reserved = {"sampling_frequency", "start_time", "end_time"}
        config_dict = vars(config) if config is not None else {}
        params = {**config_dict, **kwargs}
        valid_keys = self.__dict__.keys()
        for key, value in params.items():
            if key in valid_keys:
                setattr(self, key, value)
            elif key in reserved:
                continue
            else:
                print(f"Warning: ignoring invalid parameter: {key}")

    def set_param(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"Invalid parameter: {key}")

    # --------------------------------------------------------------------- #
    # Pre-processing                                                        #
    # --------------------------------------------------------------------- #
    def _preprocess(self, sig, fsamp):
        sig = np.asarray(sig, dtype=float)
        if self.apply_bandpass and self.bandpass is not None:
            sig = bandpass_signals(
                sig, fsamp,
                high_pass=self.bandpass[0],
                low_pass=self.bandpass[1],
                order=self.bandpass_order,
            )
        if self.apply_notch and self.notch_frequency is not None:
            sig = notch_signals(
                sig, fsamp,
                nfreq=self.notch_frequency,
                dfreq=self.notch_width,
                order=self.notch_order,
                n_harmonics=self.notch_n_harmonics,
            )
        return sig

    def _extend(self, sig):
        ext = extension(sig, self.ext_fact)
        ext = ext - np.mean(ext, axis=1, keepdims=True)
        # zero the edges, as in the CBSS pipeline
        ext[:, : self.ext_fact * 2] = 0
        ext[:, -self.ext_fact * 2:] = 0
        return ext

    # --------------------------------------------------------------------- #
    # Stage 1: reconstruct MU filters from a training signal                #
    # --------------------------------------------------------------------- #
    def reconstruct_filters(self, train_sig, fsamp, train_spikes=None):
        """Return (mu_filters, muaps, n_units, train_results) from training EMG.

        ``mu_filters`` has shape (n_channels*K, n_units); ``muaps`` is a list of
        (n_channels, N) arrays. ``train_results`` is a dict mirroring the test
        outputs for the Stage-1 (training) decomposition::

            {sources, spikes, silhouette, cov, mu_filters, muaps, centroids}

        where ``mu_filters`` here are the gCKC separation filters that produced
        ``sources`` (length n_channels*K each), not the reconstructed bank.

        ``train_spikes`` optionally supplies pre-computed discharge times (an
        iterable of per-unit sample-index arrays, e.g. the values of a CBSS
        ``results["spikes"]`` dict). When given, the gCKC search is skipped and
        these spike trains drive STA / filter reconstruction directly. The
        indices must reference the SAME (cropped) timeline as ``train_sig``.
        In this path there are no gCKC separation filters or quality metrics, so
        ``train_results``'s ``sources``/``mu_filters`` are empty and
        ``silhouette``/``cov`` are NaN.
        """
        pp = self._preprocess(train_sig, fsamp)
        n_ch = pp.shape[0]
        x_ext = self._extend(pp)

        if train_spikes is None:
            Cinv = compute_cov_inverse(x_ext, reg=self.cov_reg)
            Cx = Cinv @ x_ext

            if self.verbose_mode:
                print(f"\n{'='*80}\nSTAGE 1: gCKC decomposition of training signal"
                      f"\n{'='*80}")

            spike_trains, sil_list, cov_list, filt_list = gckc(
                x_ext, Cx, fsamp,
                n_mu=self.stage1_max_mu,
                max_iter=self.stage1_max_iter,
                tol=self.iter_tol,
                contrast = self.contrast,
                sil_th=self.sil_th,
                cov_th=self.cov_th,
                min_firing_rate=self.min_firing_rate,
                min_spikes=self.stage1_min_spikes,
                cluster=self.cluster_method,
                opt_init=self.opt_initialization,
                peel_off=self.peel_off,
                peel_win=self.peel_win,
                recur_peel_th=self.recur_peel_th,
                peel_recompute_every=self.peel_recompute_every,
                dedup=self.train_dedup,
                dedup_th=self.match_th,
                dedup_max_shift=self.match_max_shift,
                dedup_tol=self.match_tol,
                Cinv=Cinv,
                verbose=self.verbose_mode,
            )
        else:
            # Supplied discharge times: skip gCKC, go straight to STA/recon.
            spike_trains = [np.asarray(s, dtype=int) for s in train_spikes]
            sil_list = [float("nan")] * len(spike_trains)
            cov_list = [float("nan")] * len(spike_trains)
            filt_list = []  # no gCKC separation vectors in this path
            if self.verbose_mode:
                print(f"\n{'='*80}\nSTAGE 1: using {len(spike_trains)} supplied "
                      f"spike train(s); skipping gCKC\n{'='*80}")

        if self.verbose_mode:
            print(f"\nSTAGE 1: estimating MUAPs ({self.muap_extraction}) and "
                  f"reconstructing filters ({self.recon_method})")

        if self.muap_extraction.upper() == "LS":
            muaps = least_squares_muap(x_ext, spike_trains, n_ch,
                                       n_samples=self.muap_duration)
        else:
            muaps = [spike_triggered_average(pp, spk, self.muap_duration)
                     for spk in spike_trains]

        filters = []
        kept_muaps = []
        for muap in muaps:
            if self.use_pca:
                muap = pca_smooth_muap(muap, self.pca_var)
            f = reconstruct_mu_filter(muap, self.ext_fact, self.recon_method)
            filters.append(f)
            kept_muaps.append(muap)

        # --- Stage-1 results, mirroring the test-side outputs --------------
        # ``filt_list`` is empty when spike trains were supplied (no gCKC), so
        # train sources / mu_filters are left empty in that path.
        if filt_list:
            train_sources = np.stack([f @ Cx for f in filt_list], axis=0)
        else:
            train_sources = np.zeros((0, x_ext.shape[1]))
        train_spikes = {i: np.asarray(spk, dtype=int)
                        for i, spk in enumerate(spike_trains)}
        train_centroids = {}
        for i in range(train_sources.shape[0]):
            _, _, cent = est_spike_times(train_sources[i], fsamp,
                                         cluster=self.cluster_method)
            train_centroids[i] = cent
        train_results = {
            "sources": train_sources,
            "spikes": train_spikes,
            "silhouette": np.asarray(sil_list, dtype=float),
            "cov": np.asarray(cov_list, dtype=float),
            "mu_filters": (np.stack(filt_list, axis=1) if filt_list
                           else np.zeros((x_ext.shape[0], 0))),
            "muaps": kept_muaps,
            "centroids": train_centroids,
        }

        if self.debug:
            self._plot_stage1(pp, train_spikes, sil_list, kept_muaps, fsamp)

        if len(filters) == 0:
            return np.zeros((n_ch * self.ext_fact, 0)), [], 0, train_results

        mu_filters = np.stack(filters, axis=1)  # (n_ch*K, n_units)
        return mu_filters, kept_muaps, mu_filters.shape[1], train_results

    # --------------------------------------------------------------------- #
    # Stage 2 helpers                                                       #
    # --------------------------------------------------------------------- #
    def _covariance_inverse(self, train_ext, test_ext):
        mode = self.cxx_mode.lower()
        if mode == "raw":
            data = train_ext
        elif mode == "global":
            data = np.concatenate([train_ext, test_ext], axis=1)
        else:  # 'current'
            data = test_ext
        return compute_cov_inverse(data, reg=self.cov_reg)

    def _estimate_one(self, c0, x_ext, Cx, fsamp):
        """Apply / optimise a single MU filter on the test data.

        Returns (filter, source, spikes, sil, centroids).
        """
        c = c0 / (np.linalg.norm(c0) + _EPS)
        source = lmmse_source(c, Cx)
        spikes, sil, cent = est_spike_times(source, fsamp,
                                            cluster=self.cluster_method)

        if self.decomp_mode == "direct":
            return c, source, spikes, sil, cent

        stable = 0
        for _ in range(self.refine_max_iter):
            prev = spikes
            if len(spikes) == 0:
                break
            sp = np.asarray(spikes, dtype=int)
            if self.decomp_mode == "refine":
                # Eq. 4: c_θx̄ = E(θ x̄) on the pulses from the last iteration
                c = x_ext[:, sp].mean(axis=1)
            elif self.decomp_mode == "iteration":
                # Eq. 6: natural-gradient update, ∂f/∂x = t^2 contrast
                c = c + self.grad_lr * (x_ext * (source ** 2)).mean(axis=1)
            else:
                raise ValueError(f"Unknown decomp_mode: {self.decomp_mode}")
            c /= np.linalg.norm(c) + _EPS

            source = lmmse_source(c, Cx)
            spikes, sil, cent = est_spike_times(source, fsamp,
                                                cluster=self.cluster_method)
            if _spike_change(prev, spikes) < self.iter_tol:
                stable += 1
                if stable >= 2:
                    break
            else:
                stable = 0
        return c, source, spikes, sil, cent

    # --------------------------------------------------------------------- #
    # Stage-1 diagnostics                                                    #
    # --------------------------------------------------------------------- #
    def _plot_stage1(self, pp, spikes, sil, muaps, fsamp):
        """Plot the Stage-1 (training) results: discharge times over the
        (pre-processed) training EMG and the MUAP templates of the accepted
        motor units. ``pp`` is the pre-processed train signal (n_ch, n_samp);
        ``spikes`` is the {unit: indices} dict, ``muaps`` the per-unit
        (n_ch, N) templates.
        """
        try:
            import matplotlib
            # matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as e:
            print(f"[DEBUG] matplotlib unavailable ({e}); skipping Stage-1 plots")
            return

        n = len(spikes)
        if n == 0:
            print("[DEBUG] Stage-1: no accepted MUs to plot")
            return

        os.makedirs(self.debug_dir, exist_ok=True)
        pp = np.asarray(pp)
        T = pp.shape[1]
        t = np.arange(T) / fsamp
        env = np.abs(pp).sum(0)
        cmap = plt.get_cmap("tab10")

        # --- discharge times over the EMG envelope -------------------------
        fig, (ax0, ax1) = plt.subplots(
            2, 1, figsize=(12, 5), sharex=True,
            gridspec_kw={"height_ratios": [2, 1]})
        ax0.plot(t, env, "k", lw=0.4, alpha=0.8)
        ax0.set_ylabel("|EMG| sum")
        ax0.set_title("STAGE 1 (train): EMG envelope")
        positions = [np.asarray(spikes[u], int) / fsamp for u in range(n)]
        colors = [cmap(u % 10) for u in range(n)]
        ax1.eventplot(positions, colors=colors,
                      lineoffsets=np.arange(n), linelengths=0.8)
        ax1.set_yticks(np.arange(n))
        ax1.set_yticklabels([f"MU{u} sil={sil[u]:.2f}" for u in range(n)],
                            fontsize=6)
        ax1.set_xlabel("s")
        ax1.set_ylabel("MU")
        ax1.set_title("discharge times")
        fig.tight_layout()
        fig.savefig(f"{self.debug_dir}/train_01_spikes_over_emg.png", dpi=110)
        plt.close(fig)

        # --- MUAP templates: one figure per accepted unit ------------------
        for u in range(n):
            muap = np.asarray(muaps[u])
            n_ch, N = muap.shape
            tt = (np.arange(N) - N // 2) / fsamp * 1000.0
            step = float(np.abs(muap).max()) * 1.5 + _EPS
            fig, a = plt.subplots(figsize=(5, max(3, 0.35 * n_ch)))
            for ch in range(n_ch):
                a.plot(tt, muap[ch] - ch * step, color=cmap(u % 10), lw=0.6)
            a.set_title(f"STAGE 1 (train): MU{u} MUAP "
                        f"({n_ch} ch, sil={sil[u]:.2f})", fontsize=9)
            a.set_xlabel("ms")
            a.set_yticks([])
            fig.tight_layout()
            fig.savefig(f"{self.debug_dir}/train_02_template_MU{u:02d}.png",
                        dpi=110)
            plt.close(fig)

        print(f"[DEBUG] Stage-1 plots written to {self.debug_dir}/")

    # --------------------------------------------------------------------- #
    # Stim-artifact diagnostics                                              #
    # --------------------------------------------------------------------- #
    def _plot_stim_removal(self, raw, clean, template, times, fsamp):
        """Plot the before/after test signal on the channel the stim dominates
        (raw vs cleaned trace + the removed component), the cross-channel
        envelope, and the estimated multichannel stim template.
        """
        try:
            import matplotlib.pyplot as plt
        except Exception as e:
            print(f"[DEBUG] matplotlib unavailable ({e}); skipping stim plots")
            return

        os.makedirs(self.debug_dir, exist_ok=True)
        raw = np.asarray(raw, float)
        clean = np.asarray(clean, float)
        template = np.asarray(template, float)
        n_ch, T = raw.shape
        N = template.shape[1]
        t = np.arange(T) / fsamp
        times = np.asarray(times, int)

        # channel the stim artifact is largest on (clearest before/after)
        ch = int(np.argmax(np.max(np.abs(template), axis=1)))

        # window spanning the first few detected pulses (fall back: first 0.5 s)
        pad = int(0.05 * fsamp)
        if times.size:
            a = max(0, times[0] - pad)
            b = min(T, times[min(times.size, 8) - 1] + N + pad)
        else:
            a, b = 0, min(T, int(0.5 * fsamp))
        win = slice(a, b)
        tin = times[(times >= a) & (times < b)]

        # --- before / after signal traces in separate panels ---------------
        # Top: raw (before). Bottom: cleaned (after). Shared x and y so the
        # amplitude scale is directly comparable between the two.
        fig, (a0, a1) = plt.subplots(2, 1, figsize=(12, 6),
                                     sharex=True, sharey=True)
        fig.suptitle(f"Test signal before/after stim removal -- ch {ch} "
                     f"({len(times)} pulses detected)")
        a0.plot(t, raw[ch, :], "k", lw=0.6, label="raw")
        if times.size:
            a0.scatter(times / fsamp, raw[ch, times], c="r", s=14, zorder=5,
                       label="detected pulse")
        a0.set_ylabel(f"ch {ch}")
        a0.set_title("Before -- raw", fontsize=9)
        a0.legend(loc="upper right", fontsize=8)
        a1.plot(t, clean[ch, :], "b", lw=0.6, label="stim removed")
        if times.size:
            a1.scatter(times / fsamp, clean[ch, times], c="r", s=14, zorder=5)
        a1.set_ylabel(f"ch {ch}")
        a1.set_title("After -- stim removed", fontsize=9)
        a1.set_xlabel("s")
        fig.tight_layout()
        fig.savefig(f"{self.debug_dir}/stim_01_removal.png", dpi=110)
        plt.close(fig)

        # --- cross-channel envelope before/after ---------------------------
        fig, (e0, e1) = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
        e0.plot(t[win], np.abs(raw[:, win]).sum(0), "k", lw=0.5)
        if tin.size:
            e0.scatter(tin / fsamp, np.abs(raw[:, tin]).sum(0),
                       c="r", s=12, zorder=5)
        e0.set_title("Test |EMG| sum -- raw")
        e1.plot(t[win], np.abs(clean[:, win]).sum(0), "b", lw=0.5)
        e1.set_title("Test |EMG| sum -- after stim removal")
        e1.set_xlabel("s")
        fig.tight_layout()
        fig.savefig(f"{self.debug_dir}/stim_02_envelope.png", dpi=110)
        plt.close(fig)

        # --- multichannel template -----------------------------------------
        tt = (np.arange(N) - N // 2) / fsamp * 1000.0
        step = float(np.abs(template).max()) * 1.5 + _EPS
        fig, ax = plt.subplots(figsize=(5, max(3, 0.35 * n_ch)))
        for c in range(n_ch):
            ax.plot(tt, template[c] - c * step, "g", lw=0.6)
        ax.set_title(f"Stim template ({n_ch} ch, {N} samp)", fontsize=9)
        ax.set_xlabel("ms")
        ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(f"{self.debug_dir}/stim_03_template.png", dpi=110)
        plt.close(fig)
        print(f"[DEBUG] Stim-removal plots written to {self.debug_dir}/")

    # --------------------------------------------------------------------- #
    # Full pipeline                                                         #
    # --------------------------------------------------------------------- #
    def _diagnostics(self, raw_test, sources, spikes, sil, fsamp):
        """Investigate Stage-2 collapse on blanked data: detect blanks from the
        raw test signal, flag comb-like (edge-locked / stim-periodic) units, and
        write plots. Call with the PRE-dedup sources so all candidates are seen.
        """
        import os
        try:
            import matplotlib
            # matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as e:
            print(f"[DEBUG] matplotlib unavailable ({e}); skipping plots")
            plt = None

        raw_test = np.asarray(raw_test)
        T = raw_test.shape[1]
        valid = _dbg_detect_blanks(raw_test, self.blank_detect_tol)
        edges, onsets = _dbg_edges(valid)
        blank_frac = float(np.mean(~valid))
        guard = int(0.003 * fsamp)

        if self.stim_hz:
            stim_period = int(round(fsamp / self.stim_hz))
        elif onsets.size >= 2:
            stim_period = int(np.median(np.diff(onsets)))
        else:
            stim_period = 0

        print(f"\n{'='*80}\n[DEBUG] Stage-2 diagnostics")
        print(f"  blanks: {blank_frac*100:.1f}% of samples; "
              f"{onsets.size} blank onsets; "
              f"inferred stim period {stim_period} samp "
              f"({fsamp/stim_period:.1f} Hz)" if stim_period else
              f"  blanks: {blank_frac*100:.1f}% of samples; none detected")
        if blank_frac > 0 and (self.apply_bandpass or self.apply_notch):
            print("  [WARN] apply_bandpass/apply_notch is ON while the input is "
                  "blanked.\n         filtfilt rings across the zero edges and "
                  "creates a periodic comb.\n         If you already filtered "
                  "before blanking, set apply_bandpass=apply_notch=False.")

        n = sources.shape[0]
        comb = 0
        print(f"  per-unit (pre-dedup, {n} candidates):")
        ef_all, cs_all = [], []
        for u in range(n):
            ef = _dbg_edge_fraction(spikes[u], edges, guard)
            cs, _ = _dbg_isi_comb(spikes[u], fsamp, stim_period)
            ef_all.append(ef); cs_all.append(cs)
            is_comb = ef > 0.5 or cs > 0.5
            comb += int(is_comb)
            if self.verbose_mode:
                print(f"    MU{u:>2}: sil={sil[u]:.3f} edge-frac={ef:.2f} "
                      f"isi-comb={cs:.2f} spikes={len(spikes[u])}"
                      f"{'   <-- COMB-LIKE' if is_comb else ''}")
        print(f"  => {comb}/{n} candidates look comb-like "
              f"(edge-frac>0.5 or isi-comb>0.5)")

        if plt is None:
            print("=" * 80)
            return

        os.makedirs(self.debug_dir, exist_ok=True)
        t = np.arange(T) / fsamp
        win = slice(0, min(T, int(1.0 * fsamp)))
        m = min(self.debug_max_units, n)

        fig, ax = plt.subplots(figsize=(12, 3))
        ax.plot(t, np.abs(raw_test).sum(0), "k", lw=0.5)
        ax.fill_between(t, 0, np.abs(raw_test).sum(0).max(), where=~valid,
                        color="red", alpha=0.12, step="mid")
        ax.set_title(f"Blank detection ({blank_frac*100:.1f}% blanked)")
        ax.set_xlabel("s"); fig.tight_layout()
        fig.savefig(f"{self.debug_dir}/01_blank_detection.png", dpi=110)
        plt.close(fig)

        fig, axes = plt.subplots(m, 1, figsize=(12, 1.8 * m), squeeze=False)
        for u in range(m):
            a = axes[u, 0]
            a.plot(t[win], sources[u, win], "b", lw=0.5)
            sp = np.asarray(spikes[u], int)
            sp = sp[(sp >= win.start) & (sp < win.stop)]
            if sp.size:
                a.scatter(sp / fsamp, sources[u, sp], c="r", s=12, zorder=5)
            lo, hi = sources[u, win].min(), sources[u, win].max()
            a.fill_between(t[win], lo, hi, where=~valid[win], color="red",
                           alpha=0.12, step="mid")
            a.set_title(f"MU{u} sil={sil[u]:.3f} "
                        f"edge-frac={ef_all[u]:.2f} isi-comb={cs_all[u]:.2f}",
                        fontsize=9)
        axes[-1, 0].set_xlabel("s"); fig.tight_layout()
        fig.savefig(f"{self.debug_dir}/02_sources_overlay.png", dpi=110)
        plt.close(fig)

        fig, axes = plt.subplots(m, 1, figsize=(8, 1.5 * m), squeeze=False)
        stim_ms = stim_period / fsamp * 1000 if stim_period else None
        for u in range(m):
            a = axes[u, 0]
            _, isi_ms = _dbg_isi_comb(spikes[u], fsamp, stim_period)
            if isi_ms.size:
                a.hist(isi_ms, bins=60, color="gray", edgecolor="k", lw=0.3)
            if stim_ms:
                for k in range(1, 8):
                    a.axvline(k * stim_ms, color="r", ls="--", lw=0.8)
                a.set_xlim(0, stim_ms * 6)
            a.set_title(f"MU{u} ISI (ms); red = stim-period multiples", fontsize=9)
        axes[-1, 0].set_xlabel("ISI (ms)"); fig.tight_layout()
        fig.savefig(f"{self.debug_dir}/03_isi_hist.png", dpi=110)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(12, max(3, 0.25 * n)))
        for u in range(n):
            sp = np.asarray(spikes[u], int)
            if sp.size:
                ax.scatter(sp / fsamp, np.full(sp.size, u), s=4, c="k")
        ax.fill_between(t, -1, n, where=~valid, color="red", alpha=0.12, step="mid")
        ax.set_xlim(win.start / fsamp, win.stop / fsamp)
        ax.set_title("Raster (pre-dedup): aligned columns => duplicate collapse")
        ax.set_xlabel("s"); ax.set_ylabel("MU #"); fig.tight_layout()
        fig.savefig(f"{self.debug_dir}/04_raster.png", dpi=110)
        plt.close(fig)

        if n <= 60:
            M = _dbg_pairwise_agreement(spikes, T, fsamp)
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(M, vmin=0, vmax=1, cmap="hot")
            fig.colorbar(im, label="best-lag RoA")
            ax.set_title("Pairwise agreement (pre-dedup)")
            fig.tight_layout()
            fig.savefig(f"{self.debug_dir}/05_agreement.png", dpi=110)
            plt.close(fig)

        print(f"  plots written to {self.debug_dir}/\n{'='*80}")

    def decompose(self, train_sig, test_sig, fsamp,
                  mu_filters=None, muaps=None, stim_sig=None,
                  train_spikes=None):
        """Decompose ``test_sig`` using MU filters reconstructed from ``train_sig``.

        Args
        ----
        train_sig : (n_channels, n_samples) EMG used to build MU filters (x1).
        test_sig  : (n_channels, n_samples) EMG to decompose (x2).
        fsamp     : sampling frequency in Hz.
        mu_filters, muaps : optionally supply a pre-built filter bank to skip
            Stage 1 (e.g. an MU-filter library across force levels).
        train_spikes : optionally supply pre-computed discharge times (iterable
            of per-unit sample-index arrays, e.g. ``list(cbss_results["spikes"]
            .values())``) to skip only the gCKC search: the filters are still
            reconstructed by STA on ``train_sig`` at these spikes. Ignored when
            ``mu_filters`` is given. Indices must match ``train_sig``'s timeline.
        stim_sig  : optional (n_channels, n_samples) recording containing ONLY
            the stimulation artifact. When ``remove_stim_artifact`` is set, its
            template is estimated and subtracted from the raw test signal before
            pre-processing. Ignored otherwise.

        Returns
        -------
        sources    : (n_units, n_samples) LMMSE source estimates.
        spikes     : dict {unit_id: np.ndarray of discharge sample indices}.
        sil        : (n_units,) pseudo-silhouette scores.
        mu_filters : (n_channels*K, n_units) final MU filters.
        muaps      : list of (n_channels, N) MUAPs used for reconstruction.
        centroids  : dict {unit_id: K-means centroids}.
        train_results : dict mirroring the test outputs for the Stage-1
            (training) decomposition -- sources, spikes, silhouette, cov,
            mu_filters, muaps, centroids -- or ``None`` when a pre-built filter
            bank is supplied (Stage 1 skipped).
        """
        np.random.seed(self.random_seed)

        # ---- Optional: remove the stimulation artifact from the test ------
        # Done on the RAW test signal, before pre-processing, so the band-pass /
        # notch filtfilt does not ring across the artifact's sharp edges.
        
        self._stim_info = None
        test_clean = None

        if self.remove_stim_artifact and stim_sig is not None:
            if self.verbose_mode:
                print(f"\n{'='*80}\nSTIM-ARTIFACT REMOVAL: estimating template "
                      f"from stim-only recording\n{'='*80}")
            template, ref_peaks, period = estimate_stim_template(
                stim_sig, fsamp,
                win=self.stim_template_win,
                stim_hz=self.stim_hz,
                refractory_frac=self.stim_refractory_frac,
                height_frac=self.stim_height_frac,
            )
            test_clean, stim_comp, stim_times = remove_stim_artifact(
                test_sig, template, fsamp,
                stim_hz=self.stim_hz,
                scale_per_pulse=self.stim_scale_per_pulse,
                align_jitter=self.stim_align_jitter,
                subsample_align=self.stim_subsample_align,
                refractory_frac=self.stim_refractory_frac,
                height_frac=self.stim_height_frac,
                blank=self.stim_blank,
                blank_fill=self.stim_blank_fill,
                blank_pad=self.stim_blank_pad,
            )
            if self.verbose_mode:
                eff_hz = self.stim_hz or (fsamp / period if period else 0.0)
                print(f"  template: {template.shape[0]} ch x "
                      f"{template.shape[1]} samp "
                      f"({self.stim_template_win*1000:.1f} ms); "
                      f"{len(ref_peaks)} ref pulses (~{eff_hz:.1f} Hz)")
                if self.stim_blank:
                    print(f"  blanked {len(stim_times)} pulses from test "
                          f"(fill={self.stim_blank_fill}, "
                          f"pad=+/-{self.stim_blank_pad} samp)")
                else:
                    print(f"  subtracted {len(stim_times)} pulses from test "
                          f"(scale_per_pulse={self.stim_scale_per_pulse}, "
                          f"jitter=+/-{self.stim_align_jitter}, "
                          f"subsample={self.stim_subsample_align})")
            self._stim_info = {
                "template": template, "ref_peaks": ref_peaks,
                "stim_times": stim_times, "period": period,
            }
            if self.debug:
                self._plot_stim_removal(test_sig, test_clean, template,
                                        stim_times, fsamp)
            test_sig = test_clean
        elif self.remove_stim_artifact and self.verbose_mode:
            print("[WARN] remove_stim_artifact=True but no stim_sig supplied; "
                  "skipping stim-artifact removal")

        # ---- Stage 1 ------------------------------------------------------
        train_results = None
        if mu_filters is None:
            mu_filters, muaps, _, train_results = self.reconstruct_filters(
                train_sig, fsamp, train_spikes=train_spikes)
        if muaps is None:
            muaps = []

        n_units = mu_filters.shape[1]
        if n_units == 0:
            if self.verbose_mode:
                print("[WARN] no MU filters available; nothing to decompose")
            empty_sources = np.zeros((0, np.asarray(test_sig).shape[1]))
            return (empty_sources, {}, np.zeros(0), mu_filters, muaps, {},
                    train_results, test_clean)

        # ---- Stage 2 ------------------------------------------------------
        if self.verbose_mode:
            print(f"\n{'='*80}\nSTAGE 2: decomposing test signal with "
                  f"{n_units} MU filter(s) (mode={self.decomp_mode}, "
                  f"Cxx={self.cxx_mode})\n{'='*80}")

        test_pp = self._preprocess(test_sig, fsamp)
        test_ext = self._extend(test_pp)
        n_t = test_ext.shape[1]

        # covariance source depends on cxx_mode; the source is always on x2
        train_ext = None
        if self.cxx_mode.lower() in ("raw", "global"):
            train_ext = self._extend(self._preprocess(train_sig, fsamp))
        Cinv = self._covariance_inverse(train_ext, test_ext)
        Cx = Cinv @ test_ext

        sources = np.zeros((n_units, n_t))
        spikes = {i: [] for i in range(n_units)}
        sil = np.zeros(n_units)
        covs = np.zeros(n_units)
        centroids = {i: None for i in range(n_units)}
        out_filters = np.zeros_like(mu_filters)

        for j in range(n_units):
            c, source, spk, s, cent = self._estimate_one(
                mu_filters[:, j], test_ext, Cx, fsamp)
            sources[j, :] = source
            spikes[j] = np.asarray(spk, dtype=int)
            sil[j] = s
            centroids[j] = cent
            covs[j] = _cov_isi(spk, fsamp)
            out_filters[:, j] = c
            if self.verbose_mode:
                fr = len(spk) / (n_t / fsamp) if len(spk) else 0.0
                print(f"  MU {j}: sil={s:.3f} cov={covs[j]:.3f} "
                      f"fr={fr:.2f} Hz spikes={len(spk)}")

        # ---- Diagnostics (pre-dedup, all candidates) ----------------------
        if self.debug:
            self._diagnostics(test_sig, sources, spikes, sil, fsamp)

        # ---- Post-processing: duplicates + bad sources --------------------
        # core.remove_duplicates does not reindex centroids/covs, and neither
        # removal step touches muaps. To keep muaps, centroids and covs aligned
        # 1:1 with the surviving units, we tag each unit's original index onto an
        # extra row of the filter matrix; both core functions carry that row
        # along verbatim (column selection / np.delete preserve it), so we can
        # recover the survivor indices and reindex the auxiliary outputs.
        muaps_aligned = (list(muaps) if (muaps is not None
                         and len(muaps) == n_units) else None)

        tag = np.arange(n_units, dtype=float).reshape(1, -1)
        tagged = np.vstack([out_filters, tag])

        
        sources, spikes, sil, tagged = remove_duplicates(
            sources, spikes, sil, tagged, fsamp,
            max_shift=self.match_max_shift,
            tol=self.match_tol,
            threshold=self.match_th,
        )
        dup_keep = tagged[-1, :].astype(int)      # original idx of each survivor
        out_filters = tagged[:-1, :]
        covs = covs[dup_keep]
        centroids = {i: centroids[int(dup_keep[i])] for i in range(len(dup_keep))}
        if muaps_aligned is not None:
            muaps_aligned = [muaps_aligned[int(i)] for i in dup_keep]

        tag = np.arange(out_filters.shape[1], dtype=float).reshape(1, -1)
        tagged = np.vstack([out_filters, tag])
        sources, spikes, sil, tagged, centroids = remove_bad_sources(
            sources, spikes, sil, tagged, centroids, covs,
            threshold=self.sil_th,
            max_cov=self.cov_th,
            min_firing_rate=self.min_firing_rate,
            fsamp=fsamp,
        )
        bad_keep = tagged[-1, :].astype(int)      # post-dup idx of each survivor
        out_filters = tagged[:-1, :]
        covs = covs[bad_keep]
        if muaps_aligned is not None:
            muaps = [muaps_aligned[int(i)] for i in bad_keep]

        if self.verbose_mode:
            print(f"\n{'='*80}\nDECOMPOSITION COMPLETE: {sources.shape[0]} "
                  f"motor unit(s)\n{'='*80}\n")

        return sources, spikes, sil, out_filters, muaps, centroids, train_results, test_clean