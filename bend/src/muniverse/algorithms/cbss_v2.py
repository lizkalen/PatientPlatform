"""CBSS with rank-truncated whitening, channel preprocessing, and unit validity.

Differences from :mod:`cbss`:

1. Rank-truncated whitening. The extended covariance is eigen-decomposed and
   components with eigenvalue at or below ``whitening_rank_trunc`` times the
   largest are dropped rather than inverted. The whitened signal has
   ``n_kept_dims`` rows instead of ``n_channels * ext_fact``. Set
   ``whitening_rank_trunc=None`` to use ``core.whitening`` with a ridge, which
   is what :mod:`cbss` does.
2. Channel selection, common-mode removal, and per-channel amplitude
   normalisation before extension, from ``utils.preprocessing``.
3. Per-unit validity scores from ``utils.validity``, computed on the
   preprocessed channel-space signal after bad-source and duplicate removal.
4. Post-processing order and duplicate policy. Bad sources are removed first, so
   a duplicate group only holds units that already clear the silhouette and
   firing-rate gate, and the member kept for each group is the one with the
   lowest CoV-ISI rather than the highest silhouette. :mod:`cbss` deduplicates
   first and keeps the highest silhouette.

The ICA loop, spike detection, CoV-ISI refinement, peel-off, and the duplicate
criterion itself are unchanged from :mod:`cbss`.

``decompose`` returns the six outputs of ``cbss.CBSS.decompose`` in the same
order, followed by the five outputs of ``corrected_ckc.decompose``.
"""
import os

import matplotlib.pyplot as plt
import numpy as np

from .core import (bandpass_signals, notch_signals, extension, whitening,
                   est_spike_times, cov_isi, duplicate_representatives,
                   good_source_mask, gram_schmidt, peel_off)
from .utils.preprocessing import preprocess as preprocess_channels
from .utils.validity import summarize_validity, validate_units


class CBSSv2:
    """Convolutive blind source separation with preprocessing and validity gating.

    Parameters are set from a config object, keyword arguments, or
    :meth:`set_param`. Keyword arguments take precedence over the config.
    Unknown names are reported and ignored.

    Preprocessing:
        channel_criteria (tuple): Channel rules applied before extension. Any of
            "amplitude", "muap_band", "auto_muap_band". Empty tuple keeps all
            channels. See ``utils.preprocessing.select_channels``.
        criterion_bandpass (tuple or None): (low, high) in Hz used to filter the
            copy the MUAP-band rules are computed on. None when the input signal
            is already band-passed. The data itself is not filtered here.
        drop_channels (sequence or None): Explicit channel indices to drop.
            Overrides ``channel_criteria``.
        amplitude_kwargs (dict or None): Passed to
            ``utils.preprocessing.amplitude_bad_channels``. Set
            ``{"kurt_th": None}`` on stimulation recordings.
        muap_band_kwargs (dict or None): Passed to the MUAP-band rule.
        common_mode_k (int): Spatial components removed across channels. 0 skips.
        zscore_channels (bool): Divide each kept channel by its standard
            deviation.

    Filtering of the data itself, off by default as in cbss:
        apply_filters (bool): Enables the two filters below.
        bandpass (list or None), bandpass_order (int), notch_frequency (float or
        None), notch_n_harmonics (int), notch_order (int), notch_width (float).

    Extension and whitening:
        ext_fact (int): Extension factor.
        whitening_method (str): "ZCA", "PCA", or "Cholesky". Used only when
            ``whitening_rank_trunc`` is None; truncated whitening is PCA form.
        whitening_reg (str or float): Ridge for the untruncated path.
        whitening_rank_trunc (float or None): Eigenvalue cut as a fraction of
            the largest eigenvalue. None disables truncation.

    ICA:
        ica_n_iter (int): Sources extracted.
        opt_initalization (str): "activity_idx" or "random".
        opt_function_exp (float): Exponent of the contrast function.
        opt_max_iter (int), opt_tol (float): Fixed-point stopping conditions.
        source_deflation (str): "gram-schmidt", "projection_deflation", or None.
        peel_off (bool), fr_peeloff (bool): Subtract accepted sources from the
            whitened signal, optionally gated on firing rate.
        cluster_method (str): "kmeans" or "kmedoids".
        random_seed (int): Seeds the random initialisation.
        refinement_loop (bool): Run CoV-ISI minimisation on each source.

    Source acceptance:
        sil_th (float), cov_th (float), min_firing_rate (float): Silhouette,
            CoV-ISI, and firing-rate cuts.
        match_th (float), match_max_shift (float), match_tol (float): Duplicate
            removal. Two units are duplicates when the fraction of spikes that
            match within ``match_tol``, at the best lag within
            ``match_max_shift``, reaches ``match_th``; grouping is transitive.
            Duplicates are removed after the cuts above and the member with the
            lowest CoV-ISI is the one kept.
        validity_gate (str or None): "annotate" scores every surviving unit and
            keeps all of them; "filter" also discards units that fail; None
            skips scoring and returns an empty unit list.
        validity_thresholds (dict or None): Overrides for
            ``utils.validity.DEFAULT_GATE``.
        validity_half_ms (float), validity_top_m (int): STA window half length
            and channel count for the split-half correlation.

    Post-processing:
        recalibrate_filters (bool): Re-estimate each accepted filter by
            spike-triggered averaging on the un-peeled whitened signal.
            Restores temporal overlaps that peel-off removed.

    Reporting:
        verbose_mode (bool), output_source_plot (bool), plot_all_sources (bool).
    """

    def __init__(self, config=None, **kwargs):

        # Channel preprocessing
        self.channel_criteria = ("amplitude", "auto_muap_band")
        self.criterion_bandpass = (20.0, 500.0)
        self.drop_channels = None
        self.amplitude_kwargs = None
        self.muap_band_kwargs = None
        self.common_mode_k = 2
        self.zscore_channels = True

        # Optional filtering of the data
        self.apply_filters = False
        self.bandpass = [20, 500]
        self.bandpass_order = 2
        self.notch_frequency = 50
        self.notch_n_harmonics = 3
        self.notch_order = 2
        self.notch_width = 1

        # Extension and whitening
        self.ext_fact = 16
        self.whitening_method = "ZCA"
        self.whitening_reg = "auto"
        self.whitening_rank_trunc = 1e-3

        # ICA
        self.ica_n_iter = 100
        self.opt_initalization = "activity_idx"
        self.opt_function_exp = 2
        self.opt_max_iter = 100
        self.opt_tol = 1e-4
        self.source_deflation = "gram-schmidt"
        self.peel_off = True
        self.fr_peeloff = False
        self.cluster_method = "kmeans"
        self.random_seed = 1909
        self.refinement_loop = True

        # Source acceptance
        self.sil_th = 0.9
        self.cov_th = 0.35
        self.min_firing_rate = 10  # in Hz
        self.match_th = 0.3
        self.match_max_shift = 0.1
        self.match_tol = 0.001
        self.validity_gate = "annotate"
        self.validity_thresholds = None
        self.validity_half_ms = 10.0
        self.validity_top_m = 16

        # Post-processing
        self.recalibrate_filters = False

        # Reporting
        self.verbose_mode = True
        self.output_source_plot = False
        self.plot_all_sources = False

        # Filled by decompose(). See utils.validity.summarize_validity for the keys.
        self.validity_report = None
        # Filled by decompose(). See utils.preprocessing.select_channels for the keys:
        # which channel rule flagged which electrode, and the values behind each test.
        self.channel_report = None

        # Convert config object (if provided) to a dictionary
        config_dict = vars(config) if config is not None else {}

        # Merge with directly passed keyword arguments (overwrites config)
        params = {**config_dict, **kwargs}

        valid_keys = self.__dict__.keys()

        for key, value in params.items():
            if key in valid_keys:
                setattr(self, key, value)
            else:
                print(f"Warning: ignoring invalid parameter: {key}")

    def set_param(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"Invalid parameter: {key}")

    # ------------------------------------------------------------------ #
    # Whitening                                                          #
    # ------------------------------------------------------------------ #
    def whiten(self, ext_sig):
        """Whiten the extended signal, truncating the rank when configured.

        With ``whitening_rank_trunc`` set, the covariance is eigen-decomposed and
        only components with eigenvalue above ``whitening_rank_trunc`` times the
        largest are kept. The result is PCA-form whitening with the
        dimensionality reduced to the number of kept components, regardless of
        ``whitening_method``. With ``whitening_rank_trunc`` None,
        ``core.whitening`` is called and the dimensionality is preserved.

        Args:
            ext_sig (ndarray): Extended signal,
                (n_channels * ext_fact, n_samples).

        Returns:
            tuple:
                - ndarray: Whitened signal, (n_kept_dims, n_samples).
                - ndarray: Whitening matrix Z,
                  (n_kept_dims, n_channels * ext_fact). Maps an extended signal
                  into the whitened space.
                - int: n_kept_dims.
        """
        if self.whitening_rank_trunc is None:
            white_sig, Z = whitening(Y=ext_sig, method=self.whitening_method,
                                     regularization=self.whitening_reg)
            return white_sig, Z, int(white_sig.shape[0])

        cov = ext_sig @ ext_sig.T / (ext_sig.shape[1] - 1)
        lam, V = np.linalg.eigh(cov.astype(np.float64, copy=False))
        lam, V = lam[::-1], V[:, ::-1]
        if lam[0] <= 0:
            raise ValueError("Extended covariance has no positive eigenvalue.")

        n_kept = max(1, int(np.sum(lam > self.whitening_rank_trunc * lam[0])))
        Z = (V[:, :n_kept] / np.sqrt(lam[:n_kept])).T
        return Z @ ext_sig, Z, n_kept

    # ------------------------------------------------------------------ #
    # Decomposition                                                      #
    # ------------------------------------------------------------------ #
    def decompose(self, sig, fsamp, from_scratch=True, just_preprocess=False, save_path=None):
        """Decompose one recording into motor unit spike trains.

        Args:
            sig (ndarray): Input EMG, (n_channels, n_samples). The channel
                indices in the returned ``keep`` and ``dropped`` refer to these
                rows.
            fsamp (float): Sampling frequency in Hz.
            from_scratch (bool): Unused, kept for signature compatibility.
            just_preprocess (bool): Return after whitening.
            save_path (str or None): Directory for source plots, used when
                ``output_source_plot`` is set.

        Returns:
            tuple: With ``just_preprocess`` False, eleven items. The first six
            match ``cbss.CBSS.decompose``; the last five match
            ``corrected_ckc.decompose``.

                - sources (ndarray): Accepted sources, (n_units, n_samples).
                - spikes (dict): Unit index to spike sample indices.
                - sil (ndarray): Silhouette score per unit, (n_units,).
                - mu_filters (ndarray): Separation filters,
                  (n_kept_dims, n_units). Applies to the whitened signal.
                - Z (ndarray): Whitening matrix,
                  (n_kept_dims, n_kept_channels * ext_fact). Applies to the
                  extended preprocessed signal, not to raw input.
                - centroids (dict): Unit index to the two spike-detection
                  cluster centroids.
                - units (list[dict]): Validity scores per unit, ordered to match
                  the ``sources`` rows. Empty when ``validity_gate`` is None.
                  See ``utils.validity.validate_units`` for the keys.
                - keep (ndarray): Channel indices kept by preprocessing.
                - dropped (ndarray): Channel indices dropped by preprocessing.
                - n_kept_dims (int): Whitened dimensions retained.
                - X (ndarray): Preprocessed signal the units were scored
                  against, (n_kept_channels, n_samples).

            With ``just_preprocess`` True, six items: white_sig, Z, keep,
            dropped, n_kept_dims, X.

        Side effects:
            Sets ``self.validity_report``: how many units were scored before the
            hardened gate, how many passed, and which criterion refused which
            unit. Built before filtering, so it still describes the refused units
            when ``validity_gate`` is "filter". See
            ``utils.validity.summarize_validity``.
        """
        rng = np.random.default_rng(self.random_seed)

        # Create save directory if plotting is enabled
        if self.output_source_plot and save_path is not None:
            os.makedirs(save_path, exist_ok=True)

        sig = np.asarray(sig, float)
        if sig.ndim != 2:
            raise ValueError("sig must be 2D (n_channels, n_samples).")

        # Optional filtering of the data itself
        if self.apply_filters:
            if self.bandpass is not None:
                sig = bandpass_signals(sig, fsamp, high_pass=self.bandpass[0],
                                       low_pass=self.bandpass[1], order=self.bandpass_order)
            if self.notch_frequency is not None:
                sig = notch_signals(sig, fsamp, nfreq=self.notch_frequency,
                                    dfreq=self.notch_width, order=self.notch_order,
                                    n_harmonics=self.notch_n_harmonics)

        # Channel selection, common-mode removal, amplitude normalisation
        X, keep, dropped, ch_info = preprocess_channels(
            sig, fsamp,
            criteria=tuple(self.channel_criteria),
            common_mode_k=self.common_mode_k,
            zscore=self.zscore_channels,
            criterion_bandpass=self.criterion_bandpass,
            drop=self.drop_channels,
            amplitude_kwargs=self.amplitude_kwargs,
            muap_band_kwargs=self.muap_band_kwargs,
        )
        if keep.size == 0:
            raise ValueError("Channel selection dropped every channel.")
        self.channel_report = ch_info
        if self.verbose_mode:
            print(f"[INFO] Channels kept {keep.size}/{sig.shape[0]}, dropped {dropped.size} "
                  f"(criteria: {ch_info['criteria']}), "
                  f"common-mode components removed: {self.common_mode_k}")
            for rule in ("rms_high", "rms_low", "kurtosis", "muap_band"):
                n_bad = len(ch_info.get(f"{rule}_dropped", ()))
                if n_bad:
                    extra = ""
                    if rule == "muap_band":
                        extra = (f" (threshold {ch_info['muap_band_threshold']:.3f}, "
                                 f"bimodal {ch_info['muap_band_bimodal']})")
                    print(f"[INFO]   flagged by {rule}: {n_bad}{extra}")

        # Extend signals, subtract the mean, and cut the edges
        if self.verbose_mode:
            print(f"[INFO] Extending signals by factor {self.ext_fact}...")
        ext_sig = extension(X, self.ext_fact)
        ext_sig -= np.mean(ext_sig, axis=1, keepdims=True)
        ext_sig[:, : self.ext_fact * 2] = 0
        ext_sig[:, -self.ext_fact * 2:] = 0

        # Whiten the extended signals
        white_sig, Z, n_kept_dims = self.whiten(ext_sig)
        if self.verbose_mode:
            print(f"[INFO] Whitening kept {n_kept_dims}/{ext_sig.shape[0]} dimensions "
                  f"(rank_trunc: {self.whitening_rank_trunc})")

        if just_preprocess:
            return white_sig, Z, keep, dropped, n_kept_dims, X

        # Initalize the output variables
        n_samples = X.shape[1]
        sources = np.zeros((self.ica_n_iter, n_samples))
        spikes = {i: [] for i in range(self.ica_n_iter)}
        sil = np.zeros(self.ica_n_iter)
        centroids = {i: None for i in range(self.ica_n_iter)}
        mu_filters = np.zeros((white_sig.shape[0], self.ica_n_iter))

        if self.opt_initalization == "activity_idx":
            act_idx_histoty = np.array([])

        if self.verbose_mode:
            print(f"\n{'='*80}")
            print(f"STARTING CBSS V2 DECOMPOSITION")
            print(f"Max iterations: {self.ica_n_iter}, Silhouette threshold: {self.sil_th:.3f}, "
                  f"CoV threshold: {self.cov_th:.3f}")
            print(f"{'='*80}\n")

        # Loop over each MU
        for i in range(self.ica_n_iter):
            # Initalize
            if self.opt_initalization == "random":
                w = rng.standard_normal(white_sig.shape[0])
            elif self.opt_initalization == "activity_idx":
                col_norms = np.linalg.norm(white_sig, axis=0)
                col_norms[act_idx_histoty.astype(int)] = 0
                best_idx = np.argmax(col_norms)
                w = white_sig[:, best_idx]
                act_idx_histoty = np.append(act_idx_histoty, best_idx)
            else:
                raise ValueError(
                    f"Unknown opt_initalization: {self.opt_initalization}. "
                    "Use 'random' or 'activity_idx'."
                )

            # fastICA fixedpoint optimization
            w, k, early_stopped, early_stop_info = self.my_fixed_point_alg(
                w, white_sig, mu_filters, fsamp=fsamp)

            # Predict source and estimate the source quality
            sources[i, :] = w.T @ white_sig
            spikes[i], sil[i], centroids[i] = est_spike_times(
                sources[i, :], fsamp, cluster=self.cluster_method
            )
            cov = cov_isi(spikes[i], fsamp)

            # Refinement loop
            if len(spikes[i]) > 10 and self.refinement_loop:
                if self.verbose_mode:
                    print(f"Refinement loop for source {i} (initial Sil: {sil[i]:.3f}, "
                          f"CoV: {cov:.3f}, Spikes: {len(spikes[i])})")
                w, _, cov, _ = self.mimimize_covisi(w, white_sig, cov, fsamp)
                sources[i, :] = w.T @ white_sig
                spikes[i], sil[i], centroids[i] = est_spike_times(
                    sources[i, :], fsamp, cluster=self.cluster_method
                )

            # Save the optimized MU filter
            mu_filters[:, i] = w

            # Calculate firing rate for logging
            if len(spikes[i]) > 0:
                fr = len(spikes[i]) / (n_samples / fsamp)
            else:
                fr = 0.0

            # Peel-off the detected source
            if self.peel_off and self.fr_peeloff:
                if (sil[i] > self.sil_th and cov < self.cov_th
                        and fr >= self.min_firing_rate and not early_stopped):
                    white_sig, _, _ = peel_off(white_sig, spikes[i], win=0.04, fsamp=fsamp)
                    if self.verbose_mode:
                        print(f"{i}: PEEL-OFF - Silhouette: {sil[i]:.3f} "
                              f"(threshold: {self.sil_th:.3f}), CoV: {cov:.3f} "
                              f"(threshold: {self.cov_th:.3f}), FR: {fr:.2f} Hz, "
                              f"Spikes: {len(spikes[i])}")
                    if self.output_source_plot:
                        self._plot_source(sources[i, :], spikes[i], sil[i], cov, fr,
                                          'accepted', i, save_path)
            elif self.peel_off and sil[i] > self.sil_th and cov < self.cov_th:
                white_sig, _, _ = peel_off(white_sig, spikes[i], win=0.04, fsamp=fsamp)
                if self.verbose_mode:
                    print(f"{i}: PEEL-OFF - Silhouette: {sil[i]:.3f} "
                          f"(threshold: {self.sil_th:.3f}), CoV: {cov:.3f} "
                          f"(threshold: {self.cov_th:.3f}), FR: {fr:.2f} Hz, "
                          f"Spikes: {len(spikes[i])}")
                if self.output_source_plot:
                    self._plot_source(sources[i, :], spikes[i], sil[i], cov, fr,
                                      'accepted', i, save_path)
            else:
                # Determine rejection reason and source type
                rejection_reasons = []
                source_type = 'rejected_other'

                if early_stopped and self.fr_peeloff:
                    rejection_reasons.append(
                        f"early stopped at iter {early_stop_info['iteration']} "
                        f"({early_stop_info['spike_count']} < "
                        f"{early_stop_info['threshold']} spikes)")
                    source_type = 'rejected_earlystop'
                if len(spikes[i]) == 0:
                    rejection_reasons.append("no spikes detected")
                    if source_type == 'rejected_other':
                        source_type = 'rejected_nospikes'
                if self.fr_peeloff and fr < self.min_firing_rate:
                    rejection_reasons.append(
                        f"low firing rate ({fr:.2f} < {self.min_firing_rate:.2f} Hz)")
                    if source_type == 'rejected_other':
                        source_type = 'rejected_fr'
                if sil[i] <= self.sil_th:
                    rejection_reasons.append(
                        f"low silhouette ({sil[i]:.3f} <= {self.sil_th:.3f})")
                    if source_type == 'rejected_other':
                        source_type = 'rejected_sil'
                if cov >= self.cov_th:
                    rejection_reasons.append(f"high CoV ({cov:.3f} >= {self.cov_th:.3f})")
                    if source_type == 'rejected_other':
                        source_type = 'rejected_cov'

                reason_str = ", ".join(rejection_reasons) if rejection_reasons else "criteria not met"
                if self.verbose_mode:
                    print(f"{i}: NO PEEL-OFF ({reason_str}) - Silhouette: {sil[i]:.3f}, "
                          f"CoV: {cov:.3f}, FR: {fr:.2f} Hz, Spikes: {len(spikes[i])}")

                # Plot rejected source if plot_all_sources is enabled
                if self.output_source_plot and self.plot_all_sources:
                    self._plot_source(sources[i, :], spikes[i], sil[i], cov, fr,
                                      source_type, i, save_path)

        if self.verbose_mode:
            print(f"\n{'='*80}")
            print(f"POST-PROCESSING: Removing bad sources and duplicates")
            print(f"{'='*80}\n")

        # Remove bad sources first, so that a duplicate group only ever holds
        # units that already clear the silhouette and firing rate gate and the
        # representative is picked among units acceptable on their own
        n_sources_before_removal = sources.shape[0]
        good_idx = np.flatnonzero(good_source_mask(
            spikes, sil, n_samples, fsamp=fsamp,
            threshold=self.sil_th, min_firing_rate=self.min_firing_rate,
        ))
        sources, spikes, sil, mu_filters, centroids = self._select_units(
            sources, spikes, sil, mu_filters, centroids, good_idx)
        n_bad_removed = n_sources_before_removal - sources.shape[0]
        if self.verbose_mode and n_bad_removed > 0:
            print(f"BAD SOURCE REMOVAL: Removed {n_bad_removed} bad source(s) "
                  f"(sil_th: {self.sil_th:.3f}, min_firing_rate: {self.min_firing_rate:.2f} Hz)")

        # Remove duplicates, keeping the most regular train of each group. CoV is
        # recomputed from the surviving spikes so that it describes the train
        # that is actually kept rather than the one the refinement loop started from
        n_sources_before_dup = sources.shape[0]
        covs = np.array([cov_isi(spikes[i], fsamp) for i in range(sources.shape[0])])
        rep_idx = duplicate_representatives(
            spikes, sil, n_samples, fsamp,
            max_shift=self.match_max_shift, tol=self.match_tol, threshold=self.match_th,
            covs=covs, sil_th=self.sil_th, min_firing_rate=self.min_firing_rate,
        )
        sources, spikes, sil, mu_filters, centroids = self._select_units(
            sources, spikes, sil, mu_filters, centroids, rep_idx)
        n_duplicates_removed = n_sources_before_dup - sources.shape[0]
        if self.verbose_mode and n_duplicates_removed > 0:
            print(f"DUPLICATE REMOVAL: Removed {n_duplicates_removed} duplicate source(s), "
                  f"keeping the lowest CoV-ISI of each group "
                  f"(match_th: {self.match_th:.3f}, max_shift: {self.match_max_shift:.3f})")

        # Re-estimate filters on the un-peeled whitened signal so that
        # temporal overlaps removed by peel-off are restored.
        if self.recalibrate_filters and sources.shape[0] > 0:
            white_sig_clean = Z @ ext_sig
            n_recal = 0
            for i in range(sources.shape[0]):
                sp = np.asarray(spikes[i], int)
                if sp.size < 5:
                    if self.verbose_mode:
                        print(f"  MU {i}: {sp.size} spikes, keeping original filter")
                    continue
                sta = np.mean(white_sig_clean[:, sp], axis=1)
                norm = np.linalg.norm(sta)
                if norm > 1e-10:
                    mu_filters[:, i] = sta / norm
                    n_recal += 1
            for i in range(sources.shape[0]):
                sources[i, :] = mu_filters[:, i].T @ white_sig_clean
                spikes[i], sil[i], centroids[i] = est_spike_times(
                    sources[i, :], fsamp, cluster=self.cluster_method)
            del white_sig_clean
            if self.verbose_mode:
                print(f"RECALIBRATION: {n_recal}/{sources.shape[0]} filter(s) "
                      f"recalibrated by STA on the un-peeled whitened signal")

        # Validity scoring against the preprocessed channel-space signal
        units = []
        self.validity_report = summarize_validity([], self.validity_thresholds, X.shape[0])
        if self.validity_gate is not None and sources.shape[0] > 0:
            if self.validity_gate not in ("annotate", "filter"):
                raise ValueError(
                    f"Unknown validity_gate: {self.validity_gate}. "
                    "Use 'annotate', 'filter', or None."
                )
            units = validate_units(
                X, spikes, fsamp,
                gate=self.validity_thresholds,
                half_ms=self.validity_half_ms,
                top_m=self.validity_top_m,
                channel_index=keep,
            )
            # Built before filtering, so it describes every unit that was scored
            self.validity_report = summarize_validity(
                units, self.validity_thresholds, X.shape[0])

            if self.verbose_mode:
                rep = self.validity_report
                print(f"VALIDITY: {rep['n_after']}/{rep['n_before']} unit(s) pass the "
                      f"hardened gate (mode: {self.validity_gate})")
                if rep["satisfiable"] is False:
                    print(f"  NOTE: the gate cannot be satisfied on {X.shape[0]} channels; "
                          f"min_n_focal_ch and max_focality_frac conflict")
                for crit, ids in rep["refused_by"].items():
                    if ids:
                        print(f"  refused by {crit}: {len(ids)} unit(s) {ids}")

            if self.validity_gate == "filter":
                passing = [u for u in units if u["hardened"]]
                sources, spikes, sil, mu_filters, centroids = self._select_units(
                    sources, spikes, sil, mu_filters, centroids,
                    [u["unit"] for u in passing])
                units = [dict(u, source_index=u["unit"], unit=new)
                         for new, u in enumerate(passing)]

        # Show final accepted sources
        if self.verbose_mode:
            print(f"\n{'='*80}")
            print(f"FINAL ACCEPTED SOURCES: {sources.shape[0]} motor units")
            print(f"{'='*80}\n")

            for i in range(sources.shape[0]):
                cov = cov_isi(spikes[i], fsamp)
                fr = len(spikes[i]) / (sources.shape[1] / fsamp)

                tag = f", hardened: {units[i]['hardened']}" if units else ""
                print(f"MU {i}: ACCEPTED - Silhouette: {sil[i]:.3f}, CoV: {cov:.3f}, "
                      f"FR: {fr:.2f} Hz, Spikes: {len(spikes[i])}{tag}")

            print(f"\n{'='*80}")
            print(f"DECOMPOSITION COMPLETE")
            print(f"{'='*80}\n")

        return (sources, spikes, sil, mu_filters, Z, centroids,
                units, keep, dropped, n_kept_dims, X)

    @staticmethod
    def _select_units(sources, spikes, sil, mu_filters, centroids, idx):
        """Keep the units at ``idx`` and renumber the dict keys to 0..n-1.

        Args:
            sources (ndarray): (n_units, n_samples).
            spikes (dict): Unit index to spike indices.
            sil (ndarray): (n_units,).
            mu_filters (ndarray): (n_dims, n_units).
            centroids (dict): Unit index to cluster centroids.
            idx (sequence): Unit indices to keep, in the order to keep them.

        Returns:
            tuple: sources, spikes, sil, mu_filters, centroids restricted to
            ``idx`` and renumbered.
        """
        idx = list(idx)
        return (sources[idx, :],
                {new: spikes[old] for new, old in enumerate(idx)},
                sil[idx],
                mu_filters[:, idx],
                {new: centroids[old] for new, old in enumerate(idx)})

    def my_fixed_point_alg(self, w, X, B, fsamp=None):
        """Fixed-point iteration maximising the sparseness of the source.

        Args:
            w (ndarray): Initial filter, (n_dims,).
            X (ndarray): Whitened signal, (n_dims, n_samples).
            B (ndarray): Filters found so far, (n_dims, n_iter). Zero columns are
                ignored during deflation.
            fsamp (float or None): Sampling frequency in Hz. When given, the
                spike count is checked every 20 iterations and the loop stops
                early if it falls below ``min_firing_rate`` times the duration.

        Returns:
            tuple:
                - ndarray: Filter after convergence or the iteration cap.
                - int: Iterations run.
                - bool: Whether the loop stopped early on spike count.
                - dict: When it stopped early, keys ``iteration``,
                  ``spike_count``, ``threshold``. Empty otherwise.
        """

        # Define contrast function and its derivative
        # Use g(x)=x*(x**2+epsilon)**((a-1)/2) as smooth approximation of g(x) = sign(x) * abs(x)**a
        epsilon = 1e-3
        a = self.opt_function_exp
        g = lambda x: (epsilon + x**2) ** ((a - 3) / 2) * (a * x**2 + epsilon)
        gp = (
            lambda x: (a - 1)
            * x
            * (epsilon + x**2) ** ((a - 5) / 2)
            * (a * x**2 + 3 * epsilon)
        )

        TOL = self.opt_tol
        delta = np.ones(self.opt_max_iter)
        k = 0
        early_stopped = False
        early_stop_info = {}

        # Calculate minimum spike count threshold for early stopping
        min_spike_count = None
        if fsamp is not None:
            signal_duration = X.shape[1] / fsamp
            min_spike_count = int(np.ceil(self.min_firing_rate * signal_duration))

        while delta[k] > TOL and k < self.opt_max_iter - 1:
            w_last = w.copy()

            wTX = w.T @ X  # shape: (n_samples,)
            A = np.mean(gp(wTX))
            w = np.mean(X * g(wTX), axis=1) - A * w  # shape: (n_dims,)

            # Orthogonalization step
            if self.source_deflation == "projection_deflation":
                w = w - (B @ B.T) @ w
            elif self.source_deflation == "gram-schmidt":
                w = gram_schmidt(w, B)

            # Normalize. Deflation can drive w to zero once the kept subspace is
            # spanned; without this the division returns NaN and the filter, its
            # source and its spikes are all NaN from here on.
            norm = np.linalg.norm(w)
            if norm < 1e-12:
                w = w_last
                break
            w = w / norm

            # Convergence criterion
            delta[k + 1] = abs(np.dot(w, w_last) - 1)
            k += 1

            # Early stopping: check spike count every 20 iterations
            if min_spike_count is not None and k % 20 == 0:
                current_source = w.T @ X
                current_spikes, _, _ = est_spike_times(
                    current_source, fsamp, cluster=self.cluster_method)
                if len(current_spikes) < min_spike_count:
                    early_stopped = True
                    early_stop_info = {
                        'iteration': k,
                        'spike_count': len(current_spikes),
                        'threshold': min_spike_count
                    }
                    break

        return w, k, early_stopped, early_stop_info

    def mimimize_covisi(self, w, X, cov, fsamp):
        """Update the filter towards the spike-triggered mean while CoV-ISI falls.

        Args:
            w (ndarray): Initial filter, (n_dims,).
            X (ndarray): Whitened signal, (n_dims, n_samples).
            cov (float): CoV-ISI of the source produced by ``w``.
            fsamp (float): Sampling frequency in Hz.

        Returns:
            tuple:
                - ndarray: Filter from the last iteration.
                - ndarray: Spike indices from the last iteration.
                - float: CoV-ISI from the last iteration.
                - ndarray: Spike-detection cluster centroids.
        """

        cov_last = cov + 1
        centroids = None

        while cov < cov_last:
            source = w.T @ X
            spikes, _, centroids = est_spike_times(source, fsamp, cluster=self.cluster_method)
            cov_last = cov
            isi = np.diff(spikes / fsamp)
            cov = np.std(isi) / np.mean(isi)
            w = np.mean(X[:, spikes], axis=1)
            w = w / np.linalg.norm(w)

        return w, spikes, cov, centroids

    def _plot_source(self, source, spikes, sil, cov, fr, source_type, iteration, save_path=None):
        """Plot one source with its detected spikes.

        Args:
            source (ndarray): Source signal, (n_samples,).
            spikes (ndarray): Spike indices.
            sil (float): Silhouette score.
            cov (float): CoV-ISI.
            fr (float): Firing rate in Hz.
            source_type (str): One of 'accepted', 'rejected_cov', 'rejected_sil',
                'rejected_nospikes', 'rejected_earlystop', 'rejected_fr',
                'rejected_other'.
            iteration (int): Iteration index, used in the file name.
            save_path (str or None): Directory to write the figure to. None shows
                it instead.
        """
        fig, ax = plt.subplots(figsize=(12, 4))

        # Plot the source signal
        time = np.arange(len(source))
        ax.plot(time, source, 'k-', linewidth=0.5, alpha=0.7, label='Source')

        # Mark spike locations
        if len(spikes) > 0:
            ax.scatter(spikes, source[spikes], c='red', s=20, zorder=5, label='Spikes')

        # Color code by source type
        colors = {
            'accepted': 'green',
            'rejected_cov': 'orange',
            'rejected_sil': 'red',
            'rejected_nospikes': 'gray',
            'rejected_earlystop': 'brown',
            'rejected_fr': 'olive',
            'rejected_other': 'purple'
        }
        color = colors.get(source_type, 'blue')

        # Title with metrics
        title = f"Source {iteration} - {source_type.upper()}\n"
        title += f"Silhouette: {sil:.3f}, CoV: {cov:.3f}, FR: {fr:.2f} Hz, Spikes: {len(spikes)}"
        ax.set_title(title, fontsize=12, fontweight='bold', color=color)

        ax.set_xlabel('Sample')
        ax.set_ylabel('Amplitude')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
            filename = os.path.join(save_path, f"iter_{iteration:03d}_{source_type}.png")
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def _write_pipeline_sidecar(self):
        """Write the pipeline metadata into a json file."""
        # ToDo
        pass
