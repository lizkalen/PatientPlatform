import numpy as np
from scipy.stats import skew
import matplotlib.pyplot as plt
import os
from .core import bandpass_signals, notch_signals, extension, whitening, est_spike_times, remove_duplicates, remove_bad_sources, gram_schmidt, peel_off

class CBSS:
    """
    Class for performing convolutive blind source separation to identify the
    spiking activity of motor neurons using the fastICA algorithm.

    """

    def __init__(self, config=None, **kwargs):

        # Default parameters
        self.bandpass = [20, 500]
        self.bandpass_order = 2
        self.notch_frequency = 50
        self.notch_n_harmonics = 3
        self.notch_order = 2
        self.notch_width = 1
        self.ext_fact = 16
        self.whitening_method = "ZCA"
        self.whitening_reg = "auto"
        # Rank-truncated whitening: drop eigen-components with eigenvalue <=
        # whitening_rank_trunc * max (float ~1e-3) instead of amplifying the
        # near-null noise subspace. None = original behavior. See
        # processing/scripts/movement_disc_1307/mu_xclean/diagnosis for why this
        # matters on the near-singular extended-EMG covariance.
        self.whitening_rank_trunc = None
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
        self.sil_th = 0.9
        self.cov_th = 0.35
        self.min_firing_rate = 10 # in Hz
        self.match_th = 0.3
        self.match_max_shift = 0.1
        self.match_tol = 0.001
        
        self.verbose_mode = True
        self.output_source_plot = False
        self.plot_all_sources = False  # If True, plots all sources; if False, only plots accepted ones

        # Convert config object (if provided) to a dictionary
        config_dict = vars(config) if config is not None else {}

        # Merge with directly passed keyword arguments (overwrites config)
        params = {**config_dict, **kwargs}

        valid_keys = self.__dict__.keys()

        # Assign all parameters as attributes
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

    def decompose(self, sig, fsamp, from_scratch = True, just_preprocess=False, save_path=None):
        """
        Run simple decomposition

        Args:
            sig (ndarray): Input (EMG) signal (n_channels x n_samples)
            fsamp (float): Sampling frequency in Hz
            from_scratch (bool): Whether to start from scratch
            just_preprocess (bool): If True, return only preprocessed signals
            save_path (str): Directory path to save plots (if output_source_plot is True)

        Returns:
            sources (ndarray): Estimated spike responses (n_mu x n_samples)
            spikes (dict): Sample indices of motor neuron discharges
            sil (ndarray): Pseudo-silhouette scores of the estimated sources
            mu_filters (ndarray): Optimized motor unit filters
            Z (ndarray): Whitening matrix
            centroids (dict): K-means cluster centroids for each source
        """

        # Initalize random number generator
        rng = np.random.seed(self.random_seed)

        # Create save directory if plotting is enabled
        if self.output_source_plot and save_path is not None:
            os.makedirs(save_path, exist_ok=True)

        # Bandpass filter signals
        # if self.bandpass is not None:
        #     sig = bandpass_signals(
        #         sig,
        #         fsamp,
        #         high_pass=self.bandpass[0],
        #         low_pass=self.bandpass[1],
        #         order=self.bandpass_order,
        #     )

        # # Notch filter signals
        # if self.notch_frequency is not None:
        #     sig = notch_signals(
        #         sig,
        #         fsamp,
        #         nfreq=self.notch_frequency,
        #         dfreq=self.notch_width,
        #         order=self.notch_order,
        #         n_harmonics=self.notch_n_harmonics,
        #     )

        # Extend signals and subtract the mean and cut the edges
        print(f"[INFO] Extending signals by factor {self.ext_fact}...")
        ext_sig = extension(sig, self.ext_fact)
        ext_sig -= np.mean(ext_sig, axis=1, keepdims=True)

        # Remove the edges from the exteneded signal
        ext_sig[:, : self.ext_fact * 2] = 0
        ext_sig[:, -self.ext_fact * 2 :] = 0
        
        # Whiten the extended signals
        white_sig, Z = whitening(Y=ext_sig, method=self.whitening_method,
                                 regularization=self.whitening_reg,
                                 rank_trunc=self.whitening_rank_trunc)

        if just_preprocess:
            return white_sig, Z

        # Initalize the output variables
        sources = np.zeros((self.ica_n_iter, sig.shape[1]))
        spikes = {i: [] for i in range(self.ica_n_iter)}
        sil = np.zeros(self.ica_n_iter)
        covs = np.zeros(self.ica_n_iter)
        centroids = {i: None for i in range(self.ica_n_iter)}
        mu_filters = np.zeros((white_sig.shape[0], self.ica_n_iter))

        if self.opt_initalization == "activity_idx":
            act_idx_histoty = np.array([])

        if self.verbose_mode:
            print(f"\n{'='*80}")
            print(f"STARTING CBSS DECOMPOSITION")
            print(f"Max iterations: {self.ica_n_iter}, Silhouette threshold: {self.sil_th:.3f}, CoV threshold: {self.cov_th:.3f}")
            print(f"{'='*80}\n")

        # Loop over each MU
        for i in range(self.ica_n_iter):
            # Initalize
            if self.opt_initalization == "random":
                w = np.random.randn(white_sig.shape[0])
            elif self.opt_initalization == "activity_idx":
                col_norms = np.linalg.norm(white_sig, axis=0)
                col_norms[act_idx_histoty.astype(int)] = 0
                best_idx = np.argmax(col_norms)
                w = white_sig[:, best_idx]
                act_idx_histoty = np.append(act_idx_histoty, best_idx)
            else:
                ValueError("The specified initalization method is not implemented")

            # fastICA fixedpoint optimization
            w, k, early_stopped, early_stop_info = self.my_fixed_point_alg(w, white_sig, mu_filters, fsamp=fsamp)

            # Predict source and estimate the source quality
            sources[i, :] = w.T @ white_sig
            spikes[i], sil[i], centroids[i] = est_spike_times(
                sources[i, :], fsamp, cluster=self.cluster_method
            )
            if len(spikes[i]) > 2:
                isi = np.diff(spikes[i] / fsamp)
                cov = np.std(isi) / np.mean(isi)
            else:
                cov = np.inf

            # Refinement loop
            if len(spikes[i]) > 10 and self.refinement_loop:

                print(f"Refinement loop for source {i} (initial Sil: {sil[i]:.3f}, CoV: {cov:.3f}, Spikes: {len(spikes[i])})")
                w, _, cov, _ = self.mimimize_covisi(w, white_sig, cov, fsamp)
                sources[i, :] = w.T @ white_sig
                spikes[i], sil[i], centroids[i] = est_spike_times(
                    sources[i, :], fsamp, cluster=self.cluster_method
                )

            # Save the optimized MU filter
            mu_filters[:, i] = w
            covs[i] = cov

            # Calculate firing rate for logging
            if len(spikes[i]) > 0:
                fr = len(spikes[i]) / (len(sources[i, :]) / fsamp)
            else:
                fr = 0.0

            # Peel-off the detected source (skip if early stopped or firing rate too low)
            if self.peel_off and self.fr_peeloff:
                print(f"FR peel-off enabled")
                if sil[i] > self.sil_th and cov < self.cov_th and fr >= self.min_firing_rate and not early_stopped:
                    white_sig, _, _ = peel_off(white_sig, spikes[i], win=0.04, fsamp=fsamp)
                    if self.verbose_mode:
                        print(f"{i}: PEEL-OFF - Silhouette: {sil[i]:.3f} (threshold: {self.sil_th:.3f}), "
                            f"CoV: {cov:.3f} (threshold: {self.cov_th:.3f}), FR: {fr:.2f} Hz, "
                            f"Spikes: {len(spikes[i])}")
            elif self.peel_off and sil[i] > self.sil_th and cov < self.cov_th:
                print(f"FR peel-off disabled")
                white_sig, _, _ = peel_off(white_sig, spikes[i], win=0.04, fsamp=fsamp)
                if self.verbose_mode:
                    print(f"{i}: PEEL-OFF - Silhouette: {sil[i]:.3f} (threshold: {self.sil_th:.3f}), "
                        f"CoV: {cov:.3f} (threshold: {self.cov_th:.3f}), FR: {fr:.2f} Hz, "
                        f"Spikes: {len(spikes[i])}")

                # Plot accepted source
                if self.output_source_plot:
                    self._plot_source(sources[i, :], spikes[i], sil[i], cov, fr,
                                    'accepted', i, save_path)
            else:
                # Determine rejection reason and source type
                rejection_reasons = []
                source_type = 'rejected_other'

                if early_stopped and self.fr_peeloff:
                    rejection_reasons.append(f"early stopped at iter {early_stop_info['iteration']} "
                                           f"({early_stop_info['spike_count']} < {early_stop_info['threshold']} spikes)")
                    source_type = 'rejected_earlystop'
                if len(spikes[i]) == 0:
                    rejection_reasons.append("no spikes detected")
                    if source_type == 'rejected_other':
                        source_type = 'rejected_nospikes'
                if self.fr_peeloff and fr < self.min_firing_rate:
                    rejection_reasons.append(f"low firing rate ({fr:.2f} < {self.min_firing_rate:.2f} Hz)")
                    if source_type == 'rejected_other':
                        source_type = 'rejected_fr'
                if sil[i] <= self.sil_th:
                    rejection_reasons.append(f"low silhouette ({sil[i]:.3f} <= {self.sil_th:.3f})")
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
            print(f"POST-PROCESSING: Removing duplicates and bad sources")
            print(f"{'='*80}\n")

        # Remove duplicates
        n_sources_before_dup = sources.shape[0]
        sources_before_dup = sources.copy()
        spikes_before_dup = spikes.copy()
        sil_before_dup = sil.copy()

        sources, spikes, sil, mu_filters = remove_duplicates(
            sources,
            spikes,
            sil,
            mu_filters,
            fsamp,
            max_shift=self.match_max_shift,
            tol=self.match_tol,
            threshold=self.match_th,
        )
        n_duplicates_removed = n_sources_before_dup - sources.shape[0]
        if self.verbose_mode and n_duplicates_removed > 0:
            print(f"DUPLICATE REMOVAL: Removed {n_duplicates_removed} duplicate source(s) "
                  f"(match_th: {self.match_th:.3f}, max_shift: {self.match_max_shift:.3f})")

        # Remove bad sources and track which ones
        n_sources_before_removal = sources.shape[0]
        sources_before_bad = sources.copy()
        spikes_before_bad = spikes.copy()
        sil_before_bad = sil.copy()

        sources, spikes, sil, mu_filters, centroids = remove_bad_sources(
            sources,
            spikes,
            sil,
            mu_filters,
            centroids,
            covs,

            threshold=self.sil_th,
            max_cov=self.cov_th,
            min_firing_rate=self.min_firing_rate,
            fsamp=fsamp,
        )
        n_bad_removed = n_sources_before_removal - sources.shape[0]
        if self.verbose_mode and n_bad_removed > 0:
            print(f"BAD SOURCE REMOVAL: Removed {n_bad_removed} bad source(s) "
                  f"(sil_th: {self.sil_th:.3f}, min_firing_rate: {self.min_firing_rate:.2f} Hz)")

        # Show final accepted sources
        if self.verbose_mode:
            print(f"\n{'='*80}")
            print(f"FINAL ACCEPTED SOURCES: {sources.shape[0]} motor units")
            print(f"{'='*80}\n")

            for i in range(sources.shape[0]):
                if len(spikes[i]) > 1:
                    isi = np.diff(np.array(spikes[i]) / fsamp)
                    cov = np.std(isi) / np.mean(isi) if np.mean(isi) > 0 else np.inf
                    fr = len(spikes[i]) / (sources.shape[1] / fsamp)
                else:
                    cov = np.inf
                    fr = 0.0

                print(f"MU {i}: ACCEPTED - Silhouette: {sil[i]:.3f}, CoV: {cov:.3f}, "
                      f"FR: {fr:.2f} Hz, Spikes: {len(spikes[i])}")

            print(f"\n{'='*80}")
            print(f"DECOMPOSITION COMPLETE")
            print(f"{'='*80}\n")

        return sources, spikes, sil, mu_filters, Z, centroids

    def my_fixed_point_alg(self, w, X, B, fsamp=None):
        """
        Fixed-point optimization to maximize sparseness of a source signal.

        Args:
            w (np.ndarray): Initial weight vector (n_channels,)
            X (np.ndarray): Whitened signal matrix (n_channels x n_samples)
            B (np.ndarray): Current separation matrix (n_components x n_channels)
            fsamp (float): Sampling frequency in Hz (optional, for early stopping)

        Returns:
            w (np.ndarray): Optimized weight vector
            k (int): Number of iterations taken
            early_stopped (bool): True if algorithm stopped early due to low spike count
            early_stop_info (dict): Information about early stopping (spike count and threshold)
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
        # g = lambda x: x**2
        # gp = lambda x: 2*x

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
            w = np.mean(X * g(wTX), axis=1) - A * w  # shape: (n_channels,)

            # Orthogonalization step
            if self.source_deflation == "projection_deflation":
                w = w - (B @ B.T) @ w
            elif self.source_deflation == "gram-schmidt":
                w = gram_schmidt(w, B)
            else:
                pass

            # Normalize
            w = w / np.linalg.norm(w)

            # Convergence criterion
            delta[k + 1] = abs(np.dot(w, w_last) - 1)
            k += 1

            # Early stopping: check spike count every 20 iterations
            if min_spike_count is not None and k % 20 == 0:
                current_source = w.T @ X
                current_spikes, _, _ = est_spike_times(current_source, fsamp, cluster=self.cluster_method)
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
        """
        Iterativly update a motor unit filter given a set of motor neuron
        spike times as long as the coefficient of variance of the interspike
        intervall decreases.

        Args:
            w (np.ndarray): Initial weight vector
            X (np.ndarray): Whitened signal matrix (n_channels x n_samples)
            cov (float): Coefficient of variance of the initial source
            fsamp (float): Sampling rate in Hz

        Returns:
            w (np.ndarray): Optimized weight vector
            spikes (np.ndarray): Sample indices of motor neuron discharges
            cov (float): Coefficient of variance of the optimized source
            centroids (np.ndarray): Cluster centroids from final iteration

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
        """
        Plot a source signal with its detected spikes.

        Args:
            source (np.ndarray): Source signal
            spikes (np.ndarray): Spike indices
            sil (float): Silhouette score
            cov (float): Coefficient of variation
            fr (float): Firing rate in Hz
            source_type (str): Type of source ('accepted', 'rejected_cov', 'rejected_sil', 'rejected_nospikes')
            iteration (int): Iteration number
            save_path (str): Directory path to save the plot
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
        """
        Write the pipeline metadata into a json file.

        """
        # ToDo
        pass
