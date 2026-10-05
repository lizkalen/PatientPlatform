import numpy as np
import matplotlib.pyplot as plt


def plot_raster_summary(streaming_results, emg_channel=0, units_per_plot=4, save_path=None):
    """Generate raster plots of all MUs with EMG channel on top.

    Args:
        streaming_results: Dictionary returned from stream_with_decomposition
        emg_channel: Which EMG channel to display on top
        units_per_plot: Number of MUs per figure (default 4)
        save_path: Base path to save figures (will append _raster_1.png, _raster_2.png, etc.)

    Returns:
        List of figure objects
    """
    if streaming_results is None:
        print("No streaming results to plot")
        return []

    emg = streaming_results['emg']
    spikes = streaming_results['spikes']
    srate = streaming_results['srate']
    n_mus = streaming_results['n_mus']

    # Time axis for EMG
    n_samples = emg.shape[1]
    time = np.arange(n_samples) / srate

    # Colors for different MUs
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    # Calculate number of figures needed
    n_figures = int(np.ceil(n_mus / units_per_plot))
    figures = []

    for fig_idx in range(n_figures):
        start_mu = fig_idx * units_per_plot
        end_mu = min(start_mu + units_per_plot, n_mus)
        n_units_this_fig = end_mu - start_mu

        # Create figure: EMG on top + one row per MU
        fig, axes = plt.subplots(
            n_units_this_fig + 1, 1,
            figsize=(14, 2 + 1.5 * n_units_this_fig),
            sharex=True,
            gridspec_kw={'height_ratios': [2] + [1] * n_units_this_fig}
        )

        # Plot EMG channel on top
        axes[0].plot(time, emg[emg_channel], 'k-', linewidth=0.5, alpha=0.8)
        axes[0].set_ylabel(f'EMG Ch {emg_channel}', fontsize=10)
        axes[0].set_title(f'Motor Unit Raster Plot (Units {start_mu}-{end_mu-1})', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3)

        # Plot raster for each MU
        for i, mu_idx in enumerate(range(start_mu, end_mu)):
            ax = axes[i + 1]
            color = colors[mu_idx % len(colors)]

            mu_spikes = spikes.get(mu_idx, np.array([]))
            if len(mu_spikes) > 0:
                spike_times = mu_spikes / srate
                # Plot spikes as points
                ax.scatter(
                    spike_times,
                    np.zeros_like(spike_times),
                    c=[color],
                    marker='|',
                    s=100,
                    linewidths=2,
                    label=f'MU {mu_idx}'
                )

            ax.set_ylabel(f'MU {mu_idx}', fontsize=10, color=color)
            ax.set_ylim(-0.5, 0.5)
            ax.set_yticks([])
            ax.grid(True, alpha=0.3, axis='x')

            # Add spike count annotation
            n_spikes = len(mu_spikes)
            duration = n_samples / srate
            avg_fr = n_spikes / duration if duration > 0 else 0
            ax.text(
                0.98, 0.5,
                f'{n_spikes} spikes ({avg_fr:.1f} Hz)',
                transform=ax.transAxes,
                ha='right', va='center',
                fontsize=9, color=color,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.8)
            )

        axes[-1].set_xlabel('Time (s)', fontsize=10)
        plt.tight_layout()
        figures.append(fig)

        # Save figure if path provided
        if save_path:
            fig_path = f"{save_path}_raster_{fig_idx+1}.png"
            fig.savefig(fig_path, dpi=150, bbox_inches='tight')
            print(f"Saved raster plot to: {fig_path}")

    plt.show()
    return figures
