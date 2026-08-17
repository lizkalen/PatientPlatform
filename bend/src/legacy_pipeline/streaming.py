import numpy as np
import sys
import json
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Patch
from pylsl import StreamInlet, resolve_streams, resolve_byprop


from dsp.processing import (process_chunk, calculate_firing_rates, design_filters,
                         init_filter_states, apply_filters, firing_rate_sliding_window)
from dsp.loading import load_pretrained_model

from muniverse.algorithms.core import est_spike_times, extension

try:
    import serial
except ImportError:
    print("PySerial not installed. Bluetooth hand control will not work.")
    serial = None


def stream_with_decomposition(decomp_path, emg_channel_to_display=0, max_channels=None, display_seconds=10, enable_filtering=False, sil_threshold=0.75, max_display_mus=8):
    """Stream EMG with real-time decomposition, storing all data for later use.

    Args:
        decomp_path: Path to pretrained decomposition model
        emg_channel_to_display: Which EMG channel to show in the top plot
        max_channels: Maximum channels to use (None = all)
        display_seconds: Seconds of data to display
        enable_filtering: Apply 20-500 Hz bandpass and 50 Hz notch filters
        sil_threshold: SIL threshold for spike visualization
        max_display_mus: Maximum number of MUs to display (None = all). All MUs are still processed and stored.

    Returns:
        dict with 'emg', 'sources', 'spikes', 'firing_rates', 'sil_scores', 'srate', etc.
    """
    print("Looking for LSL stream...")
    try:
        streams = resolve_byprop("source_id", "RippleTrellis", timeout=5.0)
    except Exception:
        streams = []
    if not streams:
        # Fallback: try to find any stream
        streams = resolve_streams(wait_time=5.0)

    if not streams:
        print("No LSL stream found. Make sure lsl-ripple is running.")
        return None

    inlet = StreamInlet(streams[0], max_buflen=360, processing_flags=1)
    info = inlet.info()

    srate = info.nominal_srate()

    mu_filters, Z, n_mus, centroids, norm_factors, channels_to_remove = load_pretrained_model(
        decomp_path,
        srate,
        n_mus=None
    )

    # Build list of channels to keep (indices into raw LSL data)
    # channels_to_remove contains the original indices that were removed during calibration
    channels_to_remove_set = set(channels_to_remove) if channels_to_remove else set()

    # Limit display MUs while still processing all
    n_display_mus = min(n_mus, max_display_mus) if max_display_mus is not None else n_mus

    # Determine total channels from LSL stream
    total_lsl_channels = info.channel_count()

    # Build list of channel indices to keep (after removing calibration-excluded channels)
    channels_to_keep = [ch for ch in range(total_lsl_channels) if ch not in channels_to_remove_set]

    # Apply max_channels limit if specified (on top of removed channels)
    if max_channels is not None:
        channels_to_keep = channels_to_keep[:max_channels]

    n_channels = len(channels_to_keep)
    n_samples = int(srate * display_seconds)

    # Design and initialize filters for 50 Hz harmonic rejection and 20-500 Hz bandpass
    if enable_filtering:
        bp_sos, notch_sos = design_filters(srate, lowcut=20.0, highcut=500.0, notch_freq=50.0)
        bp_zi, notch_zi = init_filter_states(bp_sos, notch_sos, n_channels)
        filter_state = {'bp_zi': bp_zi, 'notch_zi': notch_zi}
    else:
        bp_sos, notch_sos, filter_state = None, None, None

    # Storage for all data (accumulated over time)
    stored_emg = []          # Raw/filtered EMG chunks (all channels)
    stored_sources = []      # MU source signal chunks
    stored_spikes = [[] for _ in range(n_mus)]  # Spike times per MU (in samples, cumulative)
    stored_firing_rates = [] # Firing rates per chunk
    stored_sil_scores = []   # SIL scores per chunk
    total_samples = [0]      # Track total samples for spike time offset

    source_data = np.zeros((n_mus, n_samples))
    emg_data = np.zeros(n_samples)  # Buffer for filtered EMG channel

    print(f"Connected to: {info.name()}")
    print(f"Sample rate: {srate} Hz")
    print(f"Channels: {total_lsl_channels} from LSL, {n_channels} after filtering (removed: {list(channels_to_remove_set) if channels_to_remove_set else 'none'})")
    print(f"Motor units: {n_mus} total (displaying {n_display_mus})")
    if enable_filtering:
        print(f"Filtering: 20-500 Hz bandpass, 50 Hz harmonics notch")
    else:
        print("Filtering: disabled")
    if sil_threshold > 0:
        print(f"SIL threshold: {sil_threshold} (spikes from MUs below this will be hidden)")

    # Initialize data buffer
    data = np.zeros((n_channels, n_samples))

    # Set up the plot
    # Add 1 extra row for the raw EMG channel at the top
    fig = plt.figure(figsize=(14, 2 + 2 * n_display_mus * 2))
    gs = fig.add_gridspec(n_display_mus * 2 + 1, 1,
                          height_ratios=[2] + [2, 1] * n_display_mus,
                          hspace=0.15)

    source_axes = []
    raster_axes = []
    source_lines = []
    raster_plots = []
    firing_rate_texts = []

    # Spike tracking for visualization (only for displayed MUs)
    spike_history = [[] for _ in range(n_display_mus)]
    sil_scores = np.zeros(n_mus)

    # Spike count history for firing rate calculation (over 5 chunks)
    spike_count_history = [[] for _ in range(n_mus)]
    firing_rates = np.zeros(n_mus)

    x = np.linspace(0, display_seconds, n_samples)

    # Create EMG channel plot at the top
    ax_emg = fig.add_subplot(gs[0, 0])
    emg_line, = ax_emg.plot(x, emg_data, 'g-', linewidth=0.8)
    ax_emg.set_ylabel(f'EMG Ch {emg_channel_to_display}', fontweight='bold', color='green')
    ax_emg.set_xlim(0, display_seconds)
    ax_emg.grid(True, alpha=0.3)
    ax_emg.set_xticklabels([])
    emg_title = 'Filtered EMG (20-500 Hz, 50 Hz harmonics removed)' if enable_filtering else 'Raw EMG Signal'
    ax_emg.set_title(emg_title, fontsize=10, loc='left')

    for mu_idx in range(n_display_mus):
        # Source signal plot (offset by 1 for EMG channel at top)
        ax_source = fig.add_subplot(gs[mu_idx * 2 + 1, 0])
        line, = ax_source.plot(x, source_data[mu_idx], 'b-', linewidth=0.8)
        source_lines.append(line)
        ax_source.set_ylabel(f'MU {mu_idx}', fontweight='bold')
        ax_source.set_xlim(0, display_seconds)
        ax_source.grid(True, alpha=0.3)
        ax_source.set_xticklabels([])
        source_axes.append(ax_source)

        # Add firing rate text annotation (top-right corner)
        fr_text = ax_source.text(0.98, 0.95, '0.0 Hz', transform=ax_source.transAxes,
                                  ha='right', va='top', fontsize=9, color='darkred',
                                  fontweight='bold', bbox=dict(boxstyle='round,pad=0.3',
                                  facecolor='white', edgecolor='gray', alpha=0.8))
        firing_rate_texts.append(fr_text)

        # Raster plot (offset by 1 for EMG channel at top)
        ax_raster = fig.add_subplot(gs[mu_idx * 2 + 2, 0])
        scatter = ax_raster.scatter([], [], c='red', marker='|', s=100, linewidths=2)
        raster_plots.append(scatter)
        ax_raster.set_xlim(0, display_seconds)
        ax_raster.set_ylim(-50, 50)
        ax_raster.set_yticks([])
        ax_raster.set_ylabel('Spikes', fontsize=9)
        ax_raster.grid(True, alpha=0.3)
        if mu_idx < n_display_mus - 1:
            ax_raster.set_xticklabels([])
        raster_axes.append(ax_raster)

    raster_axes[-1].set_xlabel('Time (s)')
    adaptation_str = "Fixed filters"
    spike_str = "Centroid-based" if centroids else "K-means"

    fig.suptitle(f'Motor Unit Activity - {adaptation_str} ({spike_str} spike detection, )',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()

    # State for overlap handling between chunks (avoids edge effects)
    prev_tail = [None]  # Mutable container for closure

    def update(frame):
        # Pull all available samples
        samples, _ = inlet.pull_chunk(timeout=0.0, max_samples=int(srate))

        if samples:
            samples = np.array(samples).T  # Shape: (all_lsl_channels, n_samples)
            # Keep only the channels that match calibration (remove excluded channels)
            samples = samples[channels_to_keep, :]
            n_new = samples.shape[1]

            # Safety: if a very large chunk arrives, only keep the newest part that fits the display buffer.
            if n_new > n_samples:
                samples = samples[:, -n_samples:]
                n_new = n_samples

            # Apply bandpass (20-500 Hz) and 50 Hz harmonic notch filters if enabled
            # Filter states are maintained across chunks to avoid edge effects
            if enable_filtering:
                samples, filter_state['bp_zi'], filter_state['notch_zi'] = apply_filters(
                    samples, bp_sos, notch_sos, filter_state['bp_zi'], filter_state['notch_zi']
                )

            # Update EMG data buffer with filtered signal
            emg_data[:-n_new] = emg_data[n_new:]
            emg_data[-n_new:] = samples[emg_channel_to_display, :]

            # Update EMG plot with dynamic y-axis
            emg_line.set_ydata(emg_data)
            emg_min, emg_max = np.min(emg_data), np.max(emg_data)
            emg_margin = (emg_max - emg_min) * 0.1 if emg_max != emg_min else 1.0
            ax_emg.set_ylim(emg_min - emg_margin, emg_max + emg_margin)


            # Decode motor units using fixed (offline) whitening + MU filters
            # Normalization is applied before spike detection if norm_factors available (Farina 2025)
            sources, spikes, sil_scores, prev_tail[0] = process_chunk(
                samples, mu_filters, Z, n_mus, srate, centroids, prev_tail[0], norm_factors
            )

            # Trim the prepended overlap samples from output and rebase spike indices
            src_len = sources.shape[1]
            trim_left = src_len - n_new  # Remove the R-1 prepended samples
            if trim_left > 0:
                sources = sources[:, trim_left:]
                rebased_spikes = []
                for mu_idx in range(n_mus):
                    mu_spikes = np.asarray(spikes[mu_idx], dtype=int)
                    if mu_spikes.size == 0:
                        rebased_spikes.append(mu_spikes)
                        continue
                    # Keep only spikes in the new portion and adjust indices
                    mu_spikes = mu_spikes[mu_spikes >= trim_left] - trim_left
                    rebased_spikes.append(mu_spikes)
                spikes = rebased_spikes

            # Calculate firing rates over rolling window of 5 chunks
            chunk_duration = n_new / srate
            spikes_dict = {i: spikes[i] for i in range(n_mus)}
            nonlocal spike_count_history, firing_rates
            firing_rates, spike_count_history = calculate_firing_rates(
                spikes_dict, n_mus, chunk_duration, spike_count_history
            )

            # Store data for later saving
            stored_emg.append(samples.copy())
            stored_sources.append(sources.copy())
            for mu_idx in range(n_mus):
                # Convert spike indices to absolute sample positions
                abs_spikes = np.asarray(spikes[mu_idx], dtype=int) + total_samples[0]
                stored_spikes[mu_idx].extend(abs_spikes.tolist())
            stored_firing_rates.append(firing_rates.copy())
            stored_sil_scores.append(sil_scores.copy())
            total_samples[0] += n_new

            # Update source signal buffer (all MUs)
            source_data[:, :-n_new] = source_data[:, n_new:]
            source_data[:, -n_new:] = sources

            # Update plots (only displayed MUs)
            for mu_idx in range(n_display_mus):
                # Update firing rate and SIL score text annotations
                sil_str = f'SIL: {sil_scores[mu_idx]:.2f}'
                if sil_scores[mu_idx] < sil_threshold:
                    sil_str += ' (filtered)'
                firing_rate_texts[mu_idx].set_text(f'{firing_rates[mu_idx]:.1f} Hz | {sil_str}')

                source_lines[mu_idx].set_ydata(source_data[mu_idx])

                # Dynamic y-axis: update limits based on current data
                y_data = source_data[mu_idx]
                y_min, y_max = np.min(y_data), np.max(y_data)
                margin = (y_max - y_min) * 0.1 if y_max != y_min else 1.0
                source_axes[mu_idx].set_ylim(y_min - margin, y_max + margin)

                # Shift existing spike times to the left (data scrolls)
                spike_history[mu_idx] = [t - n_new for t in spike_history[mu_idx] if t - n_new >= 0]

                # Add new spikes (convert to buffer indices) - only if SIL score >= threshold
                mu_spikes = spikes[mu_idx]
                if len(mu_spikes) > 0 and sil_scores[mu_idx] >= sil_threshold:
                    # New spikes are at the end of the buffer, offset from n_samples - n_new
                    new_spike_indices = (n_samples - n_new) + np.asarray(mu_spikes, dtype=int)
                    spike_history[mu_idx].extend(new_spike_indices.tolist())

                # Update raster plot
                spike_times = np.array(spike_history[mu_idx])
                if len(spike_times) > 0:
                    spike_seconds = (spike_times / n_samples) * display_seconds
                    spike_y = np.zeros_like(spike_seconds)
                    raster_plots[mu_idx].set_offsets(np.c_[spike_seconds, spike_y])
                else:
                    raster_plots[mu_idx].set_offsets(np.empty((0, 2)))


        return [emg_line] + source_lines + raster_plots + firing_rate_texts

    ani = FuncAnimation(fig, update, interval=50, blit=False, cache_frame_data=False)
    plt.show()

    # Concatenate and return all stored data after window closes
    if stored_emg:
        all_emg = np.hstack(stored_emg)
        all_sources = np.hstack(stored_sources)
        all_firing_rates = np.vstack(stored_firing_rates)
        all_sil_scores = np.vstack(stored_sil_scores)
        # Convert spike lists to arrays
        all_spikes = {mu: np.array(stored_spikes[mu], dtype=int) for mu in range(n_mus)}

        duration = all_emg.shape[1] / srate
        print(f"Stored {all_emg.shape[1]} samples ({duration:.2f} seconds)")
        print(f"  EMG: {all_emg.shape[0]} channels")
        print(f"  Sources: {all_sources.shape[0]} MUs")
        print(f"  Spikes per MU: {[len(all_spikes[mu]) for mu in range(n_mus)]}")

        return {
            'emg': all_emg,
            'sources': all_sources,
            'spikes': all_spikes,
            'firing_rates': all_firing_rates,
            'sil_scores': all_sil_scores,
            'srate': srate,
            'n_channels': n_channels,
            'n_mus': n_mus,
            'filtered': enable_filtering
        }
    else:
        print("No data was collected")
        return None


def plot_emg_stream(max_channels=None, display_seconds=50, enable_filtering=True):
    """Stream and display EMG data with optional filtering, storing all channels.

    Args:
        max_channels: Maximum number of channels to display (None = channel_count//4)
        display_seconds: Number of seconds to display in the plot
        enable_filtering: Whether to apply 20-500 Hz bandpass and 50 Hz notch filters

    Returns:
        dict with 'data' (filtered EMG array), 'srate' (sample rate), 'n_channels' (total channels)
    """
    print("Looking for LSL stream...")
    try:
        streams = resolve_byprop("source_id", "RippleTrellis", timeout=5.0)
    except Exception:
        streams = []
    if not streams:
        # Fallback: try to find any stream
        streams = resolve_streams(wait_time=5.0)

    if not streams:
        print("No LSL stream found. Make sure lsl-ripple is running.")
        return None

    inlet = StreamInlet(streams[0], max_buflen=360, processing_flags=1)
    info = inlet.info()

    srate = info.nominal_srate()
    total_channels = info.channel_count()  # Store ALL channels
    display_channels = min(total_channels, max_channels) if max_channels is not None else total_channels // 4
    n_samples = int(srate * display_seconds)

    # Design and initialize filters for all channels
    if enable_filtering:
        bp_sos, notch_sos = design_filters(srate, lowcut=20.0, highcut=500.0, notch_freq=50.0)
        bp_zi, notch_zi = init_filter_states(bp_sos, notch_sos, total_channels)
        filter_state = {'bp_zi': bp_zi, 'notch_zi': notch_zi}
    else:
        bp_sos, notch_sos, filter_state = None, None, None

    print(f"Connected to: {info.name()}")
    print(f"Sample rate: {srate} Hz")
    print(f"Channels: {total_channels} (displaying {display_channels})")
    if enable_filtering:
        print("Filtering: 20-500 Hz bandpass, 50 Hz harmonics notch")
    else:
        print("Filtering: disabled")

    # Initialize display buffer (only for displayed channels)
    display_data = np.zeros((display_channels, n_samples))

    # Storage for all channels (accumulated over time)
    stored_chunks = []

    # Set up the plot
    fig, axes = plt.subplots(display_channels, 1, figsize=(12, 2 * display_channels), sharex=True)
    if display_channels == 1:
        axes = [axes]

    lines = []
    x = np.linspace(0, display_seconds, n_samples)

    for i, ax in enumerate(axes):
        line, = ax.plot(x, display_data[i], 'b-', linewidth=0.5)
        lines.append(line)
        ax.set_ylabel(f'Ch {i+1}')
        ax.set_xlim(0, display_seconds)

    axes[-1].set_xlabel('Time (s)')
    title = 'Ripple Trellis - Real-time Signal (Filtered)' if enable_filtering else 'Ripple Trellis - Real-time Signal (Raw)'
    fig.suptitle(title)
    plt.tight_layout()

    def update(frame):
        # Pull all available samples
        samples, _ = inlet.pull_chunk(timeout=0.0, max_samples=int(srate))

        if samples:
            samples = np.array(samples).T  # Shape: (all_channels, samples)
            n_new = samples.shape[1]

            # Apply filters to ALL channels if enabled
            if enable_filtering:
                samples, filter_state['bp_zi'], filter_state['notch_zi'] = apply_filters(
                    samples, bp_sos, notch_sos, filter_state['bp_zi'], filter_state['notch_zi']
                )

            # Store the filtered (or raw) data for all channels
            stored_chunks.append(samples.copy())

            # Update display buffer (only for displayed channels)
            display_samples = samples[:display_channels]
            display_data[:, :-n_new] = display_data[:, n_new:]
            display_data[:, -n_new:] = display_samples

            # Update plot lines
            for i, line in enumerate(lines):
                line.set_ydata(display_data[i])
                axes[i].relim()
                axes[i].autoscale_view(scalex=False)

        return lines

    ani = FuncAnimation(fig, update, interval=50, blit=False, cache_frame_data=False)
    plt.show()

    # Concatenate all stored chunks after window closes
    if stored_chunks:
        all_data = np.hstack(stored_chunks)
        print(f"Stored {all_data.shape[1]} samples ({all_data.shape[1]/srate:.2f} seconds) across {all_data.shape[0]} channels")
        return {
            'data': all_data,
            'srate': srate,
            'n_channels': total_channels,
            'filtered': enable_filtering
        }
    else:
        print("No data was collected")
        return None



def stream_with_classification(
    decomp_path,
    mu1_idx=0,
    mu2_idx=1,
    firing_rate_threshold=10.0,
    window_sec=1.0,
    emg_channel_to_display=0,
    max_channels=None,
    display_seconds=10,
    enable_filtering=False,
    sil_threshold=0.75,
    bt_port=None,
):
    """Stream EMG with real-time decomposition and 2-MU movement classification.

    Classifies each time point as:
        0 = rest (neither MU above threshold)
        1 = movement 1 (both MU1 and MU2 firing above threshold)
        2 = movement 2 (only MU2 firing above threshold)

    Args:
        decomp_path: Path to pretrained decomposition model
        mu1_idx: Index of motor unit 1 for classification
        mu2_idx: Index of motor unit 2 for classification
        firing_rate_threshold: Firing rate threshold in Hz
        window_sec: Sliding window duration in seconds for firing rate computation
        emg_channel_to_display: Which EMG channel to show in the top plot
        max_channels: Maximum channels to use (None = all)
        display_seconds: Seconds of data to display
        enable_filtering: Apply 20-500 Hz bandpass and 50 Hz notch filters
        sil_threshold: SIL threshold for spike visualization
        bt_port: Optional Bluetooth COM port for hand control (e.g., "COM5").
                 If provided, sends trigger_open_close on rest↔movement transitions.

    Returns:
        dict with 'emg', 'sources', 'spikes', 'firing_rates', 'sil_scores',
        'srate', 'movement_labels', etc.
    """
    print("Looking for LSL stream...")
    try:
        streams = resolve_byprop("source_id", "RippleTrellis", timeout=5.0)
    except Exception:
        streams = []
    if not streams:
        streams = resolve_streams(wait_time=5.0)

    if not streams:
        print("No LSL stream found. Make sure lsl-ripple is running.")
        return None

    inlet = StreamInlet(streams[0], max_buflen=360, processing_flags=1)
    info = inlet.info()
    srate = info.nominal_srate()

    mu_filters, Z, n_mus, centroids, norm_factors, channels_to_remove = load_pretrained_model(
        decomp_path, srate, n_mus=None
    )

    if mu1_idx >= n_mus or mu2_idx >= n_mus:
        raise ValueError(f"mu1_idx={mu1_idx} or mu2_idx={mu2_idx} exceeds n_mus={n_mus}")

    channels_to_remove_set = set(channels_to_remove) if channels_to_remove else set()
    total_lsl_channels = info.channel_count()
    channels_to_keep = [ch for ch in range(total_lsl_channels) if ch not in channels_to_remove_set]
    if max_channels is not None:
        channels_to_keep = channels_to_keep[:max_channels]
    n_channels = len(channels_to_keep)
    n_samples = int(srate * display_seconds)

    if enable_filtering:
        bp_sos, notch_sos = design_filters(srate, lowcut=20.0, highcut=500.0, notch_freq=50.0)
        bp_zi, notch_zi = init_filter_states(bp_sos, notch_sos, n_channels)
        filter_state = {'bp_zi': bp_zi, 'notch_zi': notch_zi}
    else:
        bp_sos, notch_sos, filter_state = None, None, None

    # Storage for all data (accumulated over time)
    stored_emg = []
    stored_sources = []
    stored_spikes = [[] for _ in range(n_mus)]
    stored_firing_rates = []
    stored_sil_scores = []
    total_samples = [0]

    # Display buffers
    source_data = np.zeros((n_mus, n_samples))
    emg_data = np.zeros(n_samples)
    # Spike train buffers for the two classification MUs (display window)
    spike_train_mu1 = np.zeros(n_samples)
    spike_train_mu2 = np.zeros(n_samples)
    # Movement label buffer for display window
    movement_label = np.zeros(n_samples, dtype=int)

    # --- Bluetooth hand control setup ---
    bt_serial = None
    send_id = [1]  # Mutable counter for send IDs
    prev_is_active = [False]  # Track previous state: False=rest, True=any movement

    def send_bt_command(command, args=None):
        """Send command to Bluetooth hand."""
        if bt_serial is None:
            return
        if args is None:
            args = []
        msg = json.dumps({"command": command, "send": send_id[0], "args": args})
        print(f"[BT] TX -> {msg}")
        try:
            bt_serial.write(msg.encode("utf-8"))
            send_id[0] += 1
            # Optional: read response (non-blocking)
            bt_serial.timeout = 0.1
            data = bt_serial.read(4096)
            if data:
                text = data.decode("utf-8", errors="replace")
                print(f"[BT] RX <- {text}")
        except Exception as e:
            print(f"[BT] Error: {e}")

    if bt_port is not None:
        if serial is None:
            print(f"Warning: PySerial not installed. Cannot connect to {bt_port}")
        else:
            try:
                print(f"Connecting to Bluetooth hand on {bt_port}...")
                bt_serial = serial.Serial(bt_port, timeout=2)
                print(f"[BT] Connected. Testing heartbeat...")
                send_bt_command("heart_beat")
                print("[BT] Hand control enabled.\n")
            except Exception as e:
                print(f"[BT] Failed to connect to {bt_port}: {e}")
                bt_serial = None

    print(f"Connected to: {info.name()}")
    print(f"Sample rate: {srate} Hz")
    print(f"Channels: {total_lsl_channels} from LSL, {n_channels} after filtering")
    print(f"Motor units: {n_mus} total, classifying with MU{mu1_idx} & MU{mu2_idx}")
    print(f"Classification: threshold={firing_rate_threshold} Hz, window={window_sec}s")

    # Spike tracking for raster visualization
    spike_history_mu1 = []
    spike_history_mu2 = []
    sil_scores = np.zeros(n_mus)
    spike_count_history = [[] for _ in range(n_mus)]
    firing_rates = np.zeros(n_mus)

    x = np.linspace(0, display_seconds, n_samples)

    # --- Plot layout: EMG + classification, MU1 raster, MU2 raster, firing rates ---
    fig, axes = plt.subplots(
        4, 1,
        figsize=(16, 10),
        sharex=True,
        gridspec_kw={'height_ratios': [2.5, 1, 1, 1.5]}
    )

    # Panel 0: EMG with classification overlay
    ax_emg = axes[0]
    emg_line, = ax_emg.plot(x, emg_data, 'k-', linewidth=0.4, alpha=0.8)
    ax_emg.set_ylabel(f'EMG Ch {emg_channel_to_display}', fontsize=10)
    ax_emg.set_title(
        f'Movement Classification — MU {mu1_idx} & MU {mu2_idx} '
        f'(threshold={firing_rate_threshold} Hz, window={window_sec}s)',
        fontsize=12, fontweight='bold'
    )
    ax_emg.set_xlim(0, display_seconds)
    ax_emg.grid(True, alpha=0.3)
    ax_emg.legend(
        handles=[
            Patch(color='blue', alpha=0.25, label=f'Movement 1 (both MU{mu1_idx} & MU{mu2_idx})'),
            Patch(color='orange', alpha=0.25, label=f'Movement 2 (only MU{mu2_idx})'),
        ],
        loc='upper right', fontsize=9
    )
    # Store vspan patches for efficient clearing
    classification_patches = []

    # Panel 1: MU1 raster
    ax1 = axes[1]
    scatter1 = ax1.scatter([], [], c='tab:blue', marker='|', s=100, linewidths=2)
    ax1.set_ylabel(f'MU {mu1_idx}', fontsize=10, color='tab:blue')
    ax1.set_xlim(0, display_seconds)
    ax1.set_ylim(-0.5, 0.5)
    ax1.set_yticks([])
    ax1.grid(True, alpha=0.3, axis='x')

    # Panel 2: MU2 raster
    ax2 = axes[2]
    scatter2 = ax2.scatter([], [], c='tab:orange', marker='|', s=100, linewidths=2)
    ax2.set_ylabel(f'MU {mu2_idx}', fontsize=10, color='tab:orange')
    ax2.set_xlim(0, display_seconds)
    ax2.set_ylim(-0.5, 0.5)
    ax2.set_yticks([])
    ax2.grid(True, alpha=0.3, axis='x')

    # Panel 3: Firing rates
    ax_fr = axes[3]
    fr_line1, = ax_fr.plot(x, np.zeros(n_samples), color='tab:blue', linewidth=1, label=f'MU {mu1_idx} FR')
    fr_line2, = ax_fr.plot(x, np.zeros(n_samples), color='tab:orange', linewidth=1, label=f'MU {mu2_idx} FR')
    ax_fr.axhline(firing_rate_threshold, color='red', linestyle='--', linewidth=1, alpha=0.7,
                  label=f'Threshold ({firing_rate_threshold} Hz)')
    ax_fr.set_ylabel('Firing Rate (Hz)', fontsize=10)
    ax_fr.set_xlabel('Time (s)', fontsize=10)
    ax_fr.set_xlim(0, display_seconds)
    ax_fr.legend(loc='upper right', fontsize=9)
    ax_fr.grid(True, alpha=0.3)

    plt.tight_layout()

    prev_tail = [None]

    def update(frame):
        nonlocal spike_count_history, firing_rates, classification_patches

        samples, _ = inlet.pull_chunk(timeout=0.0, max_samples=int(srate))

        if samples:
            samples = np.array(samples).T
            samples = samples[channels_to_keep, :]
            n_new = samples.shape[1]

            if n_new > n_samples:
                samples = samples[:, -n_samples:]
                n_new = n_samples

            if enable_filtering:
                samples, filter_state['bp_zi'], filter_state['notch_zi'] = apply_filters(
                    samples, bp_sos, notch_sos, filter_state['bp_zi'], filter_state['notch_zi']
                )

            # Update EMG display buffer
            emg_data[:-n_new] = emg_data[n_new:]
            emg_data[-n_new:] = samples[emg_channel_to_display, :]
            emg_line.set_ydata(emg_data)
            emg_min, emg_max = np.min(emg_data), np.max(emg_data)
            emg_margin = (emg_max - emg_min) * 0.1 if emg_max != emg_min else 1.0
            ax_emg.set_ylim(emg_min - emg_margin, emg_max + emg_margin)

            # Decompose
            sources, spikes, sil_scores_chunk, prev_tail[0] = process_chunk(
                samples, mu_filters, Z, n_mus, srate, centroids, prev_tail[0], norm_factors
            )

            # Trim overlap
            src_len = sources.shape[1]
            trim_left = src_len - n_new
            if trim_left > 0:
                sources = sources[:, trim_left:]
                rebased_spikes = []
                for mu_idx in range(n_mus):
                    mu_spikes = np.asarray(spikes[mu_idx], dtype=int)
                    if mu_spikes.size == 0:
                        rebased_spikes.append(mu_spikes)
                        continue
                    mu_spikes = mu_spikes[mu_spikes >= trim_left] - trim_left
                    rebased_spikes.append(mu_spikes)
                spikes = rebased_spikes

            # Calculate firing rates (chunk-based, for storage)
            chunk_duration = n_new / srate
            spikes_dict = {i: spikes[i] for i in range(n_mus)}
            firing_rates, spike_count_history = calculate_firing_rates(
                spikes_dict, n_mus, chunk_duration, spike_count_history
            )

            # Store data
            stored_emg.append(samples.copy())
            stored_sources.append(sources.copy())
            for mu_idx in range(n_mus):
                abs_spikes = np.asarray(spikes[mu_idx], dtype=int) + total_samples[0]
                stored_spikes[mu_idx].extend(abs_spikes.tolist())
            stored_firing_rates.append(firing_rates.copy())
            stored_sil_scores.append(sil_scores_chunk.copy())
            total_samples[0] += n_new

            # Update source buffer
            source_data[:, :-n_new] = source_data[:, n_new:]
            source_data[:, -n_new:] = sources

            # --- Update spike train buffers for classification ---
            spike_train_mu1[:-n_new] = spike_train_mu1[n_new:]
            spike_train_mu1[-n_new:] = 0.0
            mu1_spk = np.asarray(spikes[mu1_idx], dtype=int)
            if mu1_spk.size > 0:
                valid = mu1_spk[(mu1_spk >= 0) & (mu1_spk < n_new)]
                spike_train_mu1[n_samples - n_new + valid] = 1.0

            spike_train_mu2[:-n_new] = spike_train_mu2[n_new:]
            spike_train_mu2[-n_new:] = 0.0
            mu2_spk = np.asarray(spikes[mu2_idx], dtype=int)
            if mu2_spk.size > 0:
                valid = mu2_spk[(mu2_spk >= 0) & (mu2_spk < n_new)]
                spike_train_mu2[n_samples - n_new + valid] = 1.0

            # --- Compute firing rates over display window ---
            spk_times_mu1 = np.where(spike_train_mu1 > 0)[0]
            spk_times_mu2 = np.where(spike_train_mu2 > 0)[0]

            fr_mu1 = firing_rate_sliding_window(spk_times_mu1, n_samples, srate, window_sec)
            fr_mu2 = firing_rate_sliding_window(spk_times_mu2, n_samples, srate, window_sec)

            # --- Classify ---
            mu1_active = fr_mu1 >= firing_rate_threshold
            mu2_active = fr_mu2 >= firing_rate_threshold

            movement_label[:] = 0
            movement_label[mu1_active & mu2_active] = 1
            movement_label[~mu1_active & mu2_active] = 2

            # --- Detect transitions and trigger hand control ---
            # Check the most recent classification (rightmost sample in display window)
            current_label = movement_label[-1]
            current_is_active = (current_label > 0)  # True if movement 1 or 2, False if rest

            if current_is_active != prev_is_active[0]:
                # Transition detected
                if current_is_active:
                    print(f"[TRANSITION] rest → movement {current_label}")
                    send_bt_command("trigger_open_close")
                else:
                    print(f"[TRANSITION] movement → rest")
                    send_bt_command("trigger_open_close")
                prev_is_active[0] = current_is_active

            # --- Update classification overlay ---
            for p in classification_patches:
                p.remove()
            classification_patches = []

            color_map = {1: 'blue', 2: 'orange'}
            for label_val, color in color_map.items():
                mask = (movement_label == label_val).astype(int)
                d = np.diff(np.concatenate(([0], mask, [0])))
                starts = np.where(d == 1)[0]
                ends = np.where(d == -1)[0]
                for s, e in zip(starts, ends):
                    patch = ax_emg.axvspan(
                        s / n_samples * display_seconds,
                        e / n_samples * display_seconds,
                        color=color, alpha=0.25
                    )
                    classification_patches.append(patch)

            # --- Update firing rate plot ---
            fr_line1.set_ydata(fr_mu1)
            fr_line2.set_ydata(fr_mu2)
            fr_max = max(np.max(fr_mu1), np.max(fr_mu2), firing_rate_threshold * 1.5)
            ax_fr.set_ylim(0, fr_max * 1.1 if fr_max > 0 else 1.0)

            # --- Update raster plots ---
            spike_history_mu1[:] = [t - n_new for t in spike_history_mu1 if t - n_new >= 0]
            spike_history_mu2[:] = [t - n_new for t in spike_history_mu2 if t - n_new >= 0]

            if mu1_spk.size > 0 and sil_scores_chunk[mu1_idx] >= sil_threshold:
                new_indices = (n_samples - n_new) + mu1_spk
                spike_history_mu1.extend(new_indices.tolist())
            if mu2_spk.size > 0 and sil_scores_chunk[mu2_idx] >= sil_threshold:
                new_indices = (n_samples - n_new) + mu2_spk
                spike_history_mu2.extend(new_indices.tolist())

            spike_times_1 = np.array(spike_history_mu1)
            if len(spike_times_1) > 0:
                spike_sec_1 = (spike_times_1 / n_samples) * display_seconds
                scatter1.set_offsets(np.c_[spike_sec_1, np.zeros_like(spike_sec_1)])
            else:
                scatter1.set_offsets(np.empty((0, 2)))

            spike_times_2 = np.array(spike_history_mu2)
            if len(spike_times_2) > 0:
                spike_sec_2 = (spike_times_2 / n_samples) * display_seconds
                scatter2.set_offsets(np.c_[spike_sec_2, np.zeros_like(spike_sec_2)])
            else:
                scatter2.set_offsets(np.empty((0, 2)))

        return [emg_line, fr_line1, fr_line2, scatter1, scatter2]

    ani = FuncAnimation(fig, update, interval=50, blit=False, cache_frame_data=False)
    plt.show()

    # Close Bluetooth connection
    if bt_serial is not None:
        try:
            bt_serial.close()
            print("[BT] Connection closed.")
        except Exception as e:
            print(f"[BT] Error closing connection: {e}")

    # Return all stored data after window closes
    if stored_emg:
        all_emg = np.hstack(stored_emg)
        all_sources = np.hstack(stored_sources)
        all_firing_rates = np.vstack(stored_firing_rates)
        all_sil_scores = np.vstack(stored_sil_scores)
        all_spikes = {mu: np.array(stored_spikes[mu], dtype=int) for mu in range(n_mus)}

        # Compute final classification over full stored signal
        n_total = all_sources.shape[1]
        fr_mu1_full = firing_rate_sliding_window(all_spikes[mu1_idx], n_total, srate, window_sec)
        fr_mu2_full = firing_rate_sliding_window(all_spikes[mu2_idx], n_total, srate, window_sec)
        all_movement_labels = np.zeros(n_total, dtype=int)
        all_movement_labels[(fr_mu1_full >= firing_rate_threshold) & (fr_mu2_full >= firing_rate_threshold)] = 1
        all_movement_labels[~(fr_mu1_full >= firing_rate_threshold) & (fr_mu2_full >= firing_rate_threshold)] = 2

        duration = all_emg.shape[1] / srate
        rest_time = np.sum(all_movement_labels == 0) / srate
        mov1_time = np.sum(all_movement_labels == 1) / srate
        mov2_time = np.sum(all_movement_labels == 2) / srate

        print(f"Stored {all_emg.shape[1]} samples ({duration:.2f} seconds)")
        print(f"  EMG: {all_emg.shape[0]} channels")
        print(f"  Sources: {all_sources.shape[0]} MUs")
        print(f"  Classification summary:")
        print(f"    Rest:       {rest_time:.1f}s ({100*rest_time/duration:.1f}%)")
        print(f"    Movement 1: {mov1_time:.1f}s ({100*mov1_time/duration:.1f}%) — both MU{mu1_idx} & MU{mu2_idx}")
        print(f"    Movement 2: {mov2_time:.1f}s ({100*mov2_time/duration:.1f}%) — only MU{mu2_idx}")

        return {
            'emg': all_emg,
            'sources': all_sources,
            'spikes': all_spikes,
            'firing_rates': all_firing_rates,
            'sil_scores': all_sil_scores,
            'movement_labels': all_movement_labels,
            'srate': srate,
            'n_channels': n_channels,
            'n_mus': n_mus,
            'filtered': enable_filtering,
            'mu1_idx': mu1_idx,
            'mu2_idx': mu2_idx,
            'firing_rate_threshold': firing_rate_threshold,
            'window_sec': window_sec,
        }
    else:
        print("No data was collected")
        return None
