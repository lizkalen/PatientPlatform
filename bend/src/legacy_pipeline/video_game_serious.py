import numpy as np
import sys
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation
from pylsl import StreamInlet, resolve_streams, resolve_byprop
import time
from typing import List, Optional


from dsp.processing import process_chunk, calculate_firing_rates, design_filters, init_filter_states, apply_filters
from dsp.loading import load_pretrained_model
from .dof_config import DOFConfig, compute_dof_control_signals


def stream_cursor_control(decomp_path, target_fr=70.0, max_channels=None, enable_filtering=True,
                          n_targets=10, target_tolerance=0.1, target_hold_time=0.5,
                          workspace_size=1.0, dof_configs: Optional[List[DOFConfig]] = None):
    """Control a 2D cursor using motor units' firing rates.

    Control mapping depends on dof_configs:
    - If dof_configs is None: MU0 -> X, MU1 -> Y (legacy behavior)
    - If dof_configs provided: Uses configured control schemes per DOF

    DOF types:
    - SINGLE: One MU's FR controls the DOF
    - MEAN: Average FR of multiple MUs
    - SUM: Sum of FRs (both must be active for max control)
    - DIFFERENCE: |FR1 - FR2| controls the DOF

    Args:
        decomp_path: Path to decomposition results
        target_fr: Target firing rate for max position (default 70 Hz)
        max_channels: Maximum EMG channels to use (None = all)
        enable_filtering: Apply bandpass and notch filters
        n_targets: Number of targets to complete before ending
        target_tolerance: Radius around target for success (fraction of workspace)
        target_hold_time: Time cursor must stay in target zone (seconds)
        workspace_size: Size of the workspace (default 1.0)
        dof_configs: List of DOFConfig objects defining control scheme per axis

    Returns:
        dict with:
            - Task results: targets_completed, target_times, target_positions, cursor_positions
            - EMG data: emg, sources, spikes, firing_rates, sil_scores (same format as stream_with_decomposition)
            - Metadata: srate, n_channels, n_mus, filtered, dof_configs
    """
    # Connect to LSL stream
    print("Looking for LSL stream...")
    try:
        streams = resolve_byprop("source_id", "RippleTrellis", timeout=5.0)
    except Exception:
        streams = []
    if not streams:
        streams = resolve_streams(wait_time=5.0)

    if not streams:
        print("No LSL stream found.")
        return None

    inlet = StreamInlet(streams[0], max_buflen=360, processing_flags=1)
    info = inlet.info()
    srate = info.nominal_srate()

    # Load decomposition model (expecting 2 MUs)
    mu_filters, Z, n_mus, centroids, norm_factors, channels_to_remove = load_pretrained_model(decomp_path, srate, n_mus=None)

    # Validate MU count based on DOF configs
    if dof_configs is not None:
        # Check that all MU indices in configs are valid
        max_mu_idx = max(mu for cfg in dof_configs for mu in cfg.motor_units)
        if max_mu_idx >= n_mus:
            print(f"Error: DOF config references MU{max_mu_idx} but only {n_mus} MUs available.")
            return None
        print(f"Using DOF configurations with {n_mus} motor units")
        for cfg in dof_configs:
            print(f"  {cfg.describe()}")
    else:
        # Legacy behavior: need exactly 2 MUs
        if n_mus != 2:
            print(f"Warning: Expected 2 MUs, got {n_mus}. Using first 2.")
            n_mus = min(n_mus, 2)

    # Build list of channels to keep (remove calibration-excluded channels)
    total_lsl_channels = info.channel_count()
    channels_to_remove_set = set(channels_to_remove) if channels_to_remove else set()
    channels_to_keep = [ch for ch in range(total_lsl_channels) if ch not in channels_to_remove_set]

    # Apply max_channels limit if specified
    if max_channels is not None:
        channels_to_keep = channels_to_keep[:max_channels]

    n_channels = len(channels_to_keep)

    # Setup filters
    if enable_filtering:
        bp_sos, notch_sos = design_filters(srate, lowcut=20.0, highcut=500.0, notch_freq=50.0)
        bp_zi, notch_zi = init_filter_states(bp_sos, notch_sos, n_channels)
        filter_state = {'bp_zi': bp_zi, 'notch_zi': notch_zi}
    else:
        bp_sos, notch_sos, filter_state = None, None, None

    print(f"Connected to: {info.name()}")
    print(f"Sample rate: {srate} Hz")
    print(f"Target firing rate: {target_fr} Hz")
    print(f"Number of targets: {n_targets}")
    print(f"Target tolerance: {target_tolerance * 100:.0f}% of workspace")
    print(f"Hold time required: {target_hold_time:.1f}s")

    # Game state
    cursor_pos = np.array([workspace_size / 2, workspace_size / 2])
    firing_rates = np.zeros(2)
    spike_count_history = [[] for _ in range(n_mus)]
    prev_tail = [None]

    # Target state
    current_target = [None]
    targets_completed = [0]
    in_target_since = [None]
    game_start_time = [None]
    game_running = [True]

    # Results tracking
    target_times = []
    target_positions = []
    cursor_positions = []  # Store cursor position at each update

    # Data storage (same format as stream_with_decomposition)
    stored_emg = []
    stored_sources = []
    stored_spikes = [[] for _ in range(n_mus)]
    stored_firing_rates = []
    stored_sil_scores = []
    total_samples = [0]

    def generate_target():
        """Generate a new random target position."""
        margin = target_tolerance * workspace_size
        x = np.random.uniform(margin, workspace_size - margin)
        y = np.random.uniform(margin, workspace_size - margin)
        return np.array([x, y])

    # Setup matplotlib figure
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(0, workspace_size)
    ax.set_ylim(0, workspace_size)
    ax.set_aspect('equal')
    ax.set_facecolor('#1a1a2e')
    # Set axis labels based on DOF configuration
    if dof_configs is not None:
        ax.set_xlabel(dof_configs[0].describe(), fontsize=12)
        ax.set_ylabel(dof_configs[1].describe(), fontsize=12)
    else:
        ax.set_xlabel('MU 0 Firing Rate', fontsize=12)
        ax.set_ylabel('MU 1 Firing Rate', fontsize=12)

    # Target zone (tolerance circle)
    target_zone = Circle((0.5, 0.5), target_tolerance * workspace_size,
                         fill=True, facecolor='green', alpha=0.3,
                         edgecolor='lime', linewidth=2)
    ax.add_patch(target_zone)

    # Target center
    target_marker, = ax.plot([], [], 'o', color='lime', markersize=15, markeredgecolor='white', markeredgewidth=2)

    # Cursor
    cursor_marker, = ax.plot([], [], 'o', color='cyan', markersize=20, markeredgecolor='white', markeredgewidth=3)

    # Trail for cursor (last few positions)
    trail_length = 20
    trail_positions = []
    trail_line, = ax.plot([], [], '-', color='cyan', alpha=0.3, linewidth=2)

    # Text displays
    title_text = ax.set_title('2D Cursor Control Task', fontsize=16, fontweight='bold', color='white')

    info_text = ax.text(0.5, 0.98, '', transform=ax.transAxes, ha='center', va='top', fontsize=12,
                        color='white', bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

    fr_text = ax.text(0.02, 0.02, '', transform=ax.transAxes, ha='left', va='bottom', fontsize=11,
                      color='white', bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

    hold_text = ax.text(0.5, 0.5, '', transform=ax.transAxes, ha='center', va='center', fontsize=24,
                        color='yellow', fontweight='bold')

    def update(frame_num):
        nonlocal firing_rates, spike_count_history, cursor_pos, trail_positions

        if not game_running[0]:
            return [cursor_marker, target_marker, target_zone, trail_line, info_text, fr_text, hold_text]

        # Initialize game on first frame
        if game_start_time[0] is None:
            game_start_time[0] = time.time()
            current_target[0] = generate_target()
            target_positions.append(current_target[0].copy())

        # Pull EMG samples
        samples, _ = inlet.pull_chunk(timeout=0.0, max_samples=int(srate))

        if samples:
            samples = np.array(samples).T  # Shape: (all_lsl_channels, n_samples)
            # Keep only the channels that match calibration (remove excluded channels)
            samples = samples[channels_to_keep, :]
            n_new = samples.shape[1]

            # Apply filters
            if enable_filtering and filter_state:
                samples, filter_state['bp_zi'], filter_state['notch_zi'] = apply_filters(
                    samples, bp_sos, notch_sos, filter_state['bp_zi'], filter_state['notch_zi']
                )

            # Process chunk to get spikes (with normalization if available)
            sources, spikes, sil_scores, prev_tail[0] = process_chunk(
                samples, mu_filters, Z, n_mus, srate, centroids, prev_tail[0], norm_factors
            )

            # Calculate firing rates
            chunk_duration = n_new / srate
            spikes_dict = {i: spikes[i] for i in range(n_mus)}
            firing_rates, spike_count_history = calculate_firing_rates(
                spikes_dict, n_mus, chunk_duration, spike_count_history
            )

            # Store data for later saving
            stored_emg.append(samples.copy())
            stored_sources.append(sources.copy())
            for mu_idx in range(n_mus):
                abs_spikes = np.asarray(spikes[mu_idx], dtype=int) + total_samples[0]
                stored_spikes[mu_idx].extend(abs_spikes.tolist())
            stored_firing_rates.append(firing_rates.copy())
            stored_sil_scores.append(sil_scores.copy())
            total_samples[0] += n_new

        # Update cursor position based on firing rates
        if dof_configs is not None:
            # Use DOF configurations
            control_signals = compute_dof_control_signals(firing_rates, dof_configs, target_fr)
            cursor_pos[0] = control_signals[0] * workspace_size
            cursor_pos[1] = control_signals[1] * workspace_size
        else:
            # Legacy: direct MU to axis mapping
            cursor_pos[0] = min(firing_rates[0] / target_fr, 1.0) * workspace_size
            cursor_pos[1] = min(firing_rates[1] / target_fr, 1.0) * workspace_size

        # Store cursor position with timestamp
        cursor_positions.append({
            'pos': cursor_pos.copy(),
            'time': time.time() - game_start_time[0] if game_start_time[0] else 0,
            'firing_rates': firing_rates.copy(),
            'target_idx': targets_completed[0]
        })

        # Update trail
        trail_positions.append(cursor_pos.copy())
        if len(trail_positions) > trail_length:
            trail_positions.pop(0)

        # Check if cursor is in target zone
        distance_to_target = np.linalg.norm(cursor_pos - current_target[0])
        in_target = distance_to_target <= target_tolerance * workspace_size

        current_time = time.time()

        if in_target:
            if in_target_since[0] is None:
                in_target_since[0] = current_time

            hold_duration = current_time - in_target_since[0]

            # Show hold progress
            if hold_duration < target_hold_time:
                progress = hold_duration / target_hold_time * 100
                hold_text.set_text(f'{progress:.0f}%')
                target_zone.set_facecolor('yellow')
            else:
                # Target acquired!
                hold_text.set_text('')
                target_times.append(current_time - game_start_time[0])
                targets_completed[0] += 1

                if targets_completed[0] >= n_targets:
                    # Game complete
                    game_running[0] = False
                    info_text.set_text(f'COMPLETE! {n_targets} targets in {target_times[-1]:.1f}s')
                    hold_text.set_text('DONE!')
                    target_zone.set_visible(False)
                    target_marker.set_visible(False)
                else:
                    # Generate new target
                    current_target[0] = generate_target()
                    target_positions.append(current_target[0].copy())
                    in_target_since[0] = None
                    target_zone.set_facecolor('green')
        else:
            in_target_since[0] = None
            hold_text.set_text('')
            target_zone.set_facecolor('green')

        # Update visual elements
        cursor_marker.set_data([cursor_pos[0]], [cursor_pos[1]])

        if current_target[0] is not None:
            target_marker.set_data([current_target[0][0]], [current_target[0][1]])
            target_zone.set_center(current_target[0])

        if trail_positions:
            trail_x = [p[0] for p in trail_positions]
            trail_y = [p[1] for p in trail_positions]
            trail_line.set_data(trail_x, trail_y)

        # Update text
        elapsed = current_time - game_start_time[0] if game_start_time[0] else 0
        info_text.set_text(f'Targets: {targets_completed[0]}/{n_targets}  |  Time: {elapsed:.1f}s')
        # Build FR text based on DOF configs
        if dof_configs is not None:
            fr_parts = []
            for i, cfg in enumerate(dof_configs):
                axis = 'X' if i == 0 else 'Y'
                ctrl_signal = cfg.compute_control_signal(firing_rates, target_fr) * 100
                fr_parts.append(f'{axis}: {ctrl_signal:.0f}%')
            fr_text.set_text('  |  '.join(fr_parts))
        else:
            fr_text.set_text(f'MU0 (X): {firing_rates[0]:.1f} Hz  |  MU1 (Y): {firing_rates[1]:.1f} Hz')

        return [cursor_marker, target_marker, target_zone, trail_line, info_text, fr_text, hold_text]

    ani = FuncAnimation(fig, update, interval=50, blit=False, cache_frame_data=False)
    plt.show()

    # Concatenate stored data
    if stored_emg:
        all_emg = np.hstack(stored_emg)
        all_sources = np.hstack(stored_sources)
        all_firing_rates = np.vstack(stored_firing_rates)
        all_sil_scores = np.vstack(stored_sil_scores)
        all_spikes = {mu: np.array(stored_spikes[mu], dtype=int) for mu in range(n_mus)}

        duration = all_emg.shape[1] / srate
        print(f"\nStored {all_emg.shape[1]} samples ({duration:.2f} seconds)")
        print(f"  EMG: {all_emg.shape[0]} channels")
        print(f"  Sources: {all_sources.shape[0]} MUs")
        print(f"  Spikes per MU: {[len(all_spikes[mu]) for mu in range(n_mus)]}")
    else:
        all_emg = None
        all_sources = None
        all_firing_rates = None
        all_sil_scores = None
        all_spikes = None
        print("No EMG data was collected")

    # Serialize DOF configs for storage
    dof_configs_data = None
    if dof_configs is not None:
        from .dof_config import serialize_dof_configs
        dof_configs_data = serialize_dof_configs(dof_configs)

    # Return results (compatible with stream_with_decomposition format + task-specific data)
    return {
        # Task-specific results
        'targets_completed': targets_completed[0],
        'total_targets': n_targets,
        'target_times': target_times,
        'target_positions': [p.tolist() for p in target_positions],
        'cursor_positions': cursor_positions,
        'total_time': target_times[-1] if target_times else None,
        'mean_acquisition_time': np.mean(np.diff([0] + target_times)) if len(target_times) > 0 else None,
        'target_fr': target_fr,
        'target_tolerance': target_tolerance,
        'target_hold_time': target_hold_time,
        'workspace_size': workspace_size,
        'dof_configs': dof_configs_data,
        # EMG data (same format as stream_with_decomposition)
        'emg': all_emg,
        'sources': all_sources,
        'spikes': all_spikes,
        'firing_rates': all_firing_rates,
        'sil_scores': all_sil_scores,
        'srate': srate,
        'n_channels': n_channels,
        'n_mus': n_mus,
        'filtered': enable_filtering,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="2D Cursor Control with Motor Units")
    parser.add_argument("decomp_path", help="Path to decomposition results")
    parser.add_argument("--target-fr", type=float, default=70.0, help="Target firing rate for max position")
    parser.add_argument("--n-targets", type=int, default=10, help="Number of targets to complete")
    parser.add_argument("--tolerance", type=float, default=0.1, help="Target tolerance (fraction of workspace)")
    parser.add_argument("--hold-time", type=float, default=0.5, help="Time to hold in target (seconds)")
    parser.add_argument("--no-filter", action="store_true", help="Disable filtering")

    args = parser.parse_args()

    results = stream_cursor_control(
        args.decomp_path,
        target_fr=args.target_fr,
        n_targets=args.n_targets,
        target_tolerance=args.tolerance,
        target_hold_time=args.hold_time,
        enable_filtering=not args.no_filter,
    )

    if results:
        print("\n--- Results ---")
        print(f"Targets completed: {results['targets_completed']}/{results['total_targets']}")
        if results['total_time']:
            print(f"Total time: {results['total_time']:.1f}s")
            print(f"Mean acquisition time: {results['mean_acquisition_time']:.2f}s per target")
