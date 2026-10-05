"""Simple left/right cue game for motor unit control training."""

import numpy as np
import sys
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.animation import FuncAnimation
from pylsl import StreamInlet, resolve_streams, resolve_byprop
import time


from dsp.processing import process_chunk, design_filters, init_filter_states, apply_filters
from dsp.loading import load_pretrained_model


def calculate_firing_rates_custom(spikes, n_mus, chunk_duration, spike_history, window_size=20):
    """Calculate firing rates with configurable window size.

    Args:
        spikes: dict mapping MU index to array of spike indices for current chunk
        n_mus: number of motor units
        chunk_duration: duration of one chunk in seconds
        spike_history: list of lists with spike counts from previous chunks
        window_size: number of chunks to use for averaging (default 20)

    Returns:
        firing_rates: array of shape (n_mus,) with firing rates in Hz
        spike_history: updated history
    """
    for mu_idx in range(n_mus):
        current_count = len(spikes[mu_idx]) if mu_idx in spikes else 0
        spike_history[mu_idx].append(current_count)

        # Keep only the last window_size chunks
        if len(spike_history[mu_idx]) > window_size:
            spike_history[mu_idx] = spike_history[mu_idx][-window_size:]

    # Calculate firing rates (spikes per second)
    firing_rates = np.zeros(n_mus)
    for mu_idx in range(n_mus):
        total_spikes = sum(spike_history[mu_idx])
        total_duration = len(spike_history[mu_idx]) * chunk_duration
        firing_rates[mu_idx] = total_spikes / total_duration if total_duration > 0 else 0.0

    return firing_rates, spike_history


def stream_cue_game(decomp_path, n_reps=50, target_fr=70.0, threshold=0.3,
                    hold_time=0.5, max_channels=None, enable_filtering=True,
                    fr_window_size=20):
    """Simple left/right cue game controlled by motor unit firing rates.

    Uses difference between MU0 and MU1:
    - MU0 > MU1 → move RIGHT
    - MU1 > MU0 → move LEFT

    Args:
        decomp_path: Path to decomposition results (needs 2 MUs)
        n_reps: Number of repetitions (default 50)
        target_fr: Target firing rate for full movement (default 70 Hz)
        threshold: Threshold for success detection (fraction of target, default 0.3)
        hold_time: Time to hold correct direction (seconds, default 0.5)
        max_channels: Maximum EMG channels to use (None = all)
        enable_filtering: Apply bandpass and notch filters
        fr_window_size: Number of chunks for firing rate calculation (default 20)

    Returns:
        dict with game results
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
    mu_filters, Z, n_mus, centroids, norm_factors, channels_to_remove = load_pretrained_model(
        decomp_path, srate, n_mus=None
    )

    if n_mus < 2:
        print(f"Error: Need at least 2 MUs, got {n_mus}")
        return None
    if n_mus > 2:
        print(f"Warning: Using first 2 of {n_mus} MUs")
        n_mus = 2

    # Build list of channels to keep
    total_lsl_channels = info.channel_count()
    channels_to_remove_set = set(channels_to_remove) if channels_to_remove else set()
    channels_to_keep = [ch for ch in range(total_lsl_channels) if ch not in channels_to_remove_set]

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
    print(f"Target FR: {target_fr} Hz")
    print(f"Threshold: {threshold * 100:.0f}%")
    print(f"Hold time: {hold_time}s")
    print(f"FR window: {fr_window_size} chunks")
    print(f"Total reps: {n_reps}")

    # Game state
    firing_rates = np.zeros(2)
    spike_count_history = [[] for _ in range(n_mus)]
    prev_tail = [None]

    # Trial state
    current_rep = [0]
    successful_reps = [0]
    current_direction = [None]  # 'left' or 'right'
    in_correct_since = [None]
    game_running = [True]
    game_start_time = [None]
    trial_start_time = [None]

    # Results tracking
    trial_results = []  # List of dicts with trial info

    def generate_cue():
        """Generate a random left or right cue."""
        return np.random.choice(['left', 'right'])

    # Setup matplotlib figure
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_facecolor('#1a1a2e')

    # Title
    title_text = ax.text(0, 1.3, 'LEFT / RIGHT CUE GAME', ha='center', va='center',
                         fontsize=24, fontweight='bold', color='white')

    # Cue display area (center)
    cue_bg = Rectangle((-0.8, 0.2), 1.6, 0.8, fill=True, facecolor='#2d2d44',
                       edgecolor='white', linewidth=2)
    ax.add_patch(cue_bg)

    # Cue text (arrow direction)
    cue_text = ax.text(0, 0.6, '', ha='center', va='center', fontsize=72,
                       fontweight='bold', color='white')

    # Position indicator bar at bottom
    bar_bg = Rectangle((-1.0, -0.2), 2.0, 0.15, fill=True, facecolor='#333355',
                       edgecolor='white', linewidth=1)
    ax.add_patch(bar_bg)

    # Center line
    ax.axvline(0, ymin=0.1, ymax=0.2, color='white', linewidth=2, linestyle='--')

    # Position indicator (moves left/right)
    position_marker = Rectangle((-0.05, -0.18), 0.1, 0.11, fill=True,
                                 facecolor='cyan', edgecolor='white', linewidth=2)
    ax.add_patch(position_marker)

    # Labels
    ax.text(-1.0, -0.35, 'LEFT (MU1)', ha='center', fontsize=12, color='#ff6b6b')
    ax.text(1.0, -0.35, 'RIGHT (MU0)', ha='center', fontsize=12, color='#4ecdc4')

    # Score and status display
    score_text = ax.text(0, 1.1, '', ha='center', va='center', fontsize=16, color='white')
    status_text = ax.text(0, 0.05, '', ha='center', va='center', fontsize=14, color='yellow')
    fr_text = ax.text(0, -0.45, '', ha='center', va='center', fontsize=11, color='#888888')

    # Feedback indicator
    feedback_text = ax.text(0, 0.6, '', ha='center', va='center', fontsize=48,
                            fontweight='bold', color='lime')

    def update(frame_num):
        nonlocal firing_rates, spike_count_history

        if not game_running[0]:
            return [cue_text, position_marker, score_text, status_text, fr_text, feedback_text]

        # Initialize game on first frame
        if game_start_time[0] is None:
            game_start_time[0] = time.time()
            current_direction[0] = generate_cue()
            trial_start_time[0] = time.time()

        # Pull EMG samples
        samples, _ = inlet.pull_chunk(timeout=0.0, max_samples=int(srate))

        if samples:
            samples = np.array(samples).T
            samples = samples[channels_to_keep, :]
            n_new = samples.shape[1]

            # Apply filters
            if enable_filtering and filter_state:
                samples, filter_state['bp_zi'], filter_state['notch_zi'] = apply_filters(
                    samples, bp_sos, notch_sos, filter_state['bp_zi'], filter_state['notch_zi']
                )

            # Process chunk
            sources, spikes, sil_scores, prev_tail[0] = process_chunk(
                samples, mu_filters, Z, n_mus, srate, centroids, prev_tail[0], norm_factors
            )

            # Calculate firing rates with custom window size
            chunk_duration = n_new / srate
            spikes_dict = {i: spikes[i] for i in range(n_mus)}
            firing_rates, spike_count_history = calculate_firing_rates_custom(
                spikes_dict, n_mus, chunk_duration, spike_count_history, fr_window_size
            )

        # Compute control signal (difference-based)
        # Positive = right (MU0 dominant), Negative = left (MU1 dominant)
        diff = (firing_rates[0] - firing_rates[1]) / target_fr
        diff = np.clip(diff, -1.0, 1.0)

        # Update position marker
        position_marker.set_x(diff - 0.05)

        # Determine detected direction
        if abs(diff) > threshold:
            detected_direction = 'right' if diff > 0 else 'left'
        else:
            detected_direction = None

        current_time = time.time()
        correct = detected_direction == current_direction[0]

        # Update cue display
        if current_direction[0] == 'left':
            cue_text.set_text('\u2190')  # Left arrow
            cue_text.set_color('#ff6b6b')
        else:
            cue_text.set_text('\u2192')  # Right arrow
            cue_text.set_color('#4ecdc4')

        # Check if holding correct direction
        if correct and detected_direction is not None:
            if in_correct_since[0] is None:
                in_correct_since[0] = current_time

            hold_duration = current_time - in_correct_since[0]

            if hold_duration < hold_time:
                # Show progress
                progress = hold_duration / hold_time * 100
                status_text.set_text(f'HOLD: {progress:.0f}%')
                position_marker.set_facecolor('yellow')
            else:
                # Success!
                successful_reps[0] += 1
                trial_results.append({
                    'rep': current_rep[0] + 1,
                    'direction': current_direction[0],
                    'success': True,
                    'reaction_time': current_time - trial_start_time[0] - hold_time,
                    'fr_mu0': firing_rates[0],
                    'fr_mu1': firing_rates[1]
                })

                current_rep[0] += 1

                if current_rep[0] >= n_reps:
                    # Game complete
                    game_running[0] = False
                    cue_text.set_text('')
                    feedback_text.set_text('COMPLETE!')
                    feedback_text.set_color('lime')
                    status_text.set_text('')
                else:
                    # Next trial
                    current_direction[0] = generate_cue()
                    trial_start_time[0] = current_time
                    in_correct_since[0] = None
                    position_marker.set_facecolor('cyan')
        else:
            in_correct_since[0] = None
            status_text.set_text('')

            # Color marker based on correctness
            if detected_direction is None:
                position_marker.set_facecolor('gray')
            elif correct:
                position_marker.set_facecolor('cyan')
            else:
                position_marker.set_facecolor('#ff4444')

        # Update score display
        elapsed = current_time - game_start_time[0] if game_start_time[0] else 0
        score_text.set_text(f'Rep: {current_rep[0] + 1}/{n_reps}  |  Success: {successful_reps[0]}  |  Time: {elapsed:.1f}s')

        # Update FR display
        fr_text.set_text(f'MU0: {firing_rates[0]:.1f} Hz  |  MU1: {firing_rates[1]:.1f} Hz  |  Diff: {diff*100:.0f}%')

        return [cue_text, position_marker, score_text, status_text, fr_text, feedback_text]

    ani = FuncAnimation(fig, update, interval=50, blit=False, cache_frame_data=False)
    plt.show()

    # Calculate results
    total_time = time.time() - game_start_time[0] if game_start_time[0] else 0
    success_rate = successful_reps[0] / n_reps * 100 if n_reps > 0 else 0

    print("\n" + "=" * 50)
    print("GAME RESULTS")
    print("=" * 50)
    print(f"Total reps: {n_reps}")
    print(f"Successful: {successful_reps[0]}")
    print(f"Success rate: {success_rate:.1f}%")
    print(f"Total time: {total_time:.1f}s")

    if trial_results:
        successful_trials = [t for t in trial_results if t['success']]
        if successful_trials:
            reaction_times = [t['reaction_time'] for t in successful_trials]
            print(f"Mean reaction time: {np.mean(reaction_times):.2f}s")

    return {
        'n_reps': n_reps,
        'successful': successful_reps[0],
        'success_rate': success_rate,
        'total_time': total_time,
        'trial_results': trial_results,
        'target_fr': target_fr,
        'threshold': threshold,
        'hold_time': hold_time,
        'fr_window_size': fr_window_size
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Left/Right Cue Game with Motor Units")
    parser.add_argument("decomp_path", help="Path to decomposition results")
    parser.add_argument("--n-reps", type=int, default=50, help="Number of repetitions")
    parser.add_argument("--target-fr", type=float, default=70.0, help="Target firing rate")
    parser.add_argument("--threshold", type=float, default=0.3, help="Detection threshold (0-1)")
    parser.add_argument("--hold-time", type=float, default=0.5, help="Hold time for success (seconds)")
    parser.add_argument("--fr-window", type=int, default=20, help="Chunks for FR calculation")
    parser.add_argument("--no-filter", action="store_true", help="Disable filtering")

    args = parser.parse_args()

    results = stream_cue_game(
        args.decomp_path,
        n_reps=args.n_reps,
        target_fr=args.target_fr,
        threshold=args.threshold,
        hold_time=args.hold_time,
        fr_window_size=args.fr_window,
        enable_filtering=not args.no_filter
    )
