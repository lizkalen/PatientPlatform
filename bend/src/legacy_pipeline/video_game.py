import numpy as np
import sys
import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pylsl import StreamInlet, resolve_streams, resolve_byprop


from dsp.processing import process_chunk, calculate_firing_rates, design_filters, init_filter_states, apply_filters
from dsp.loading import load_pretrained_model


def stream_video_game(decomp_path, video_path, target_fr=70.0, max_channels=None, enable_filtering=True):
    """Play a video controlled by 2 motor units' firing rates.

    MU 0 firing rate -> video plays forward
    MU 1 firing rate -> video plays backward
    Speed is proportional to firing_rate / target_fr (capped at 1.0)

    Args:
        decomp_path: Path to decomposition results (must have exactly 2 MUs)
        video_path: Path to video file
        target_fr: Target firing rate for max speed (default 70 Hz)
        max_channels: Maximum EMG channels to use (None = all)
        enable_filtering: Apply bandpass and notch filters

    Returns:
        dict with game results
    """
    # Load video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video loaded: {total_frames} frames, {video_fps:.1f} fps, {video_width}x{video_height}")

    # Load first frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read first frame")
        cap.release()
        return None
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

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
        cap.release()
        return None

    inlet = StreamInlet(streams[0], max_buflen=360, processing_flags=1)
    info = inlet.info()
    srate = info.nominal_srate()

    # Load decomposition model (expecting 2 MUs)
    mu_filters, Z, n_mus, centroids, norm_factors, channels_to_remove = load_pretrained_model(decomp_path, srate, n_mus=None)

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

    # Game state
    frame_position = [total_frames // 2]  # Start in the middle
    firing_rates = np.zeros(2)
    spike_count_history = [[] for _ in range(n_mus)]
    prev_tail = [None]

    # Setup matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.subplots_adjust(bottom=0.15)

    # Video display
    img_display = ax.imshow(frame_rgb)
    ax.axis('off')

    # Speed indicator bar at the bottom
    ax_speed = fig.add_axes([0.1, 0.05, 0.8, 0.03])
    ax_speed.set_xlim(-1, 1)
    ax_speed.set_ylim(0, 1)
    ax_speed.axvline(0, color='white', linewidth=2)
    speed_bar = ax_speed.barh(0.5, 0, height=0.8, color='green')
    ax_speed.set_facecolor('gray')
    ax_speed.set_yticks([])
    ax_speed.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax_speed.set_xticklabels(['MU1 70Hz', '', 'Stop', '', 'MU0 70Hz'])

    # Text displays
    title_text = ax.set_title('Motor Unit Video Control', fontsize=14, fontweight='bold')
    info_text = ax.text(0.5, 0.02, '', transform=ax.transAxes, ha='center', fontsize=11,
                        color='white', bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

    frame_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, ha='left', va='top', fontsize=10,
                         color='white', bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

    def update(frame_num):
        nonlocal firing_rates, spike_count_history

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

        # Calculate speed based on firing rates
        # MU 0 -> forward, MU 1 -> backward
        speed_forward = min(firing_rates[0] / target_fr, 1.0)
        speed_backward = min(firing_rates[1] / target_fr, 1.0)
        net_speed = speed_forward - speed_backward  # -1 to 1

        # Update frame position
        # Scale: at max speed, move ~2 frames per update (50ms interval = 40 fps effective)
        frame_delta = net_speed * 2
        frame_position[0] = np.clip(frame_position[0] + frame_delta, 0, total_frames - 1)

        # Read the frame at current position
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_position[0]))
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_display.set_array(frame_rgb)

        # Update speed bar
        bar_color = 'blue' if net_speed > 0 else 'red' if net_speed < 0 else 'gray'
        speed_bar[0].set_width(net_speed)
        speed_bar[0].set_color(bar_color)

        # Update text
        info_text.set_text(f'MU0: {firing_rates[0]:.1f} Hz (forward)  |  MU1: {firing_rates[1]:.1f} Hz (backward)')
        frame_text.set_text(f'Frame: {int(frame_position[0])}/{total_frames}  |  Speed: {net_speed*100:.0f}%')

        return [img_display, speed_bar[0], info_text, frame_text]

    ani = FuncAnimation(fig, update, interval=50, blit=False, cache_frame_data=False)
    plt.show()

    # Cleanup
    cap.release()

    return {
        'final_frame': frame_position[0],
        'total_frames': total_frames,
    }
