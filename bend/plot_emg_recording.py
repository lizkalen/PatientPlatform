"""
EMG Recording Visualizer
========================

Visualizes EMG recordings with timeline events overlaid.
Shows phase transitions (tutorial, prep, move, rest, transition) as colored regions.

Usage:
    python plot_emg_recording.py <recording.pkl> [--channels 0 1 2] [--output fig.png]
"""

import argparse
import pickle as pkl
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Phase colors (matching traffic light theme from frontend)
PHASE_COLORS = {
    'tutorial': '#9C27B0',    # Purple
    'prep': '#FFC107',        # Yellow/Amber
    'move': '#4CAF50',        # Green
    'rest': '#F44336',        # Red
    'transition': '#2196F3',  # Blue
    'idle': '#9E9E9E',        # Gray
}

PHASE_LABELS = {
    'tutorial': 'Tutorial',
    'prep': 'Prep',
    'move': 'Move',
    'rest': 'Rest',
    'transition': 'Transition',
    'idle': 'Idle',
}


def load_recording(filepath: str) -> Dict[str, Any]:
    """Load a .pkl recording file."""
    with open(filepath, 'rb') as f:
        return pkl.load(f)


def extract_phase_regions(timeline: Dict, total_duration_ms: float) -> List[tuple]:
    """
    Extract phase regions from timeline events.

    Returns list of (start_ms, duration_ms, phase_name) tuples.
    """
    if not timeline or 'events' not in timeline:
        return []

    regions = []
    phase_events = [e for e in timeline['events'] if e['type'] == 'phase_change']

    for i, event in enumerate(phase_events):
        start_ms = event['offsetMs']
        phase = event.get('phase', 'idle')

        # Calculate duration until next phase or end
        if i + 1 < len(phase_events):
            end_ms = phase_events[i + 1]['offsetMs']
        else:
            end_ms = total_duration_ms

        duration_ms = end_ms - start_ms
        if duration_ms > 0:
            regions.append((start_ms, duration_ms, phase))

    return regions


def plot_emg_with_timeline(
    data: np.ndarray,
    srate: float,
    timeline: Optional[Dict] = None,
    channels: Optional[List[int]] = None,
    title: str = "EMG Recording",
    figsize: tuple = (14, 8),
    pre_trigger_s: float = 0.0,
) -> plt.Figure:
    """
    Plot EMG data with timeline phase regions overlaid.

    Args:
        data: EMG data array [channels x samples]
        srate: Sample rate in Hz
        timeline: Session timeline dict with events
        channels: List of channel indices to plot (None = all, max 8)
        title: Plot title
        figsize: Figure size
        pre_trigger_s: Pre-trigger time in seconds (shifts phase regions to align with data)

    Returns:
        matplotlib Figure object
    """
    n_channels, n_samples = data.shape
    duration_s = n_samples / srate
    duration_ms = duration_s * 1000
    time = np.linspace(0, duration_s, n_samples)

    # Select channels to plot
    if channels is None:
        channels = list(range(min(n_channels, 8)))  # Max 8 channels by default
    channels = [c for c in channels if 0 <= c < n_channels]
    n_plot = len(channels)

    if n_plot == 0:
        raise ValueError("No valid channels to plot")

    # Create figure
    fig, axes = plt.subplots(n_plot, 1, figsize=figsize, sharex=True)
    if n_plot == 1:
        axes = [axes]

    # Extract phase regions from timeline
    phase_regions = extract_phase_regions(timeline, duration_ms) if timeline else []

    # Track which phases are present for legend
    phases_present = set()

    for ax_idx, ch_idx in enumerate(channels):
        ax = axes[ax_idx]

        # Plot phase regions as background colors (shifted by pre-trigger)
        for start_ms, dur_ms, phase in phase_regions:
            start_s = (start_ms / 1000) + pre_trigger_s
            dur_s = dur_ms / 1000
            color = PHASE_COLORS.get(phase, PHASE_COLORS['idle'])
            ax.axvspan(start_s, start_s + dur_s, alpha=0.3, color=color, linewidth=0)
            phases_present.add(phase)

        # Mark pre-trigger region if present
        if pre_trigger_s > 0:
            ax.axvspan(0, pre_trigger_s, alpha=0.15, color='#607D8B', linewidth=0, hatch='//')
            ax.axvline(pre_trigger_s, color='#607D8B', linestyle='--', linewidth=1, alpha=0.7)

        # Plot EMG signal
        ax.plot(time, data[ch_idx], 'k-', linewidth=0.5, alpha=0.8)

        # Style
        ax.set_ylabel(f'Ch {ch_idx}', fontsize=10)
        ax.set_xlim(0, duration_s)
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.tick_params(axis='both', labelsize=8)

        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # X-axis label on bottom plot
    axes[-1].set_xlabel('Time (s)', fontsize=11)

    # Title
    fig.suptitle(title, fontsize=13, fontweight='bold')

    # Create legend for phases
    if phases_present or pre_trigger_s > 0:
        legend_patches = []
        # Add pre-trigger to legend if present
        if pre_trigger_s > 0:
            legend_patches.append(
                Patch(facecolor='#607D8B', alpha=0.3, label=f'Pre-trigger ({pre_trigger_s:.0f}s)')
            )
        # Add phase colors
        legend_patches.extend([
            Patch(facecolor=PHASE_COLORS.get(p, '#999'), alpha=0.5,
                  label=PHASE_LABELS.get(p, p.title()))
            for p in ['tutorial', 'prep', 'move', 'rest', 'transition']
            if p in phases_present
        ])
        fig.legend(
            handles=legend_patches,
            loc='upper right',
            bbox_to_anchor=(0.99, 0.99),
            fontsize=9,
            framealpha=0.9,
        )

    plt.tight_layout()
    plt.subplots_adjust(top=0.93)

    return fig


def print_recording_info(recording: Dict):
    """Print summary information about the recording."""
    print("\n" + "="*60)
    print("RECORDING INFO")
    print("="*60)

    data = recording.get('data')
    if data is not None:
        n_ch, n_samp = data.shape
        srate = recording.get('srate', 0)
        duration = n_samp / srate if srate else 0
        print(f"  Channels:      {n_ch}")
        print(f"  Samples:       {n_samp:,}")
        print(f"  Sample rate:   {srate} Hz")
        print(f"  Duration:      {duration:.2f} s")
        print(f"  Filtered:      {recording.get('filtered', 'N/A')}")
        print(f"  Pre-trigger:   {recording.get('pre_trigger_seconds', 'N/A')} s")
        print(f"  Stream type:   {recording.get('stream_type', 'N/A')}")
        print(f"  Timestamp:     {recording.get('timestamp', 'N/A')}")

    # Trial metadata
    meta = recording.get('trial_metadata')
    if meta:
        print("\n" + "-"*40)
        print("TRIAL METADATA")
        print("-"*40)
        print(f"  Subject:       {meta.get('subjectId', 'N/A')}")
        print(f"  Session:       {meta.get('sessionId', 'N/A')}")
        print(f"  Sequence:      {meta.get('sequenceName', 'N/A')}")
        print(f"  Total items:   {meta.get('totalItems', 'N/A')}")
        print(f"  Start time:    {meta.get('startTime', 'N/A')}")
        if meta.get('notes'):
            print(f"  Notes:         {meta.get('notes')}")

        settings = meta.get('settings', {})
        if settings:
            print(f"\n  Settings:")
            print(f"    Prep time:         {settings.get('prepTime', 0)} s")
            print(f"    Rest between reps: {settings.get('restBetweenReps', 0)} s")
            print(f"    Pause between:     {settings.get('pauseBetweenItems', 0)} s")
            print(f"    Playback speed:    {settings.get('playbackSpeed', 1.0)}x")

        items = meta.get('items', [])
        if items:
            print(f"\n  Sequence items ({len(items)}):")
            for item in items:
                print(f"    [{item.get('index')}] {item.get('model')} / anim {item.get('animation')} x{item.get('repetitions')}")

    # Session timeline
    timeline = recording.get('session_timeline')
    if timeline:
        print("\n" + "-"*40)
        print("SESSION TIMELINE")
        print("-"*40)
        print(f"  Session start: {timeline.get('sessionStart', 'N/A')}")
        print(f"  Session end:   {timeline.get('sessionEnd', 'N/A')}")
        total_ms = timeline.get('totalDurationMs', 0)
        print(f"  Total duration: {total_ms/1000:.2f} s")

        summary = timeline.get('summary', {})
        if summary:
            print(f"\n  Phase durations:")
            print(f"    Tutorial:    {summary.get('totalTutorialTimeMs', 0)/1000:.2f} s")
            print(f"    Prep:        {summary.get('totalPrepTimeMs', 0)/1000:.2f} s")
            print(f"    Move:        {summary.get('totalMoveTimeMs', 0)/1000:.2f} s")
            print(f"    Rest:        {summary.get('totalRestTimeMs', 0)/1000:.2f} s")
            print(f"    Transition:  {summary.get('totalTransitionTimeMs', 0)/1000:.2f} s")
            print(f"\n  Completed:")
            print(f"    Items:       {summary.get('itemsCompleted', 0)}")
            print(f"    Repetitions: {summary.get('totalRepetitions', 0)}")

        events = timeline.get('events', [])
        if events:
            print(f"\n  Timeline events ({len(events)}):")
            for event in events[:20]:  # Limit to first 20
                offset_s = event.get('offsetMs', 0) / 1000
                etype = event.get('type', '')
                if etype == 'phase_change':
                    phase = event.get('phase', '')
                    item = event.get('item', 0)
                    rep = event.get('rep', 0)
                    print(f"    {offset_s:7.2f}s  {etype:15} -> {phase:12} (item {item}, rep {rep})")
                elif etype in ('rep_complete', 'item_complete'):
                    item = event.get('item', 0)
                    rep = event.get('rep', '')
                    print(f"    {offset_s:7.2f}s  {etype:15} item {item}" + (f", rep {rep}" if rep else ""))
                else:
                    print(f"    {offset_s:7.2f}s  {etype}")
            if len(events) > 20:
                print(f"    ... and {len(events) - 20} more events")
    else:
        print("\n  (No session timeline data)")

    print("="*60 + "\n")


def save_phase_segments(
    data: np.ndarray,
    srate: float,
    timeline: Optional[Dict],
    pre_trigger_s: float,
    output_dir: Path,
    recording: Dict,
):
    """
    Save EMG data segments for each phase into separate .pkl files.

    Each phase occurrence gets its own file, e.g.:
        prep_001.pkl, prep_002.pkl, move_001.pkl, move_002.pkl, ...

    Args:
        data: EMG data array [channels x samples]
        srate: Sample rate in Hz
        timeline: Session timeline dict with events
        pre_trigger_s: Pre-trigger time in seconds
        output_dir: Directory to save the segment files
        recording: Original recording dict (for metadata)
    """
    _, n_samples = data.shape
    duration_ms = (n_samples / srate) * 1000

    phase_regions = extract_phase_regions(timeline, duration_ms) if timeline else []
    if not phase_regions:
        print("No phase regions found in timeline. Nothing to save.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Count occurrences per phase for numbering
    phase_counts: Dict[str, int] = {}

    for start_ms, dur_ms, phase in phase_regions:
        # Convert to sample indices (accounting for pre-trigger shift)
        start_s = (start_ms / 1000) + pre_trigger_s
        end_s = start_s + (dur_ms / 1000)

        start_idx = max(0, int(round(start_s * srate)))
        end_idx = min(n_samples, int(round(end_s * srate)))

        if end_idx <= start_idx:
            continue

        # Increment phase counter
        phase_counts[phase] = phase_counts.get(phase, 0) + 1
        count = phase_counts[phase]

        segment_data = data[:, start_idx:end_idx]

        segment = {
            'data': segment_data,
            'srate': srate,
            'phase': phase,
            'phase_occurrence': count,
            'start_s': start_s,
            'end_s': end_s,
            'n_samples': segment_data.shape[1],
            'duration_s': (end_idx - start_idx) / srate,
            'filtered': recording.get('filtered'),
            'stream_type': recording.get('stream_type'),
            'trial_metadata': recording.get('trial_metadata'),
        }

        filename = f"{phase}_{count:03d}.pkl"
        filepath = output_dir / filename
        with open(filepath, 'wb') as f:
            pkl.dump(segment, f)

        print(f"  Saved {filepath.name:25s}  ({segment_data.shape[1]:>7,} samples, {segment['duration_s']:.2f}s)")

    print(f"\nSaved {sum(phase_counts.values())} segments to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize EMG recording with timeline events",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('recording', type=str, help='Path to .pkl recording file')
    parser.add_argument('--channels', '-c', type=int, nargs='+', default=None,
                        help='Channel indices to plot (default: first 8)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Save figure to file (e.g., output.png)')
    parser.add_argument('--no-show', action='store_true',
                        help='Do not display the plot (useful with --output)')
    parser.add_argument('--info-only', action='store_true',
                        help='Only print recording info, do not plot')
    parser.add_argument('--divide', action='store_true',
                        help='Save EMG segments for each phase into <filename>_divided/ folder')
    parser.add_argument('--figsize', type=float, nargs=2, default=[14, 8],
                        help='Figure size in inches (width height)')

    args = parser.parse_args()

    # Load recording
    filepath = Path(args.recording)
    if not filepath.exists():
        print(f"Error: File not found: {filepath}")
        return 1

    print(f"Loading: {filepath}")
    recording = load_recording(filepath)

    # Print info
    print_recording_info(recording)

    if args.info_only:
        return 0

    # Extract data
    data = recording.get('data')
    srate = recording.get('srate')
    timeline = recording.get('session_timeline')
    meta = recording.get('trial_metadata', {})

    if data is None or srate is None:
        print("Error: Recording missing data or sample rate")
        return 1

    # Build title
    title_parts = []
    if meta.get('subjectId'):
        title_parts.append(f"Subject: {meta['subjectId']}")
    if meta.get('sessionId'):
        title_parts.append(f"Session: {meta['sessionId']}")
    if meta.get('sequenceName'):
        title_parts.append(f"Sequence: {meta['sequenceName']}")
    title = " | ".join(title_parts) if title_parts else f"EMG Recording ({filepath.stem})"

    # Get pre-trigger offset
    pre_trigger_s = recording.get('pre_trigger_seconds', 0) or 0

    # Save phase segments if requested
    if args.divide:
        divided_dir = filepath.parent / f"{filepath.stem}_divided"
        print(f"\nDividing recording by phase...")
        save_phase_segments(
            data=data,
            srate=srate,
            timeline=timeline,
            pre_trigger_s=pre_trigger_s,
            output_dir=divided_dir,
            recording=recording,
        )

    # Plot
    fig = plot_emg_with_timeline(
        data=data,
        srate=srate,
        timeline=timeline,
        channels=args.channels,
        title=title,
        figsize=tuple(args.figsize),
        pre_trigger_s=pre_trigger_s,
    )

    # Save if requested
    if args.output:
        output_path = Path(args.output)
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to: {output_path}")

    # Show plot
    if not args.no_show:
        plt.show()

    return 0


if __name__ == '__main__':
    exit(main())
