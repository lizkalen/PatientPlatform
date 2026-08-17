"""Real-time plot of LSL stream from Ripple device."""

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pylsl import StreamInlet, resolve_streams, resolve_byprop
from legacy_pipeline.streaming import stream_with_decomposition, stream_with_classification, plot_emg_stream

# Configuration
DISPLAY_SECONDS = 50  # How many seconds of data to show
MAX_CHANNELS = None  # Max channels to display (set lower if you have many)


def main():

    # stream_with_classification(
    #     decomp_path=r"C:\Users\velar\SynologyDrive\Personal\ExperimentalData\0602026Session1\online\processed_8reps\combined_multi_trial_model_8_23.pkl",
    #     mu1_idx=9,
    #     mu2_idx=1,
    #     emg_channel_to_display=0,
    #     max_channels=MAX_CHANNELS,
    #     display_seconds=DISPLAY_SECONDS,
    #     sil_threshold=0.1,
    #     enable_filtering=True
    # )

    plot_emg_stream(
        max_channels=MAX_CHANNELS,
        display_seconds=DISPLAY_SECONDS,
        enable_filtering=True
    )
if __name__ == "__main__":
    main()
