# Ripple Analysis Functions

# Loading/Import related packages
import sys
import os

#Usual suspects
import pandas as pd
import numpy as np
import json
import pickle as pkl
import matplotlib.pyplot as plt

#Extras for plotting
from matplotlib.patches import Patch

#Extra for typing
from collections import defaultdict

import glob 

# --------------------------------------------------------------------------
#   Load data and visualize
# --------------------------------------------------------------------------

def get_task_data(s_dir, n_reps, move_names, sampling_freq, win=None):
    """
    Load EMG data from multiple pickle files, 
    then combine repitions per movement type, with the option of specifying a window within each repition to keep.

    Parameters:
    - s_dir: str, directory where the pickle files are located.
    - n_reps: int, number of repetitions per movement.
    - movement_names: list of str, names of movements sorted in the order of collection in task.
    - file_type: str, the type of files to load (e.g., 'move' for movement files, or 'rest' for rest files).
    - sampling_freq: float, sampling frequency of data collection in Hz.
    - win: float, optional, window size in seconds to extract from the middle of the movement
    - duration. If None, the entire rep duration is used.

    Returns:
    - iso_dict: dict containing each movement and rest as keys. Each contains:
        - 'movement': dict with movement data, containing keys for:
            - 'reps': dict with data from each movement rep number, each rep number containing keys for:
                - 'emg': dataframe with emg data for rep
                    dim = n_channels x n_samples
                - 'duration: dataframe with duration for rep
            - 'combined': ndarray containing combined data for entire movement, all reps combined
        - 'rest': dict wth data from each rest rep number, same structure as 'movement'
        - 'm_r_combined': ndarray containing combined data from movement and rest, one rep after another
        - 'all_emg': all data in order of executation with each movement, rest, prep, and transition files

    """
    move_files = sorted(glob.glob(os.path.join(s_dir, '*move*')))
    rest_files = sorted(glob.glob(os.path.join(s_dir, '*rest*')))
    prep_files = sorted(glob.glob(os.path.join(s_dir, '*prep*')))
    transition_files = sorted(glob.glob(os.path.join(s_dir, '*transition*')))
    tutorial_files = sorted(glob.glob(os.path.join(s_dir, '*tutorial*')))

    half_win_samples = int((win / 2) * sampling_freq) if win is not None else None
    n_rest_reps = n_reps - 1  # every movement has 1 fewer rest rep than move/prep reps

    metadata = []
    interleaved_emg = [] # interleave rep-by-rep: prep rep1, move rep1, rest rep1, prep rep2, ...

    current_sample = 0  # Track the current sample index for metadata


    for move_num, move_type in enumerate(move_names):
        move_start_idx = move_num * n_reps
        prep_start_idx = move_num * n_reps
        rest_start_idx = move_num * n_rest_reps


        for rep_num in range(n_reps):
            # Gather prep data
            prep_data = pd.read_pickle(prep_files[prep_start_idx + rep_num])
            prep_emg = np.asarray(prep_data['data'])
            prep_len = prep_emg.shape[1]
            prep_dur = prep_data['duration_s']

            interleaved_emg.append(prep_emg)
            metadata.append({
                'pose': move_type,
                'phase': 'prep',
                'rep': rep_num,
                'duration': prep_dur,
                'start_idx': current_sample,
                'end_idx': current_sample + prep_len
            })

            current_sample += prep_len  # Update current sample index after prep

            # Gather movement data
            move_data = pd.read_pickle(move_files[move_start_idx + rep_num])
            move_emg = np.asarray(move_data['data'])
            move_len = move_emg.shape[1]
            move_dur = move_data['duration_s']
            
            interleaved_emg.append(move_emg)
            metadata.append({
                'pose': move_type,
                'phase': 'movement',
                'rep': rep_num,
                'duration': move_dur,
                'start_idx': current_sample,
                'end_idx': current_sample + move_len
            })
            current_sample += move_len  # Update current sample index after movement

            # Gather rest data
            if rep_num < n_rest_reps:  # Only gather rest data if there are still rest reps left
                rest_data = pd.read_pickle(rest_files[rest_start_idx + rep_num])
                rest_emg = np.asarray(rest_data['data'])
                rest_len = rest_emg.shape[1]
                rest_dur = rest_data['duration_s']
                
                interleaved_emg.append(rest_emg)
                metadata.append({
                    'pose': move_type,
                    'phase': 'rest',
                    'rep': rep_num,
                    'duration': rest_dur,
                    'start_idx': current_sample,
                    'end_idx': current_sample + rest_len
                })
                current_sample += rest_len  # Update current sample index after rest

        full_emg = np.hstack(interleaved_emg)
        metadata_df = pd.DataFrame(metadata)


    return {
        'emg': full_emg,
        'metadata': metadata_df,
        'fs': sampling_freq
    }


def plot_emg_with_movement(emg_arr, poses, durations, fs, channel=(0, 0), ax=None, save_path = None):

    """
    Plot a single EMG channel, with a coloured patch behind each pose.

    Parameters:
    - emg_array: np.ndarray, n_channels x n_samples (flat).
    - poses: sequence of str, the name of the pose performed in each block, in order.
    - durations: sequence of float, the duration of each pose, same order as poses.
      If fs is given, these are interpreted as seconds. If fs is None, these are
      interpreted as raw counts (e.g. reps or samples) directly on the signal's own index.
    - fs: float, the sampling frequency of the EMG in Hz.
    - channel: (row, col) for a grid array, or an int for a flat one.
    - ax: optional matplotlib axis, so this can be reused inside bigger figures.

    Returns:
    - ax: the axis the signal was drawn on.
    """


    ch = channel if isinstance(channel, int) else channel[0]
    signal = emg_arr[ch, :]
    ch_label = f"channel {ch}"

    # With fs given, build a real time axis in seconds; without it, fall back to a
    # plain integer index, and treat durations as counts on that same index rather
    # than as seconds to be scaled
    if fs is not None:
        t = np.arange(signal.size) / fs
        x_label = 'Time (s)'
    else:
        t = np.arange(signal.size)
        x_label = 'Rep'


    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))

    # One colour per unique pose name: if the same pose is repeated later in the recording,
    # it keeps the same colour (dict.fromkeys preserves the order of first appearance)
    unique_poses = list(dict.fromkeys(poses))
    cmap = plt.get_cmap('tab10' if len(unique_poses) <= 10 else 'tab20')
    colors = {pose: cmap(i % cmap.N) for i, pose in enumerate(unique_poses)}

    # The poses are performed one after the other, so a pose starts where the previous ones ended:
    # the cumulative sum of the durations gives us the boundaries of every block, in seconds
    edges = np.concatenate(([0.0], np.cumsum(durations)))


    for pose, start, end in zip(poses, edges[:-1], edges[1:]):
        # Clipped to the end of the signal, in case the durations add up to more than what was recorded
        ax.axvspan(min(start, t[-1]), min(end, t[-1]), color=colors[pose], alpha=0.25, lw=0)

    ax.plot(t, signal, color='0.15', lw=0.4)

    ax.set_xlim(0, t[-1])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(f'EMG - {ch_label}')
    ax.set_title('EMG signal with intended poses')

    # A patch legend, one entry per pose, drawn with the same alpha as the patches
    handles = [Patch(facecolor=colors[pose], alpha=0.25, label=pose) for pose in unique_poses]
    ax.legend(handles=handles, fontsize=8, loc='upper right', ncol=min(len(handles), 4))

    if save_path is not None:
        plt.savefig(save_path,dpi=300,bbox_inches="tight")

    plt.show()
        
    return fig, ax

def concat_movement(emg_arr, metadata, fs, pose, phase):
    """
    Extracts and concatenates EMG signal segments corresponding to a specific 
    movement pose and phase across all repetitions.

    Parameters:
    - emg: 2D array of shape (n_channels, total_n_samples)
    - metadata: DataFrame containing segment definitions 
                ('pose', 'phase', 'start_idx', 'end_idx')
    - pose: string, exact name of the movement pose (e.g., 'fist')
    - phase: string, phase type to isolate ('movement', 'rest', or 'prep')

    Returns:
    - concatenated_emg: 2D array of shape (n_channels, concatenated_samples)
                        containing only the target pose data.
    - matching_rows: DataFrame containing the metadata rows corresponding 
                     to the extracted segments.
    """
    # Filter metadata for the specific pose and phase
    matching_rows = metadata[
        (metadata['pose'] == pose) & 
        (metadata['phase'] == phase)
    ]

    if matching_rows.empty:
        raise ValueError(
            f"No entries found for movement='{pose}' with phase='{phase}'."
        )

    # Slice each segment from the full EMG array according to their samples from metadata
    segments = [
        emg_arr[:, row['start_idx']:row['end_idx']] 
        for _, row in matching_rows.iterrows()
    ]

    # Horizontally concatenate along time axis (axis=1)
    concatenated_emg = np.hstack(segments)

    return concatenated_emg, matching_rows


# --------------------------------------------------------------------------
#   Digital Filters
# --------------------------------------------------------------------------

# def bandpass_signals(emg_dict, fsamp, high_pass=20, low_pass=500, order=2):
#     """
#     Bandpass filter emg data using a butterworth filter

#     Args:
#         emg_data (ndarray): emg data (n_channels x n_samples)
#         fsamp (float): Sampling frequency
#         low_pass (float): Cut-off frequency for the low-pass filter
#         high_pass (float): Cut-off frequency for the high-pass filter
#         order (int): Order of the filter

#     Returns:
#         ndarray : filtered emg data (n_channels x n_samples)
#     """
#     emg_data = emg_dict['emg_combined']
#     b, a = butter(order, [high_pass, low_pass], fs=fsamp, btype="band")
#     emg_dict['emg_combined'] = filtfilt(b, a, emg_data, axis=1)

#     return emg_dict

# def notch_signals(emg_dict, fsamp, nfreq=50, dfreq=1, order=2, n_harmonics=3):
#     emg_data = emg_dict['emg_combined']
#     harmonics = nfreq * np.arange(1, n_harmonics + 1)

#     for i in np.arange(n_harmonics):
#         b, a = butter(
#             order,
#             [harmonics[i] - dfreq, harmonics[i] + dfreq],
#             fs=fsamp,
#             btype="bandstop",
#         )
#         emg_data = filtfilt(b, a, emg_data, axis=1) 

#     emg_dict['emg_combined'] = emg_data
#     return emg_dict

from scipy.signal import butter, filtfilt

def band_notch_sig(sig, fsamp, high_pass=20, low_pass=500, order_band=2, 
                   order_notch = 2, nfreq=50, dfreq = 1, n_harmonics = 3):
    """
    Bandpass filter emg data using a butterworth filter

    Args:
        emg_data (ndarray): emg data (n_channels x n_samples)
        fsamp (float): Sampling frequency
        low_pass (float): Cut-off frequency for the low-pass filter
        high_pass (float): Cut-off frequency for the high-pass filter
        order (int): Order of the filter

    Returns:
        ndarray : filtered emg data (n_channels x n_samples)
    """
    b, a = butter(order_band, [high_pass, low_pass], fs=fsamp, btype="band")
    sig_band = filtfilt(b, a, sig, axis=1)

    harmonics = nfreq * np.arange(1, n_harmonics + 1)
    
    for hfreq in harmonics:
        bn, an = butter(
            order_notch,
            [hfreq - dfreq, hfreq + dfreq],
            fs=fsamp,
            btype="bandstop",
        )
        sig_filt = filtfilt(bn, an, sig_band, axis=1) 

    return sig_filt

from scipy.interpolate import interp1d

def remove_spikes(emg_data, threshold_std=5):
    """
    Remove large spikes via simple thresholding and linear interpolation.
    
    Parameters
    ----------
    emg : ndarray, shape (n_channels, n_samples)
    threshold_std : float
        Number of standard deviations for threshold
    
    Returns
    -------
    emg_clean : ndarray
    spike_mask : ndarray, bool
    """
    emg_clean = emg_data.copy()
    spike_mask = np.zeros_like(emg_data, dtype=bool)
    
    for ch in range(emg_data.shape[0]):
        signal = emg_data[ch]
        threshold = threshold_std * np.std(signal)
        
        spikes = np.abs(signal) > threshold
        spike_mask[ch] = spikes
        
        if np.any(spikes):
            clean_idx = np.where(~spikes)[0]
            spike_idx = np.where(spikes)[0]
            
            if len(clean_idx) > 1:
                interp = interp1d(clean_idx, signal[clean_idx], 
                                  kind='linear', bounds_error=False, 
                                  fill_value='extrapolate')
                emg_clean[ch, spike_idx] = interp(spike_idx)

    return emg_clean, spike_mask

# --------------------------------------------------------------------------
#   Signal Quality
# --------------------------------------------------------------------------
from scipy.signal import welch
from scipy.stats import shapiro

def calc_sig_qual(emg_sig, emg_noise, fsamp):
    """
    Computes Signal-to-Noise Ratio (SNR; in dB) and Normality (Shapiro-Wilk correlation metric; in dB).
    """
    power_sig = np.mean(emg_sig ** 2, axis=1)
    power_noise = np.mean(emg_noise ** 2, axis=1)
    snr_db = 10 * np.log10(power_sig / power_noise)
    rms_noise = power_noise ** 0.5

    # Normality metric: Average Shapiro-Wilk statistic across sub-windows
    sub_sample = emg_sig[:, :min(500, emg_sig.shape[1])]
    normality_stats = [shapiro(sub_sample[ch, :]).statistic for ch in range(sub_sample.shape[0])]

    return {
        'snr_per_channel': snr_db,
        'mean_snr': float(np.mean(snr_db)),
        'normality_stats': normality_stats
    }



def calc_PSD(sig, fsamp=2048, nperseg=2048, noverlap=2048/2):
    '''
    Compute the power spectral density for each channel using Welch's method.

    Args:
        sig (ndarray): Multi-channel signal (Channels x Samples)
        fsamp (float): Sampling rate in Hz
        nperseg (int): Number of data points per segment
        overlap (int): Number of overlapping samples
    '''

    f, _ = welch(sig[0,:], fs=fsamp, nperseg=nperseg, noverlap=noverlap)

    P = np.zeros((sig.shape[0], f.shape[0]))

    for i in range(sig.shape[0]):
        _ , P[i,:] = welch(sig[i,:], fs=fsamp, nperseg=nperseg, noverlap=noverlap)

    return P, f



import itertools
from scipy.signal import find_peaks

def analyze_movement_snr(data_dict, samp_freq, nsamples,
                          channels_to_plot=np.arange(13,25), zoom_window=(24.9,25.1),
                          save_path = None,
                          title_str = None):
    """
    Compute signal/noise separation, SNR, and PSD for one movement type,
    using the 'combined' entry built by get_task_data.

    Movement reps are treated as 'signal'; prep and rest reps are treated
    as 'noise' and merged into a single noise segment wherever they are
    back-to-back (e.g. rest_i immediately followed by prep_{i+1}).

    Parameters:
    - datadict: dict, output of get_task_data.
    - samp_freq: float, sampling frequency in Hz, used to convert
      durations (seconds) into sample counts.
    - calc_psd_func: callable, PSD function with signature
      (data_2d) -> (Pxx, freqs), matching calc_PSD used elsewhere.
    - channels_to_plot: list of int, row indices of emg to show in the
      trace plots. Defaults to the middle 12 channels.
    - zoom_window: tuple (t_start, t_end) in seconds for the zoomed
      trace panel. Defaults to a 0.2 s window .

    Returns:
    - results: dict with SNR, noise power/RMS, band power fractions,
      and the raw concatenated 'sig'/'noise' arrays.
    """
    emg_df = data_dict['emg_combined']
    poses = data_dict['poses'] 
    durations = data_dict['duration']

    emg_data = emg_df.copy()


    block_lengths = [int(round(d * samp_freq)) for d in durations] # get sample count from the duration of each movement
    boundaries = np.concatenate(([0], np.cumsum(block_lengths))) # add sample counts cumulatively to get transitions throughout task

    # label rest and prep segments as noise
    labels = ['noise' if p in ('prep', 'rest') else 'signal' for p in poses]

    merged_segments = []  # merge segments with same label, list of (label, start_idx, end_idx)
    block_idx = 0
    for label, group in itertools.groupby(labels):
        group_len = len(list(group)) # group order will look like noise, signal, noise, signal...
        start_idx = boundaries[block_idx]
        end_idx = boundaries[block_idx + group_len]
        merged_segments.append((label, start_idx, end_idx))
        block_idx += group_len

    sig = np.concatenate([emg_data[:, s:e] for lbl, s, e in merged_segments if lbl == 'signal'], axis=1)
    noise = np.concatenate([emg_data[:, s:e] for lbl, s, e in merged_segments if lbl == 'noise'], axis=1)

    # Calculate the power spectral density (PSD)
    Pn, fn = calc_PSD(noise)
    Ps, fsig = calc_PSD(sig)

    # Calculate the SNR
    power_sig = np.mean(sig ** 2)
    power_noise = np.mean(noise ** 2)
    snr = 10 * np.log10(power_sig / power_noise)
    rms_noise = power_noise ** 0.5

    P_mean = np.mean(Pn, axis=0)
    P_tot = np.sum(P_mean)

    # Plot a few channels from the recording together with the power spectrum
    channels = channels_to_plot
    t = np.linspace(0, (nsamples-1)/samp_freq, nsamples)

    p_med = np.median(Ps, axis = 0)

    troughs_filt, props = find_peaks(-np.log10(p_med), prominence=0.5)
    trough_freqs = fsig[troughs_filt]

    diffs_t = np.diff(np.sort(trough_freqs))

    fig, ax = plt.subplots(2, 2, figsize=(18,6))
    plt.subplots_adjust(hspace=0.5)
    ax = ax.flatten() 

    # Full signal
    for i, ch_idx in enumerate(channels):
        trace = emg_df[ch_idx, :]
        ax[0].plot(t,trace + 1 * i, lw=0.5)
    # ax[0].plot(t,emg_df/3, lw=2, color="gray")
    # ax[0].plot(t,emg_df/3, lw=1, color="black")    
    ax[0].set_xlabel("Time (s)")
    ax[0].set_ylabel("Amplitude + offset (mV)")
    #ax[0].set_ylim(-1.5, len(channels) + 1)
    ax[0].set_title(title_str + "EMG signals")
    # Now let's focus on some details
    for i, ch_idx in enumerate(channels):
        trace = emg_df[ch_idx, :]
        ax[1].plot(t,trace + 1 * i, lw=0.5)  
    ax[1].set_xlabel("Time (s)")
    ax[1].set_ylabel("Amplitude + offset (mV)")
    #ax[1].set_ylim(-1.5, len(channels) + 1)
    ax[1].set_xlim(zoom_window)
    ax[1].set_xticks([24.9, 25, 25.1])
    ax[1].set_title(title_str + "EMG signals (zoom)")
    # Plot the PSD
    ax[2].semilogy(fn,Pn.T, lw=0.1, color=[0.7, 0.7, 1])
    ax[2].semilogy(fsig,Ps.T, lw=0.1, color=[1, 0.7, 0.7])
    ax[2].semilogy(fn,np.percentile(Pn, 50, axis=0), lw=1, color="blue", label="noise")
    ax[2].semilogy(fsig,np.percentile(Ps, 50, axis=0), lw=1, color="red", label="signal")
    ax[2].set_xlabel("Frequency (Hz)")
    ax[2].set_ylabel("PSD (mV$^2$/Hz)")
    #ax[2].set_ylim(1e-9, 1e-2)
    ax[2].legend()
    ax[2].set_title(title_str + "PSD")

    # Plot intereferences on PSD
    ax[3].semilogy(fsig, p_med)
    ax[3].semilogy(trough_freqs, p_med[troughs_filt], "rx", markersize=8)
    ax[3].set_xlabel("Frequency (Hz)")
    ax[3].set_ylabel("PSD (mV²/Hz)")
    ax[3].set_title("Detected interference peaks")

    if save_path is not None:
            plt.savefig(save_path,dpi=300,bbox_inches="tight")
    plt.show()

    return {
        'snr_db': snr, 'Pn': Pn, 'Ps': Ps, 
        'fn': fn, 'fsig': fsig, 
        'rms_noise': rms_noise,
        'diffs_t': diffs_t,
        'trough_freqs': trough_freqs,
       # 'noise_floor_median': noise_floor_median,
        #'frac_low_pct': frac_low, 'frac_pli_pct': frac_pli,
       # 'frac_med_pct': frac_med, 'frac_high_pct': frac_high,
        'sig': sig, 'noise': noise,
    }



def get_snr(sig_df, noise_df, samp_freq, calc_psd_func=None,
            title_str="", save_path=None, mark_interference=True,
            show_individual_channels=True):
    """
    Compute SNR and PSD for signal (movement) and noise EMG recordings, and
    plot the two PSDs overlaid on a single axis.

    Parameters:
    - sig_df, noise_df: DataFrames (or arrays) of EMG data,
        dim n_channels x n_samples
    - samp_freq: float, sampling frequency in Hz needed for calcPSD
    - calc_psd_func: callable, (data_2d) -> (Pxx, freqs). Defaults to the
        global `calc_PSD`.
    - title_str: str, prefix for the plot title
    - save_path: str or None, if given the figure is saved there
    - mark_interference: bool, mark troughs detected in the median signal PSD
    - show_individual_channels: bool, draw faint per-channel PSD lines behind
        the median lines

    Returns:
    - results: dict with SNR, PSDs, frequency vectors, noise RMS,
      interference trough frequencies, and the raw sig/noise arrays.
    """
    if calc_psd_func is None:
        calc_psd_func = calc_PSD

    # DataFrame to array with dim (n_channels x n_samples)
    sig = np.asarray(sig_df, dtype=float)
    noise = np.asarray(noise_df, dtype=float)

    # PSD of each recording
    Ps, fsig = calc_psd_func(sig, fsamp=samp_freq, nperseg=samp_freq, noverlap=samp_freq/2)
    Pn, fn = calc_psd_func(noise, fsamp=samp_freq, nperseg=samp_freq, noverlap=samp_freq/2)

    # SNR
    power_sig = np.mean(sig ** 2)
    power_noise = np.mean(noise ** 2)
    snr = 10 * np.log10(power_sig / power_noise)
    rms_noise = power_noise ** 0.5

    # Median PSD across channels
    p_med_sig = np.median(Ps, axis=0)
    p_med_noise = np.median(Pn, axis=0)

    # Interference troughs in the median signal PSD
    troughs, _ = find_peaks(-np.log10(p_med_sig), prominence=0.5)
    trough_freqs = fsig[troughs]
    diffs_t = np.diff(np.sort(trough_freqs))

    # Overlaid PSD plot
    fig, ax = plt.subplots(figsize=(10, 5))

    if show_individual_channels:
        ax.semilogy(fn, Pn.T, lw=0.1, color=[0.7, 0.7, 1])
        ax.semilogy(fsig, Ps.T, lw=0.1, color=[1, 0.7, 0.7])

    ax.semilogy(fn, p_med_noise, lw=1.5, color="blue", label="noise (median)")
    ax.semilogy(fsig, p_med_sig, lw=1.5, color="red", label="signal (median)")

    if mark_interference:
        ax.semilogy(trough_freqs, p_med_sig[troughs], "kx", markersize=8,
                    label="detected interference")

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (mV$^2$/Hz)")
    ax.set_title(f"{title_str} PSD: signal vs noise (SNR = {snr:.1f} dB)".strip())
    ax.legend()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    return {
        'snr_db': snr, 'Pn': Pn, 'Ps': Ps,
        'fn': fn, 'fsig': fsig,
        'rms_noise': rms_noise,
        'diffs_t': diffs_t,
        'trough_freqs': trough_freqs,
        'sig': sig, 'noise': noise
    }

# --------------------------------------------------------------------------
#   Post Processing
# --------------------------------------------------------------------------

def sta_muap(emg, spike_indices, wave_dur=25, fsamp=1000):
    """
    Spike-triggered averaging (STA) to extract Motor Unit Action Potential (MUAP) waveform.
    
    Parameters:
    - emg: 2D array (n_channels, n_samples)
    - spike_indices: 1D array of sample indices where MU fired
    - wave_dur: int duration of each MUAP, in ms
    - fsamp: sampling frequency, Hz

    Returns:
    - muap: 1D array of average signal across all spike times
    """
    n_ch, n_samples = emg.shape

    # Calculate total duration in samples and get the half-window length of waveform
    wave_samples = int(np.round((wave_dur / 1000.0) * fsamp))

    # Place upper bound at s n_samples - wave_samples to ensure that our waveform samples doesn't exceed our signal length
    # lower bounda at 0 (start of signal)
    valid_spikes = [
        s for s in spike_indices 
        if 0 <= s < n_samples - wave_samples
    ]

    # return empty array if no spikes
    if not valid_spikes:
        return np.zeros((n_ch, wave_samples))

    # Initiate waveforms
    cutouts = np.zeros((len(valid_spikes), n_ch, wave_samples))

    # select emg signal for the waveform length at each spike 
    for i, s in enumerate(valid_spikes):
        cutouts[i] = emg[:, s : s + wave_samples]

    # MUAP is the average signal across all spikes
    muap = np.mean(cutouts, axis=0)

    return muap