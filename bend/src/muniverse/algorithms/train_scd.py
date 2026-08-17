import json
import logging
import warnings
from pathlib import Path
from importlib import resources

import numpy as np
import scipy.io as sio
import torch

from .config.structures import Config, set_random_seed
from .models.scd import SwarmContrastiveDecomposition
from .processing.preprocess import compute_extension_factor_bounds

set_random_seed(seed=42)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


_MUAP_DURATION_MS = 15.0   # fixed assumption embedded in constraint derivation
_N_SOURCES        = 30     # fixed assumption embedded in constraint derivation


def _validate_configuration(neural_data: torch.Tensor, config: Config) -> None:
    """
    Validate extension_factor and peak-separation parameters before training.

    Extension-factor bounds (at fs = sampling_frequency):
      K_min  — minimum K for model identifiability:
               derived from  K*M >= N*(K+L-1)  →  K >= ceil(N*(L-1)/(M-N))
               where M = clean channels, L = MUAP samples, N = _N_SOURCES.
      K_max  — maximum K from temporal separation:
               derived from  L + K - 1 < T  →  K <= T - L
               where T = floor(fs / config.max_firing_rate_hz).

    A ValueError is raised when K > K_max (temporal aliasing guaranteed).
    A UserWarning is issued when K < K_min (model is under-determined in
    theory; the algorithm may still succeed for sparse EMG).

    Peak-separation check:
    A UserWarning is issued when reset_peak_separation_ms >= min ISI at
    max_firing_rate_hz, which would prevent detecting consecutive spikes
    from the fastest expected units.
    """
    if config.sampling_frequency is None:
        return

    fs   = config.sampling_frequency
    K    = config.extension_factor
    fmax = config.max_firing_rate_hz
    M    = neural_data.shape[1] - (len(config.bad_channels) if config.bad_channels else 0)
    L    = int(_MUAP_DURATION_MS * fs / 1000)
    T    = int(fs / fmax)

    k_min, k_max = compute_extension_factor_bounds(
        num_channels=neural_data.shape[1],
        bad_channels=config.bad_channels,
        sampling_frequency=fs,
        muap_duration_ms=_MUAP_DURATION_MS,
        n_sources=_N_SOURCES,
        max_firing_rate_hz=fmax,
    )

    if k_max <= 0:
        raise ValueError(
            f"Temporal-separation constraint yields k_max={k_max} <= 0. "
            f"MUAP duration assumption ({_MUAP_DURATION_MS} ms = {L} samples) "
            f"is too long relative to the minimum ISI at {fmax} Hz "
            f"({T} samples). Reduce max_firing_rate_hz or check sampling_frequency."
        )

    if K > k_max:
        raise ValueError(
            f"extension_factor={K} violates the temporal-separation constraint "
            f"(L + K - 1 < T). Maximum allowed K is {k_max} "
            f"(L={L} samples, T={T} samples at {fmax} Hz). "
            f"Reduce extension_factor to at most {k_max}."
        )

    if M > _N_SOURCES and K < k_min:
        warnings.warn(
            f"extension_factor={K} is below the theoretical minimum k_min={k_min} "
            f"for model identifiability (M={M} clean channels, N={_N_SOURCES} sources, "
            f"L={L} samples). The algorithm may still converge for sparse EMG.",
            UserWarning,
            stacklevel=3,
        )

    # Peak-separation check: warn if reset_peak_separation_ms >= min ISI
    min_isi_ms = 1000.0 / fmax
    if config.reset_peak_separation_ms >= min_isi_ms:
        warnings.warn(
            f"reset_peak_separation_ms={config.reset_peak_separation_ms} ms "
            f"(= {config.reset_peak_separation} samples at {fs} Hz) >= "
            f"minimum ISI at {fmax} Hz ({min_isi_ms:.1f} ms = {T} samples). "
            f"Consecutive spikes from the fastest units may be missed.",
            UserWarning,
            stacklevel=3,
        )


def load_config(config_name: str = "default", config_file: Path = None) -> Config:
    """
    Load configuration from JSON file.
    
    Parameters
    ----------
    config_name : str
        Name of the configuration to load (e.g., "default")
    config_file : Path, optional
        Path to custom config file. If None, uses built-in configs.json
    
    Returns
    -------
    Config
        Configuration object
    """
    if config_file is None:
        # Load from package's built-in configs.json
        with resources.files("scd").joinpath("configs.json").open("r") as f:
            config_data = json.load(f)
    else:
        with open(config_file, "r") as f:
            config_data = json.load(f)

    selected_config = config_data.get(config_name, config_data["default"])
    logger.info(f"Loaded config: {config_name}")
    return Config(**selected_config)

def load_data(path: Path, key: str = "emg", device: str = "cuda") -> torch.Tensor:
    """
    Load neural data from .mat or .npy file.
    
    Parameters
    ----------
    path : Path
        Path to data file
    key : str
        Key/variable name in .mat file (default: "emg")
    device : str
        Device to load tensor to ("cuda" or "cpu")
    
    Returns
    -------
    torch.Tensor
        Neural data with shape (time, channels)
    """
    path = Path(path)
    
    # Load data based on file format
    if path.suffix == ".mat":
        mat = sio.loadmat(path)
        data = mat[key]
    elif path.suffix == ".npy":
        data = np.load(path)
    else:
        raise ValueError(f"Unsupported format: {path.suffix}. Use .mat or .npy")
    
    # Convert to tensor
    neural_data = torch.from_numpy(data).to(device=device, dtype=torch.float32)
    
    # Ensure shape is (time, channels) - time should be the longer dimension
    if neural_data.shape[1] > neural_data.shape[0]:
        neural_data = neural_data.T
        logger.info(f"Transposed data to (time, channels)")
    
    logger.info(f"Loaded data from {path.name}, shape: {neural_data.shape}")
    return neural_data

def preprocess_data(neural_data: torch.Tensor, config: Config) -> torch.Tensor:
    """
    Preprocess neural data (slicing, bad channel removal).
    
    Parameters
    ----------
    neural_data : torch.Tensor
        Raw neural data (time, channels)
    config : Config
        Configuration object
    
    Returns
    -------
    torch.Tensor
        Preprocessed neural data
    """
    start_idx = int(config.start_time * config.sampling_frequency)
    
    if config.end_time == -1 or config.end_time <= 0:
        end_idx = None
    else:
        end_idx = int(config.end_time * config.sampling_frequency)
    
    neural_data = neural_data[start_idx:end_idx, :]
    
    # Zero out bad channels if specified
    if hasattr(config, 'bad_channels') and config.bad_channels:
        neural_data[:, config.bad_channels] = 0
        logger.info(f"Zeroed bad channels: {config.bad_channels}")
    
    logger.info(f"Preprocessed data shape: {neural_data.shape}")
    return neural_data

def train_model(neural_data: torch.Tensor, config: Config) -> tuple:
    """
    Run the SwarmContrastiveDecomposition model.
    
    Parameters
    ----------
    neural_data : torch.Tensor
        Preprocessed neural data (time, channels)
    config : Config
        Configuration object
    
    Returns
    -------
    tuple
        (dictionary, predicted_timestamps)
    """
    model = SwarmContrastiveDecomposition()
    predicted_timestamps, dictionary = model.run(neural_data, config)
    return dictionary, predicted_timestamps

def train(
    path: Path,
    config_name: str = "default",
    config_file: Path = None,
    key: str = "emg",
    **config_overrides
) -> tuple:
    """
    Full training pipeline: load data, preprocess, and train.
    
    Parameters
    ----------
    path : Path
        Path to data file (.mat or .npy)
    config_name : str
        Name of configuration to load from configs.json
    config_file : Path, optional
        Path to custom config file
    key : str
        Key/variable name in .mat file
    **config_overrides
        Override specific config values (e.g., max_iterations=100)
    
    Returns
    -------
    tuple
        (dictionary, predicted_timestamps)
    
    Example
    -------
    >>> dictionary, timestamps = train("data/emg.mat", config_name="default")
    >>> dictionary, timestamps = train("data/emg.mat", max_iterations=100)
    """
    # Load config
    config = load_config(config_name, config_file)
    
    # Apply any overrides
    for key_name, value in config_overrides.items():
        if hasattr(config, key_name):
            setattr(config, key_name, value)
            logger.info(f"Config override: {key_name} = {value}")
    
    # Load and preprocess data
    neural_data = load_data(path, key=key, device=config.device)
    neural_data = preprocess_data(neural_data, config)

    # Validate extension_factor and peak-separation parameters
    _validate_configuration(neural_data, config)

    # Train
    dictionary, timestamps = train_model(neural_data, config)
    
    return dictionary, timestamps