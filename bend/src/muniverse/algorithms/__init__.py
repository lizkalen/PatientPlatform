"""
Benchmark algorithms for decomposition.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from ..utils.containers import pull_container, verify_container_engine
from .decomposition import decompose_cbss, decompose_cbss_v2, decompose_scd


def init():
    """
    Initialize the algorithms module.
    This includes verifying container engines and pulling container images if needed.
    If both Docker and Singularity are available, Singularity will be used by default.

    Returns:
        str: The selected container engine ("docker" or "singularity")
    """
    # Check availability of both engines
    docker_available = verify_container_engine("docker")
    singularity_available = verify_container_engine("singularity")

    # Select engine based on availability
    if singularity_available:
        engine = "singularity"
    elif docker_available:
        engine = "docker"
    else:
        raise RuntimeError("No container engine (Docker or Singularity) is available. Please install one first.")

    # Get container name (using default)
    container_name = "pranavm19/muniverse:scd"

    # Pull container if needed
    pull_container(container_name, engine)
    print(f"[INFO] Algorithms module initialized using {engine}.")

    return engine


def decompose_recording(
    data: Union[str, np.ndarray],
    method: str = "cbss",
    algorithm_config: Optional[Dict] = None,
    engine: Optional[str] = None,
    container: Optional[str] = None,
) -> Tuple[Dict, Dict]:
    """
    Decompose EMG recordings using specified method.

    Args:
        data: Either a path to input data file (.npy or .edf) or a numpy array of EMG data
        method: Decomposition method to use ("scd", "cbss", or "cbss_v2")
        algorithm_config: Optional dictionary containing algorithm configuration
        engine: Container engine to use ("docker" or "singularity")
        container: Path to container image (required for SCD method)

    Returns:
        Tuple containing:
        - Dictionary with decomposition results containing:
          * sources: Estimated sources
          * spikes: Spike timing dictionary
          * silhouette: Quality metrics (if available)
        - Dictionary with processing metadata
        
    Note:
        For UpperBound decomposition, use decompose_upperbound() directly.
    """
    # Route to appropriate method
    if method == "scd":
        if container is None or engine is None:
            raise ValueError("Container path and engine must be provided for SCD method")
        
        # For SCD, handle file path input by converting to numpy array first
        if isinstance(data, str):
            data_path = Path(data)
            if not data_path.exists():
                raise FileNotFoundError(f"Input data file not found: {data_path}")
            if data_path.suffix not in [".npy", ".edf"]:
                raise ValueError(
                    f"Unsupported file format: {data_path.suffix}. Must be .npy or .edf"
                )

            # Load data into numpy array
            if data_path.suffix == ".edf":
                from edfio import read_edf
                raw = read_edf(data_path)
                n_channels = raw.num_signals
                data = np.stack([raw.signals[i].data for i in range(n_channels)])
            else:  # .npy
                data = np.load(data_path)
        
        # Validate numpy array
        if not isinstance(data, np.ndarray):
            raise TypeError("data must be either a file path (str) or numpy array")
        if data.ndim != 2:
            raise ValueError("EMG data must be a 2D array (channels x samples)")
        
        return decompose_scd(
            data=data,
            algorithm_config=algorithm_config,
            engine=engine,
            container=container,
        )
    elif method in ("cbss", "cbss_v2"):
        # For internal methods, convert to numpy array if needed
        if isinstance(data, str):
            data_path = Path(data)
            if not data_path.exists():
                raise FileNotFoundError(f"Input data file not found: {data_path}")
            if data_path.suffix not in [".npy", ".edf"]:
                raise ValueError(
                    f"Unsupported file format: {data_path.suffix}. Must be .npy or .edf"
                )

            # Load data into numpy array
            if data_path.suffix == ".edf":
                from edfio import read_edf

                raw = read_edf(data_path)
                n_channels = raw.num_signals
                data = np.stack([raw.signals[i].data for i in range(n_channels)])
            else:  # .npy
                data = np.load(data_path)

        # Validate numpy array
        if not isinstance(data, np.ndarray):
            raise TypeError("data must be either a file path (str) or numpy array")
        if data.ndim != 2:
            raise ValueError("EMG data must be a 2D array (channels x samples)")

        # Call CBSS method
        if method == "cbss_v2":
            return decompose_cbss_v2(
                data=data,
                algorithm_config=algorithm_config
            )
        return decompose_cbss(
            data=data,
            algorithm_config=algorithm_config
        )
    else:
        raise ValueError(
            f"Unknown method: {method}. Must be one of: scd, cbss, cbss_v2"
        )
