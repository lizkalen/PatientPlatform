"""
High-level wrapper functions for EMG decomposition with logging.

These functions never raise exceptions. They always return (results, log_data),
even on failure. Failed decompositions return {"sources": None, ...} with logs containing
error details. Callers could check if results["sources"] is None to detect failures.
"""

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ..utils.logging import AlgorithmLogger
from .cbss import CBSS
from .cbss_v2 import CBSSv2
from .upperbound import UpperBound
from .ae_decomposer import AEDecoder, AEDecoderConfig


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from a JSON file."""
    with open(config_path, "r") as f:
        return json.load(f)


def decompose_scd(
    data: np.ndarray,
    algorithm_config: Optional[Dict] = None,
    engine: str = "singularity",
    container: str = "environment/muniverse_scd.sif",
    metadata: Optional[Dict] = None,
) -> Tuple[Dict, Dict]:
    """
    Run SCD decomposition using container.

    Args:
        data: numpy array of EMG data (channels x samples)
        algorithm_config: Optional dictionary containing algorithm configuration
        engine: Container engine to use ("docker" or "singularity")
        container: Path to container image
        metadata: Optional dictionary containing input data metadata for logging

    Returns:
        Tuple containing:
        - Dictionary with decomposition results containing:
          * sources: Estimated sources
          * spikes: Spike timing dictionary
          * silhouette: Quality metrics (if available)
        - Dictionary with processing metadata
    """
    # Initialize logger
    logger = AlgorithmLogger()
    logger.add_generated_by(
        name="Swarm Contrastive Decomposition",
        url="https://github.com/AgneGris/swarm-contrastive-decomposition.git",
        commit="632a9ad041cf957584926d6b5cc64b7fe741e9eb",
        license="Creative Commons Attribution-NonCommercial 4.0 International Public License",
    )

    # Set input data information
    if metadata:
        logger.set_input_data(file_name=metadata["filename"], file_format=metadata["format"])
    else:
        logger.set_input_data(file_name="numpy_array", file_format="npy")

    # Load and set algorithm configuration
    if algorithm_config:
        algo_cfg = algorithm_config
        logger.set_algorithm_config(algo_cfg)
    else:
        # Load default configuration
        config_dir = Path(__file__).parent.parent.parent / "configs"
        algorithm_config = config_dir / "scd.json"
        if not algorithm_config.exists():
            raise FileNotFoundError(
                f"Default SCD config not found at {algorithm_config}"
            )
        algo_cfg = load_config(algorithm_config)
        logger.set_algorithm_config(algo_cfg)

    # Create single run directory following neuromotion pattern
    with tempfile.TemporaryDirectory() as run_dir:
        run_dir = Path(run_dir)

        try:
            # Save data as standardized input file
            input_data_path = run_dir / "input_data.npy"
            np.save(input_data_path, data)
            
            # Save config as standardized config file (using already loaded config)
            config_path = run_dir / "config.json"
            with open(config_path, 'w') as f:
                json.dump(algo_cfg, f, indent=2)

            # Get the absolute path to the script
            current_dir = Path(__file__).parent
            run_script_path = current_dir / "_run_scd.sh"
            script_path = current_dir / "_run_scd.py"

            if not run_script_path.exists():
                raise FileNotFoundError(f"Script not found at {run_script_path}")

            # Build container command with unified run_dir
            cmd = [
                str(run_script_path),
                engine,
                container,
                str(script_path),
                str(run_dir),
            ]

            # Run container
            subprocess.run(cmd, check=True, cwd=current_dir)
            print(f"[INFO] Decomposition completed successfully")
            logger.set_return_code("run.sh", 0)

            # Load results from container output files in run_dir
            results = {}
            
            # Load sources if available
            sources_path = run_dir / "predicted_sources.npz"
            if sources_path.exists():
                sources_data = np.load(sources_path)
                results["sources"] = sources_data["predicted_sources"]
            else:
                results["sources"] = None
            
            # Load spikes if available
            spikes_path = run_dir / "predicted_timestamps.tsv"
            if spikes_path.exists():
                spikes_df = pd.read_csv(spikes_path, sep="\t")
                # Convert back to dictionary format
                spikes_dict = {}
                for unit_id in spikes_df["unit_id"].unique():
                    unit_spikes = spikes_df[spikes_df["unit_id"] == unit_id]["timestamp"].values
                    spikes_dict[unit_id] = unit_spikes.tolist()
                results["spikes"] = spikes_dict
            else:
                results["spikes"] = {}
            
            # SCD doesn't typically provide silhouette scores
            # TODO: Compute silhouette scores
            results["silhouette"] = None
            
            # Log output files for tracking
            for root, _, files in os.walk(run_dir):
                for file in files:
                    if "input_data.npy" in file:
                        continue
                    file_path = os.path.join(root, file)
                    logger.add_output(file_path, os.path.getsize(file_path))

        except Exception as e:
            print(f"[ERROR] SCD decomposition failed: {str(e)}")
            logger.set_return_code("run.sh", 1)
            results = {"sources": None, "spikes": {}, "silhouette": None}

        finally:
            # Always finalize logger to ensure metadata is captured
            logger.finalize(engine, container)
        
        return results, logger.log_data


def decompose_upperbound(
    data: np.ndarray,
    muaps: np.ndarray,
    algorithm_config: Optional[Dict] = None,
    metadata: Optional[Dict] = None,
) -> Tuple[Dict, Dict]:
    """
    Run upperbound decomposition.

    Args:
        data: EMG data array (channels x samples)
        muaps: MUAPs array (n_motor_units x n_channels x duration)
        algorithm_config: Optional path to algorithm configuration JSON file
        metadata: Optional dictionary containing input data metadata for logging

    Returns:
        Tuple containing:
        - Dictionary with decomposition results containing:
          * sources: Estimated sources
          * spikes: Spike timing dictionary
          * silhouette: Quality metrics
          * mu_filters: Motor unit filters
        - Dictionary with processing metadata
    """
    # Initialize logger
    logger = AlgorithmLogger()
    
    # Set input data information
    if metadata:
        logger.set_input_data(file_name=metadata["filename"], file_format=metadata["format"])
    else:
        logger.set_input_data(file_name="numpy_array", file_format="npy")

    # Load and set algorithm configuration
    if algorithm_config:
        # Handle nested Config structure if present
        if "Config" in algorithm_config:
            algo_cfg = algorithm_config["Config"]
        else:
            # Assume the dict is the config itself
            algo_cfg = algorithm_config
        logger.set_algorithm_config(algo_cfg)
    else:
        # Load default configuration
        config_dir = Path(__file__).parent.parent.parent / "configs"
        algorithm_config_path = config_dir / "upperbound.json"
        if not algorithm_config_path.exists():
            raise FileNotFoundError(
                f"Default UpperBound config not found at {algorithm_config_path}"
            )
        algo_cfg = load_config(str(algorithm_config_path))["Config"]
        logger.set_algorithm_config(algo_cfg)

    # Get sampling frequency from config
    fsamp = algo_cfg.get("sampling_frequency", 2048)

    try:
        # Initialize and run upperbound
        # Apply start and end time to data
        start_time = algo_cfg["start_time"] * algo_cfg["sampling_frequency"]
        end_time = algo_cfg["end_time"] * algo_cfg["sampling_frequency"]
        data = data[:, start_time:end_time].copy()
        
        ub = UpperBound(config=SimpleNamespace(**algo_cfg))

        # Validate muaps format
        if muaps.ndim != 3:
            raise ValueError("MUAPs must be a 3D array (n_motor_units x n_channels x duration)")

        # Run decomposition
        sources, spikes, sil, mu_filters = ub.decompose(data, muaps, fsamp=fsamp)

        # Prepare results
        results = {
            "sources": sources,
            "spikes": spikes,
            "silhouette": sil,
            "mu_filters": mu_filters,
        }

        logger.set_return_code("upperbound", 0)
        print(f"[INFO] UpperBound decomposition completed successfully")

    except Exception as e:
        print(f"[ERROR] UpperBound decomposition failed: {str(e)}")
        logger.set_return_code("upperbound", 1)
        results = {"sources": None, "spikes": {}, "silhouette": None, "mu_filters": None}
    
    finally:
        # Always finalize logger to ensure metadata is captured
        logger.finalize()
    
    return results, logger.log_data


def decompose_cbss(
    data: np.ndarray,
    algorithm_config: Optional[Dict] = None,
    metadata: Optional[Dict] = None,
    show_config: bool = False,
) -> Tuple[Dict, Dict]:
    """
    Run CBSS decomposition.

    Args:
        data: numpy array of EMG data (channels x samples)
        algorithm_config: Optional path to algorithm configuration JSON file
        metadata: Optional dictionary containing input data metadata for logging

    Returns:
        Tuple containing:
        - Dictionary with decomposition results containing:
          * sources: Estimated sources
          * spikes: Spike timing dictionary
          * silhouette: Quality metrics
        - Dictionary with processing metadata
    """
    # Initialize logger
    logger = AlgorithmLogger()

    # Set input data information
    if metadata:
        logger.set_input_data(file_name=metadata["filename"], file_format=metadata["format"])
    else:
        logger.set_input_data(file_name="numpy_array", file_format="npy")

    # Load and set algorithm configuration
    if algorithm_config:

        if isinstance(algorithm_config, str):
            # Load config from file path
            algorithm_config = load_config(algorithm_config)
            
        # Handle nested Config structure if present
        if "Config" in algorithm_config:
            algo_cfg = algorithm_config["Config"]
        else:
            # Assume the dict is the config itself
            algo_cfg = algorithm_config
        logger.set_algorithm_config(algo_cfg)
    else:
        # Load default configuration
        config_dir = Path(__file__).parent.parent.parent / "configs"
        algorithm_config = config_dir / "cbss.json"
        if not algorithm_config.exists():
            raise FileNotFoundError(
                f"Default CBSS config not found at {algorithm_config}"
            )
        algo_cfg = load_config(algorithm_config)["Config"]
        logger.set_algorithm_config(algo_cfg)

    if show_config:
        print("CBSS Configuration:")
        for key, value in algo_cfg.items():
            print(f"  {key}: {value}")

    try:
        # Apply start and end time to data
        start_time = algo_cfg["start_time"] * algo_cfg["sampling_frequency"]
        end_time = algo_cfg["end_time"] * algo_cfg["sampling_frequency"]
        data = data[:, int(start_time):int(end_time)]

        # Initialize and run CBSS with config
        cbss = CBSS(config=SimpleNamespace(**algo_cfg))
        sources, spikes, sil, mu_filters , Z , centroids = cbss.decompose(
            data, fsamp=algo_cfg["sampling_frequency"]
        )


        # Prepare results
        results = {"sources": sources, "spikes": spikes, "silhouette": sil, "mu_filters": mu_filters, "Z": Z, "centroids": centroids}

        print(f"[INFO] CBSS decomposition completed successfully")
        logger.set_return_code("cbss", 0)

    except Exception as e:
        print(f"[ERROR] CBSS decomposition failed: {str(e)}")
        logger.set_return_code("cbss", 1)
        results = {"sources": None, "spikes": {}, "silhouette": None, "mu_filters": None, "Z": None, "centroids": None}
    
    finally:
        # Always finalize logger to ensure metadata is captured
        logger.finalize()

    return results, logger.log_data


# Keys decompose_cbss_v2 consumes itself. They are removed before the rest of the
# config is handed to CBSSv2, which warns about attributes it does not have.
_CBSS_V2_WRAPPER_KEYS = ("start_time", "end_time", "sampling_frequency", "Name", "Description")


def _coerce_config(value):
    """Map the string "None" onto None, recursively, in a loaded config.

    Several configs in this repository write a disabled option as the string
    "None" rather than JSON null. CBSSv2 treats None as "off" and any string as a
    value, so the two spellings are made equivalent here.

    Args:
        value: Any value from a parsed config: dict, list, or scalar.

    Returns:
        The same structure with "None", "none", "null" and "" replaced by None.
    """
    if isinstance(value, dict):
        return {k: _coerce_config(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_coerce_config(v) for v in value]
    if isinstance(value, str) and value.strip().lower() in ("none", "null", ""):
        return None
    return value


def decompose_cbss_v2(
    data: np.ndarray,
    algorithm_config: Optional[Union[Dict, str]] = None,
    metadata: Optional[Dict] = None,
    show_config: bool = False,
) -> Tuple[Dict, Dict]:
    """
    Run CBSS v2 decomposition: CBSS with channel preprocessing, rank-truncated
    whitening and motor unit validity scoring.

    Args:
        data: EMG data (channels x samples).
        algorithm_config: Config as a dict, or a path to a JSON file. A "Config"
            key is unwrapped when present, otherwise the dict is taken as the
            config itself. None loads configs/cbss_v2.json. Keys are the CBSSv2
            attributes plus start_time, end_time and sampling_frequency, which
            this function consumes. Both the CBSS parameters and the ones
            introduced by corrected_ckc are set here; see CBSSv2 for the list.
        metadata: Optional dict for logging, keys "filename" and "format".
        show_config: Print the resolved config before running.

    Returns:
        Tuple of (results, log_data). ``results`` holds the six CBSS outputs
        followed by the five corrected_ckc outputs:
          * sources: accepted sources (n_units x n_samples)
          * spikes: unit index to spike sample indices
          * silhouette: silhouette score per unit
          * mu_filters: separation filters (n_kept_dims x n_units)
          * Z: whitening matrix (n_kept_dims x n_kept_channels * ext_fact),
            applies to the extended preprocessed signal
          * centroids: unit index to spike-detection cluster centroids
          * units: validity scores per unit, ordered to match the sources rows
          * keep: channel indices kept by preprocessing, indexing ``data`` rows
          * dropped: channel indices dropped by preprocessing
          * n_kept_dims: whitened dimensions retained
          * preprocessed: the signal the units were scored against, X in
            corrected_ckc terms (n_kept_channels x n_samples)

        and two entries describing the hardened gate itself:
          * n_units_before_validity: units that reached the gate, that is the
            survivors of the silhouette, CoV and duplicate steps. Equals
            len(units) unless validity_gate is "filter", where units holds only
            the ones that passed.
          * validity_report: dict with n_before, n_after, passed, refused,
            refused_by (criterion name to the unit indices it refused),
            n_refused_by, gate (the thresholds applied) and satisfiable. Indices
            refer to the numbering before filtering. A unit failing several
            criteria appears under each of them.

        On failure every key is None, except spikes, units, keep and dropped
        which are empty, and n_kept_dims and n_units_before_validity which are 0.
        This function does not raise, apart from a missing default config file.

    Notes:
        Spike indices refer to the sliced signal. With start_time set they are
        offset from the start of ``data`` by start_time * sampling_frequency.
        end_time below 0, or null, runs to the end of the recording.
    """
    # Initialize logger
    logger = AlgorithmLogger()

    # Set input data information
    if metadata:
        logger.set_input_data(file_name=metadata.get("filename", "numpy_array"),
                              file_format=metadata.get("format", "npy"))
    else:
        logger.set_input_data(file_name="numpy_array", file_format="npy")

    # Load and set algorithm configuration
    if algorithm_config:
        if isinstance(algorithm_config, str):
            algorithm_config = load_config(algorithm_config)
        algo_cfg = algorithm_config.get("Config", algorithm_config)
    else:
        config_dir = Path(__file__).parent.parent.parent / "configs"
        config_path = config_dir / "cbss_v2.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Default CBSS v2 config not found at {config_path}")
        algo_cfg = load_config(str(config_path))["Config"]

    algo_cfg = _coerce_config(algo_cfg)
    logger.set_algorithm_config(algo_cfg)

    if show_config:
        print("CBSS v2 Configuration:")
        for key, value in algo_cfg.items():
            print(f"  {key}: {value}")

    results = {"sources": None, "spikes": {}, "silhouette": None, "mu_filters": None,
               "Z": None, "centroids": None, "units": [], "keep": None, "dropped": None,
               "n_kept_dims": 0, "preprocessed": None,
               "n_units_before_validity": 0, "validity_report": None,
               "channel_report": None}

    try:
        fsamp = algo_cfg.get("sampling_frequency", 2048)

        # Apply start and end time to data. end_time below 0 runs to the end.
        start = int(round(algo_cfg.get("start_time", 0) * fsamp))
        end_time = algo_cfg.get("end_time", -1)
        end = None if end_time is None or end_time < 0 else int(round(end_time * fsamp))
        data = np.asarray(data)[:, start:end]

        # Hand CBSSv2 only the keys it owns
        cbss_cfg = {k: v for k, v in algo_cfg.items() if k not in _CBSS_V2_WRAPPER_KEYS}

        cbss = CBSSv2(config=SimpleNamespace(**cbss_cfg))
        (sources, spikes, sil, mu_filters, Z, centroids,
         units, keep, dropped, n_kept_dims, X) = cbss.decompose(data, fsamp=fsamp)

        report = cbss.validity_report
        results = {"sources": sources, "spikes": spikes, "silhouette": sil,
                   "mu_filters": mu_filters, "Z": Z, "centroids": centroids,
                   "units": units, "keep": keep, "dropped": dropped,
                   "n_kept_dims": n_kept_dims, "preprocessed": X,
                   "n_units_before_validity": report["n_before"],
                   "validity_report": report,
                   "channel_report": cbss.channel_report}

        print(f"[INFO] CBSS v2 decomposition completed successfully: "
              f"{report['n_before']} unit(s) before the hardened gate, "
              f"{report['n_after']} passing, "
              f"{keep.size} channel(s) kept, {n_kept_dims} whitened dimension(s)")
        if report["satisfiable"] is False:
            print(f"[WARN] The validity gate cannot be satisfied on {X.shape[0]} channels: "
                  f"min_n_focal_ch={report['gate']['min_n_focal_ch']} and "
                  f"max_focality_frac={report['gate']['max_focality_frac']} conflict")
        for crit, ids in report["refused_by"].items():
            if ids:
                print(f"[INFO]   refused by {crit}: {len(ids)} unit(s)")
        logger.set_return_code("cbss_v2", 0)

    except Exception as e:
        print(f"[ERROR] CBSS v2 decomposition failed: {str(e)}")
        logger.set_return_code("cbss_v2", 1)

    finally:
        # Always finalize logger to ensure metadata is captured
        logger.finalize()

    return results, logger.log_data


def decompose_ae(
    data: np.ndarray,
    algorithm_config: Optional[Dict] = None,
    metadata: Optional[Dict] = None,
) -> Tuple[Dict, Dict]:
    """
    Run Autoencoder-based decomposition.

    Args:
        data: EMG data (channels x samples)
        algorithm_config: Optional dict with "Config" key or the raw config dict
        metadata: Optional dict for logging (e.g., {"filename": "...", "format": "..."})

    Returns:
        results dict with:
            - sources: (n_units x n_samples)
            - spikes:  {unit_id: np.ndarray of sample indices}
            - silhouette: np.ndarray
            - mu_filters: (n_units x (m*R))
        and the logger data dict
    """
    logger = AlgorithmLogger()

    # Input data metadata
    if metadata:
        logger.set_input_data(
            file_name=metadata.get("filename", "numpy_array"),
            file_format=metadata.get("format", "npy"),
        )
    else:
        logger.set_input_data(file_name="numpy_array", file_format="npy")

    # Load config
    if algorithm_config:
        algo_cfg = algorithm_config.get("Config", algorithm_config)
        logger.set_algorithm_config(algo_cfg)
    else:
        config_dir = Path(__file__).parent.parent.parent / "configs"
        config_path = config_dir / "ae_decomposer.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Default AE config not found at {config_path}")
        algo_cfg = load_config(str(config_path))["Config"]
        logger.set_algorithm_config(algo_cfg)

    try:
        # Build strongly-typed config directly (no key remapping)
        ae_cfg = AEDecoderConfig(**algo_cfg)

        # Slice time window using config (seconds)
        fsamp = float(ae_cfg.sampling_frequency)
        start_idx = int(round(ae_cfg.start_time * fsamp))
        end_idx = int(round(ae_cfg.end_time * fsamp))
        data = data[:, start_idx:end_idx].copy()

        # Run AE
        ae = AEDecoder(ae_cfg)
        sources, spikes, sil, mu_filters = ae.decompose(data, fsamp=fsamp)

        results = {
            "sources": sources,
            "spikes": spikes,
            "silhouette": sil,
            "mu_filters": mu_filters,
        }

        logger.set_return_code("ae_decomposer", 0)
        print("[INFO] AE decomposition completed successfully")

    except Exception as e:
        print(f"[ERROR] AE decomposition failed: {e}")
        logger.set_return_code("ae_decomposer", 1)
        results = {"sources": None, "spikes": {}, "silhouette": None, "mu_filters": None}

    finally:
        # Always finalize logger to ensure metadata is captured
        logger.finalize()
    
    return results, logger.log_data


#--------------------------------------------------MINE-----------------------------------------

"""
High-level wrapper for the MUAP-based decomposition, matching the style of the
other ``decompose_*`` helpers (returns ``(results, log_data)`` and never
raises). Add this to your decomposers module next to ``decompose_cbss``.
"""

from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Optional, Tuple

import numpy as np

from ..utils.logging import AlgorithmLogger
from .muap import MUAPDecomposition


def decompose_muap(
    train_data: np.ndarray,
    test_data: np.ndarray,
    algorithm_config: Optional[Dict] = None,
    metadata: Optional[Dict] = None,
    mu_filters: Optional[np.ndarray] = None,
    muaps: Optional[list] = None,
    train_spikes: Optional[list] = None,
    stim_data: Optional[np.ndarray] = None,
    show_config: bool = False,
) -> Tuple[Dict, Dict]:
    """
    Run MUAP-based EMG decomposition (Chen, Li & Xia 2025).

    Args:
        train_data: EMG used to reconstruct MU filters (channels x samples), x1.
        test_data:  EMG to be decomposed (channels x samples), x2.
        algorithm_config: Optional dict (optionally nested under "Config").
        metadata: Optional dict for logging ({"filename": ..., "format": ...}).
        mu_filters / muaps: Optionally supply a pre-built filter bank to skip
            Stage 1 (e.g. a cross-condition MU-filter library).
        train_spikes: Optionally supply pre-computed discharge times (iterable
            of per-unit sample-index arrays, e.g.
            ``list(cbss_results["spikes"].values())``) to skip only the gCKC
            search and reconstruct the filters by STA at these spikes. Ignored
            when ``mu_filters`` is given; indices must match the (cropped)
            ``train_data`` timeline (mind ``start_time``/``end_time``).
        stim_data: Optional recording containing only the stimulation artifact
            (channels x samples). With ``remove_stim_artifact`` set in the
            config, its template is subtracted from the test signal.
        show_config: Print the resolved configuration.

    Returns:
        (results, log_data) where results contains:
          * sources, spikes, silhouette, mu_filters, muaps, centroids
          * train: dict of the Stage-1 (training) decomposition results
            (sources, spikes, silhouette, cov, mu_filters, muaps, centroids),
            or None when a pre-built filter bank is supplied.
    """
    logger = AlgorithmLogger()
    logger.add_generated_by(
        name="MUAP-based CKC decomposition (Chen, Li & Xia 2025)",
        url="https://doi.org/10.1186/s12984-025-01595-y",
        commit="",
        license="Creative Commons Attribution-NonCommercial-NoDerivatives 4.0",
    )

    if metadata:
        logger.set_input_data(file_name=metadata.get("filename", "numpy_array"),
                              file_format=metadata.get("format", "npy"))
    else:
        logger.set_input_data(file_name="numpy_array", file_format="npy")

    # Resolve configuration (supports nested {"Config": {...}} or a flat dict)
    if algorithm_config:
        if isinstance(algorithm_config, str):
            import json
            with open(algorithm_config, "r") as f:
                algorithm_config = json.load(f)
        algo_cfg = algorithm_config.get("Config", algorithm_config)
        logger.set_algorithm_config(algo_cfg)
    else:
        config_dir = Path(__file__).parent.parent.parent / "configs"
        config_path = config_dir / "muap.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Default MUAP config not found at {config_path}")
        import json
        with open(config_path, "r") as f:
            algo_cfg = json.load(f)["Config"]
        logger.set_algorithm_config(algo_cfg)

    if show_config:
        print("MUAP Configuration:")
        for key, value in algo_cfg.items():
            print(f"  {key}: {value}")

    fsamp = algo_cfg.get("sampling_frequency", 2048)

    try:
        # Optional time-window slicing on both signals (CBSS-style)
        stim = None if stim_data is None else np.asarray(stim_data)
        if "start_time" in algo_cfg and "end_time" in algo_cfg:
            s = int(algo_cfg["start_time"] * fsamp)
            e = int(algo_cfg["end_time"] * fsamp)
            train = np.asarray(train_data)[:, s:e]
            test = np.asarray(test_data)[:, s:e]
            if stim is not None:
                stim = stim[:, s:e]
        else:
            train = np.asarray(train_data)
            test = np.asarray(test_data)

        decomposer = MUAPDecomposition(config=SimpleNamespace(**algo_cfg))
        (sources, spikes, sil, filters, est_muaps, centroids,
         train_results, test_clean) = decomposer.decompose(
            train, test, fsamp=fsamp, mu_filters=mu_filters, muaps=muaps,
            train_spikes=train_spikes, stim_sig=stim,
        )

        results = {
            "sources": sources,
            "spikes": spikes,
            "silhouette": sil,
            "mu_filters": filters,
            "muaps": est_muaps,
            "centroids": centroids,
            "train": train_results,
            "stim": getattr(decomposer, "_stim_info", None),
            "test_clean" : test_clean
        }
        logger.set_return_code("muap", 0)
        print("[INFO] MUAP decomposition completed successfully")

    except Exception as e:
        print(f"[ERROR] MUAP decomposition failed: {str(e)}")
        logger.set_return_code("muap", 1)
        results = {
            "sources": None, "spikes": {}, "silhouette": None,
            "mu_filters": None, "muaps": None, "centroids": None,
            "train": train_results,
            "stim": getattr(decomposer, "_stim_info", None),
            "test_clean" : test_clean
        }

    finally:
        logger.finalize()

    return results, logger.log_data
