import sys
import os
import numpy as np
import pickle as pkl
from enum import IntEnum
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import traceback
from datetime import datetime


from .streaming import stream_with_adaptive_decomposition, stream_with_decomposition, plot_emg_stream
from .decomposition import decompose_emg, prompt_trial_selection, compute_combined_whitening_and_centroids
from .plotting import plot_raster_summary
from .video_game import stream_video_game
from .video_game_serious import stream_cursor_control
from .cue_game import stream_cue_game
from .dof_config import (
    prompt_dof_configuration, serialize_dof_configs, deserialize_dof_configs
)

class Phase(IntEnum):
    """Pipeline phases for checkpoint/resume functionality."""
    RECORDING = 1
    CHANNEL_REVIEW = 2
    DECOMPOSITION = 3
    REALTIME_STREAMING = 4
    MU_SELECTION = 5
    SELECTED_STREAMING = 6
    VIDEO_GAME = 7


class PipelineManager():

    def __init__(self, output_folder, trial_name, config_path):
        self.output_folder = output_folder
        self.trial_name = trial_name
        self.config_path = config_path
        self.checkpoint_path = os.path.join(output_folder, f"checkpoint_{trial_name}.pkl")

    def _save_checkpoint(self, phase, **data):
        """Save checkpoint with current phase and data."""
        checkpoint = {
            "phase": phase,
            "trial_name": self.trial_name,
            **data
        }
        with open(self.checkpoint_path, "wb") as f:
            pkl.dump(checkpoint, f)
        print(f"Checkpoint saved at phase {phase.name}")

    def _load_checkpoint(self):
        """Load checkpoint if it exists."""
        if os.path.exists(self.checkpoint_path):
            with open(self.checkpoint_path, "rb") as f:
                return pkl.load(f)
        return None

    def list_checkpoints(self):
        """List available checkpoint information."""
        checkpoint = self._load_checkpoint()
        available_files = self._check_available_files()

        print(f"\nTrial: {self.trial_name}")
        print("-" * 40)

        if checkpoint:
            print(f"Checkpoint: phase {Phase(checkpoint['phase']).name}")
        else:
            print("Checkpoint: None")

        print(f"\nAvailable output files:")
        for phase, exists in available_files.items():
            status = "✓" if exists else "✗"
            print(f"  {status} {phase.name}")

        print(f"\nYou can start from:")
        for phase in Phase:
            if phase == Phase.RECORDING:
                print(f"  {phase.value}. {phase.name} (always available)")
            elif phase == Phase.CHANNEL_REVIEW:
                if checkpoint and checkpoint.get("raw_recorded_trials"):
                    print(f"  {phase.value}. {phase.name} (from checkpoint)")
            elif phase == Phase.DECOMPOSITION:
                if checkpoint and checkpoint.get("recorded_trials"):
                    print(f"  {phase.value}. {phase.name} (from checkpoint)")
            elif available_files.get(phase, False):
                print(f"  {phase.value}. {phase.name}")

        return checkpoint

    def _check_available_files(self):
        """Check which output files exist."""
        return {
            Phase.REALTIME_STREAMING: os.path.exists(
                os.path.join(self.output_folder, f"results_{self.trial_name}_combined.pkl")
            ),
            Phase.MU_SELECTION: os.path.exists(
                os.path.join(self.output_folder, f"streaming_results_{self.trial_name}.pkl")
            ),
            Phase.SELECTED_STREAMING: os.path.exists(
                os.path.join(self.output_folder, f"results_{self.trial_name}_selected.pkl")
            ),
            Phase.VIDEO_GAME: os.path.exists(
                os.path.join(self.output_folder, f"results_{self.trial_name}_selected.pkl")
            ),
        }

    def _load_from_files(self, start_from):
        """Load required data from output files based on starting phase."""
        data = {}

        # Load decomposition results (needed for phases 3+)
        if start_from >= Phase.REALTIME_STREAMING:
            combined_path = os.path.join(self.output_folder, f"results_{self.trial_name}_combined.pkl")
            if os.path.exists(combined_path):
                with open(combined_path, "rb") as f:
                    combined_results, combined_metadata = pkl.load(f)
                data["combined_path"] = combined_path
                data["combined_results"] = combined_results
                data["combined_metadata"] = combined_metadata
                # Extract channels_to_remove if available
                data["channels_to_remove"] = combined_results.get("channels_to_remove", None)
                print(f"Loaded decomposition results: {combined_metadata['n_mus']} motor units")
            else:
                print(f"Error: Decomposition results not found at {combined_path}")
                return None

        # Load streaming results (needed for phase 4)
        if start_from >= Phase.MU_SELECTION:
            streaming_path = os.path.join(self.output_folder, f"streaming_results_{self.trial_name}.pkl")
            if os.path.exists(streaming_path):
                with open(streaming_path, "rb") as f:
                    streaming_results = pkl.load(f)
                data["streaming_results"] = streaming_results
                print(f"Loaded streaming results: {streaming_results['n_mus']} motor units")
            else:
                print(f"Error: Streaming results not found at {streaming_path}")
                return None

        # Load selected MU results (needed for phases 5-6)
        if start_from >= Phase.SELECTED_STREAMING:
            selected_path = os.path.join(self.output_folder, f"results_{self.trial_name}_selected.pkl")
            if os.path.exists(selected_path):
                with open(selected_path, "rb") as f:
                    selected_results, selected_metadata = pkl.load(f)
                data["selected_path"] = selected_path
                data["selected_units"] = selected_metadata.get("original_mu_indices", [])
                data["dof_configs"] = selected_metadata.get("dof_configs", None)
                print(f"Loaded selected MUs: {data['selected_units']}")
                if data["dof_configs"]:
                    print(f"Loaded DOF configs: {len(data['dof_configs'])} DOFs")
            else:
                print(f"Error: Selected MU results not found at {selected_path}")
                return None

        return data

    def _prompt_user(self, message, wait_for_key=True):
        """Display a message and optionally wait for user input."""
        print("\n" + "=" * 60)
        print(message)
        print("=" * 60)
        if wait_for_key:
            input("Press ENTER to continue...")

    def _record_trial(self, trial_num):
        """Record a single EMG trial."""
        print(f"\nStarting recording for Trial {trial_num}...")
        print(">>> Close the plot window when you want to stop recording <<<\n")

        emg_data = plot_emg_stream(
            max_channels=None,
            display_seconds=50,
        )

        if emg_data is None:
            print(f"Warning: No EMG data collected for Trial {trial_num}.")
            return None

        print(f"\nTrial {trial_num} recording complete!")
        print(f"Recorded {emg_data['data'].shape[1] / emg_data['srate']:.1f} seconds of data")
        return emg_data

    def _stack_mu_filters(self, all_results):
        """Stack MU filter matrices from multiple decomposition results."""
        mu_filters_list = [r["mu_filters"] for r in all_results if r.get("mu_filters") is not None]

        if not mu_filters_list:
            return None

        # Stack filters horizontally (W is of shape n_ext x n__mus, so along the MU dimension)
        stacked_filters = np.hstack(mu_filters_list)
        return stacked_filters

    def _stack_centroids(self, all_results):
        """Stack centroids from multiple decomposition results with re-indexed keys."""
        stacked_centroids = {}
        mu_offset = 0

        for results in all_results:
            centroids = results.get("centroids")
            if centroids is None:
                # Count MUs from this result to maintain proper offset
                n_mus = results["mu_filters"].shape[1] if results.get("mu_filters") is not None else 0
                mu_offset += n_mus
                continue

            # Re-index centroids with offset
            for mu_idx, centroid_values in centroids.items():
                new_idx = int(mu_idx) + mu_offset
                stacked_centroids[new_idx] = centroid_values

            # Update offset for next trial
            n_mus = results["mu_filters"].shape[1] if results.get("mu_filters") is not None else 0
            mu_offset += n_mus

        return stacked_centroids if stacked_centroids else None

    def _extract_selected_mus(self, combined_results, selected_units):
        """Extract only the selected MUs from combined results."""
        selected_results = combined_results.copy()

        # Extract only selected MU filters (columns)
        mu_filters = combined_results["mu_filters"]
        selected_results["mu_filters"] = mu_filters[:, selected_units]

        # Extract only selected centroids with re-indexed keys (0, 1, ...)
        centroids = combined_results.get("centroids")
        if centroids is not None:
            selected_centroids = {}
            for new_idx, old_idx in enumerate(selected_units):
                if old_idx in centroids:
                    selected_centroids[new_idx] = centroids[old_idx]
            selected_results["centroids"] = selected_centroids if selected_centroids else None
        else:
            selected_results["centroids"] = None

        # Extract only selected norm_factors with re-indexed keys (0, 1, ...)
        norm_factors = combined_results.get("norm_factors")
        if norm_factors is not None:
            selected_norm_factors = {}
            for new_idx, old_idx in enumerate(selected_units):
                if old_idx in norm_factors:
                    selected_norm_factors[new_idx] = norm_factors[old_idx]
            selected_results["norm_factors"] = selected_norm_factors if selected_norm_factors else None
        else:
            selected_results["norm_factors"] = None

        # Preserve channels_to_remove for streaming
        selected_results["channels_to_remove"] = combined_results.get("channels_to_remove", [])

        return selected_results

    def _report_exception(self, context: str, exc: BaseException) -> None:
        """
        Print and persist a full traceback for debugging.
        In ProcessPoolExecutor failures, the useful traceback is often in exc.__cause__.
        """
        parts = []
        parts.append("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))

        cause = getattr(exc, "__cause__", None)
        if cause is not None:
            parts.append("\n--- Exception __cause__ (often remote traceback) ---\n")
            parts.append("".join(traceback.format_exception(type(cause), cause, getattr(cause, "__traceback__", None))))

        tb_text = "".join(parts)

        print(f"\nERROR during {context}:\n{tb_text}")

        try:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_name = f"error_{self.trial_name}_{self._safe_slug(context)}_{ts}.log"
            log_path = os.path.join(self.output_folder, log_name)
            with open(log_path, "w", encoding="utf-8") as f:
                f.write(tb_text)
            print(f"Full traceback saved to: {log_path}")
        except Exception as log_exc:
            print(f"Additionally failed to write error log: {log_exc}")

    def _safe_slug(self, s: str) -> str:
        """Make a string safe-ish for filenames."""
        s = str(s)
        return "".join(ch if (ch.isalnum() or ch in ("-", "_")) else "_" for ch in s)[:120]

    def _prompt_channels_to_remove_for_streaming(self):
        """
        Prompt user to specify channels to remove for streaming when loading
        from an older checkpoint that doesn't have this info.

        Returns:
            List of channel indices to remove (empty if none)
        """
        print(f"\n" + "-" * 40)
        print(f"CHANNEL CONFIGURATION FOR STREAMING")
        print(f"-" * 40)
        print("The loaded model may have been trained with certain channels removed.")
        print("You need to specify which channels to exclude from the live stream.")

        response = input("\nWere any channels removed during calibration? (y/n): ").strip().lower()
        if response != 'y':
            print("Using all channels for streaming.")
            return []

        print("\nEnter channel indices to remove (comma-separated, e.g., '3,7,12')")
        print("Or enter 'none' to use all channels:")

        while True:
            user_input = input("Channels to remove: ").strip().lower()

            if user_input == 'none' or user_input == '':
                print("Using all channels for streaming.")
                return []

            try:
                channels_to_remove = [int(x.strip()) for x in user_input.split(',')]

                # Validate no negative indices
                invalid = [ch for ch in channels_to_remove if ch < 0]
                if invalid:
                    print(f"Invalid channel indices: {invalid}. Must be >= 0.")
                    continue

                # Check for duplicates
                if len(channels_to_remove) != len(set(channels_to_remove)):
                    print("Duplicate channel indices detected. Please enter unique values.")
                    continue

                # Confirm
                print(f"\nChannels to remove from live stream: {sorted(channels_to_remove)}")
                confirm = input("Confirm? (y/n): ").strip().lower()

                if confirm == 'y':
                    return sorted(channels_to_remove)
                else:
                    print("Cancelled. Enter new channel list or 'none':")

            except ValueError:
                print("Invalid input. Enter comma-separated integers (e.g., '3,7,12') or 'none'.")

    def _prompt_channel_removal(self, recorded_trials):
        """
        Prompt user to optionally remove channels from recorded EMG data.

        Args:
            recorded_trials: List of recorded EMG trial dicts with 'data' key

        Returns:
            List of channel indices to remove (empty if none)
        """
        if not recorded_trials:
            return []

        # Get channel count from first trial
        n_channels = recorded_trials[0]["data"].shape[0]

        print(f"\n" + "-" * 40)
        print(f"CHANNEL REVIEW")
        print(f"-" * 40)
        print(f"Recorded data has {n_channels} channels (indices 0 to {n_channels - 1})")

        response = input("\nDo you want to remove any channels? (y/n): ").strip().lower()
        if response != 'y':
            print("Keeping all channels.")
            return []

        print("\nEnter channel indices to remove (comma-separated, e.g., '3,7,12')")
        print("Or enter 'none' to cancel:")

        while True:
            user_input = input("Channels to remove: ").strip().lower()

            if user_input == 'none' or user_input == '':
                print("No channels removed.")
                return []

            try:
                # Parse comma-separated indices
                channels_to_remove = [int(x.strip()) for x in user_input.split(',')]

                # Validate indices
                invalid = [ch for ch in channels_to_remove if ch < 0 or ch >= n_channels]
                if invalid:
                    print(f"Invalid channel indices: {invalid}. Must be 0 to {n_channels - 1}.")
                    continue

                # Check for duplicates
                if len(channels_to_remove) != len(set(channels_to_remove)):
                    print("Duplicate channel indices detected. Please enter unique values.")
                    continue

                # Check we're not removing all channels
                if len(channels_to_remove) >= n_channels:
                    print(f"Cannot remove all {n_channels} channels. At least 1 must remain.")
                    continue

                # Confirm
                print(f"\nChannels to remove: {sorted(channels_to_remove)}")
                print(f"Remaining channels: {n_channels - len(channels_to_remove)}")
                confirm = input("Confirm removal? (y/n): ").strip().lower()

                if confirm == 'y':
                    return sorted(channels_to_remove)
                else:
                    print("Cancelled. Enter new channel list or 'none':")

            except ValueError:
                print("Invalid input. Enter comma-separated integers (e.g., '3,7,12') or 'none'.")

    def _remove_channels(self, recorded_trials, channels_to_remove):
        """
        Remove specified channels from all recorded trials.

        Args:
            recorded_trials: List of recorded EMG trial dicts with 'data' key
            channels_to_remove: List of channel indices to remove

        Returns:
            Modified recorded_trials with channels removed
        """
        if not channels_to_remove:
            return recorded_trials

        channels_to_remove = set(channels_to_remove)

        for i, trial in enumerate(recorded_trials):
            original_data = trial["data"]
            n_channels = original_data.shape[0]

            # Create mask for channels to keep
            channels_to_keep = [ch for ch in range(n_channels) if ch not in channels_to_remove]

            # Apply mask
            trial["data"] = original_data[channels_to_keep, :]

            if i == 0:
                print(f"Removed {len(channels_to_remove)} channels: {n_channels} -> {trial['data'].shape[0]} channels")

        return recorded_trials

    def run(self, start_from=Phase.RECORDING):
        """
        Run the pipeline, optionally starting from a specific phase.

        Args:
            start_from: Phase enum value to start from. Options:
                - Phase.RECORDING (1): Start fresh with recording
                - Phase.CHANNEL_REVIEW (2): Load raw trials, review/remove channels
                - Phase.DECOMPOSITION (3): Load processed trials, run decomposition
                - Phase.REALTIME_STREAMING (4): Load decomposition results, run streaming
                - Phase.MU_SELECTION (5): Load streaming results, select MUs
                - Phase.SELECTED_STREAMING (6): Load selected MUs, run streaming
                - Phase.VIDEO_GAME (7): Load selected MUs, run video game

        Data is loaded from checkpoint if available, otherwise from output files.
        """
        # Try to load data if starting from a later phase
        checkpoint = None
        file_data = None

        if start_from > Phase.RECORDING:
            # First try checkpoint
            checkpoint = self._load_checkpoint()

            if checkpoint and checkpoint["phase"] >= start_from.value - 1:
                print(f"Loaded checkpoint from phase {Phase(checkpoint['phase']).name}")
            else:
                # Fall back to loading from output files
                checkpoint = None
                if start_from == Phase.CHANNEL_REVIEW:
                    print(f"Error: Cannot start from {start_from.name} without checkpoint.")
                    print("Raw recorded trials are only saved in checkpoints.")
                    return None
                if start_from == Phase.DECOMPOSITION:
                    print(f"Error: Cannot start from {start_from.name} without checkpoint.")
                    print("Recorded trials are only saved in checkpoints.")
                    return None

                print("No valid checkpoint, loading from output files...")
                file_data = self._load_from_files(start_from)
                if file_data is None:
                    return None

        # Initialize variables that may be loaded from checkpoint
        raw_recorded_trials = []  # Original data before channel removal
        recorded_trials = []      # Processed data after channel removal
        channels_to_remove = []   # Channels removed during calibration (for streaming)
        combined_path = None
        combined_results = None
        combined_metadata = None
        streaming_results = None
        selected_units = None
        selected_path = None
        dof_configs_serialized = None

        # Phase 1: Recording
        if start_from <= Phase.RECORDING:
            self._prompt_user(
                "PHASE 1: CALIBRATION RECORDING\n\n"
                "You will now record EMG data for motor unit decomposition.\n"
                "You can record multiple trials to improve decomposition.\n"
                "Please prepare to perform the target movement.\n\n"
                "Instructions:\n"
                "  1. Get into position for the movement\n"
                "  2. Press ENTER to start recording\n"
                "  3. Perform the movement steadily\n"
                "  4. Close the plot window when finished\n"
                "  5. You will be asked if you want to record another trial"
            )

            trial_num = 1
            while True:
                emg_data = self._record_trial(trial_num)
                if emg_data is not None:
                    raw_recorded_trials.append(emg_data)

                print("\n" + "-" * 40)
                response = input(f"Trial {trial_num} complete. Record another trial? (y/n): ").strip().lower()
                if response != 'y':
                    break

                trial_num += 1
                self._prompt_user(
                    f"RECORDING TRIAL {trial_num}\n\n"
                    "Prepare for the next recording.\n"
                    "Perform the same movement as before."
                )

            if not raw_recorded_trials:
                print("Error: No EMG data collected. Aborting.")
                return None

            print(f"\n{'=' * 60}")
            print(f"Recorded {len(raw_recorded_trials)} trial(s) successfully")
            print(f"{'=' * 60}")

            # Save raw EMG data as a separate file (before any channel removal)
            emg_data_path = os.path.join(self.output_folder, f"emg_data_{self.trial_name}_raw.pkl")
            with open(emg_data_path, "wb") as f:
                pkl.dump(raw_recorded_trials, f)
            print(f"Raw EMG data saved to: {emg_data_path}")

            # Save checkpoint after recording (with raw data for potential re-processing)
            self._save_checkpoint(Phase.RECORDING, raw_recorded_trials=raw_recorded_trials)
        else:
            # Load raw recorded trials from checkpoint (needed for CHANNEL_REVIEW)
            if checkpoint:
                raw_recorded_trials = checkpoint.get("raw_recorded_trials", [])
                if not raw_recorded_trials:
                    # Fallback for older checkpoints that used recorded_trials
                    raw_recorded_trials = checkpoint.get("recorded_trials", [])
                print(f"Loaded {len(raw_recorded_trials)} raw recorded trial(s) from checkpoint")

        # Phase 2: Channel Review
        if start_from <= Phase.CHANNEL_REVIEW:
            # Deep copy raw data so we don't modify the originals
            import copy
            recorded_trials = copy.deepcopy(raw_recorded_trials)

            # Prompt for channel removal
            channels_to_remove = self._prompt_channel_removal(recorded_trials)
            if channels_to_remove:
                recorded_trials = self._remove_channels(recorded_trials, channels_to_remove)

            # Save processed EMG data
            emg_data_path = os.path.join(self.output_folder, f"emg_data_{self.trial_name}.pkl")
            with open(emg_data_path, "wb") as f:
                pkl.dump(recorded_trials, f)
            print(f"Processed EMG data saved to: {emg_data_path}")

            # Save checkpoint after channel review
            self._save_checkpoint(
                Phase.CHANNEL_REVIEW,
                raw_recorded_trials=raw_recorded_trials,
                recorded_trials=recorded_trials,
                channels_to_remove=channels_to_remove
            )
        else:
            # Load processed recorded trials from checkpoint (needed for DECOMPOSITION+)
            if checkpoint:
                recorded_trials = checkpoint.get("recorded_trials", [])
                if not recorded_trials:
                    # If no processed trials, use raw (e.g., no channels were removed)
                    recorded_trials = checkpoint.get("raw_recorded_trials", [])
                channels_to_remove = checkpoint.get("channels_to_remove", None)
                print(f"Loaded {len(recorded_trials)} processed trial(s) from checkpoint")
                if channels_to_remove is not None:
                    if channels_to_remove:
                        print(f"Channels removed during calibration: {channels_to_remove}")
                    else:
                        print("No channels were removed during calibration")
                else:
                    # Older checkpoint without channels_to_remove info - prompt user
                    print("\nWarning: Checkpoint does not contain channel removal info.")
                    channels_to_remove = self._prompt_channels_to_remove_for_streaming()

        # Phase 2: Decomposition
        if start_from <= Phase.DECOMPOSITION:
            self._prompt_user(
                "PHASE 2: DECOMPOSITION\n\n"
                f"Processing {len(recorded_trials)} recorded trial(s) to extract motor units...\n"
                "Each trial will be decomposed and saved individually.\n"
                "This may take a moment.",
                wait_for_key=False
            )

            all_results = []
            all_metadata = []
            decomp_paths = []

            # Prepare arguments for parallel execution
            decomp_args = [
                (emg_data["data"], self.config_path, self.output_folder, f"{self.trial_name}_trial{i+1}")
                for i, emg_data in enumerate(recorded_trials)
            ]

            n_trials = len(recorded_trials)
            n_workers = min(n_trials, os.cpu_count() or 1)

            if n_trials > 1 and n_workers > 1:
                print(f"\nDecomposing {n_trials} trials in parallel using {n_workers} workers...")

                decompose_fn = partial(decompose_emg, show_plot=False)

                with ProcessPoolExecutor(max_workers=n_workers) as executor:
                    futures = [
                        executor.submit(decompose_fn, *args)
                        for args in decomp_args
                    ]

                    for i, future in enumerate(futures):
                        try:
                            results, metadata, decomp_path = future.result()
                            all_results.append(results)
                            all_metadata.append(metadata)
                            decomp_paths.append(decomp_path)
                            print(f"Trial {i+1}: Extracted {metadata.get('n_mus', 'N/A')} motor units")
                        except Exception as e:
                            self._report_exception(context=f"Decomposition Trial {i+1} (parallel)", exc=e)
            else:
                # Single trial - run sequentially with plot
                for i, args in enumerate(decomp_args):
                    print(f"\nDecomposing Trial {i+1}/{n_trials}...")
                    try:
                        results, metadata, decomp_path = decompose_emg(*args, show_plot=True)
                        all_results.append(results)
                        all_metadata.append(metadata)
                        decomp_paths.append(decomp_path)
                        print(f"Trial {i+1}: Extracted {metadata.get('n_mus', 'N/A')} motor units")
                    except Exception as e:
                        self._report_exception(context=f"Decomposition Trial {i+1} (sequential)", exc=e)

            if not all_results:
                print("Error: All decompositions failed. Aborting.")
                return None

            # Stack MU filters and centroids from all trials
            stacked_mu_filters = self._stack_mu_filters(all_results)
            stacked_centroids = self._stack_centroids(all_results)

            combined_results = all_results[0].copy()
            combined_results["mu_filters"] = stacked_mu_filters
            combined_results["centroids"] = stacked_centroids
            combined_results["individual_results"] = all_results

            combined_metadata = {
                "n_trials": len(all_results),
                "n_mus": stacked_mu_filters.shape[1] if stacked_mu_filters is not None else 0,
                "individual_metadata": all_metadata,
            }

            combined_path = os.path.join(self.output_folder, f"results_{self.trial_name}_combined.pkl")
            with open(combined_path, "wb") as f:
                pkl.dump((combined_results, combined_metadata), f)

            print(f"\nDecomposition complete!")
            print(f"Total motor units extracted: {combined_metadata['n_mus']}")
            print(f"Combined results saved to: {combined_path}")

            # Save checkpoint after decomposition (include all_results for trial re-selection)
            self._save_checkpoint(
                Phase.DECOMPOSITION,
                recorded_trials=recorded_trials,
                combined_path=combined_path,
                combined_results=combined_results,
                combined_metadata=combined_metadata,
                all_results=all_results,
                all_metadata=all_metadata,
                channels_to_remove=channels_to_remove
            )
        else:
            # Load decomposition results from checkpoint or file_data
            if checkpoint:
                combined_path = checkpoint.get("combined_path")
                combined_results = checkpoint.get("combined_results")
                combined_metadata = checkpoint.get("combined_metadata")
                all_results = checkpoint.get("all_results", [])
                all_metadata = checkpoint.get("all_metadata", [])
                if combined_path is None:
                    combined_path = os.path.join(self.output_folder, f"results_{self.trial_name}_combined.pkl")
                    if os.path.exists(combined_path):
                        with open(combined_path, "rb") as f:
                            combined_results, combined_metadata = pkl.load(f)
            elif file_data:
                combined_path = file_data.get("combined_path")
                combined_results = file_data.get("combined_results")
                combined_metadata = file_data.get("combined_metadata")
                all_results = []
                all_metadata = []
                # Get channels_to_remove from file_data
                channels_to_remove_from_file = file_data.get("channels_to_remove", None)
                if channels_to_remove_from_file is not None:
                    channels_to_remove = channels_to_remove_from_file
                    if channels_to_remove:
                        print(f"Channels removed during calibration: {channels_to_remove}")
                    else:
                        print("No channels were removed during calibration")
                else:
                    # Older file without channels_to_remove info - prompt user
                    print("\nWarning: Loaded results do not contain channel removal info.")
                    channels_to_remove = self._prompt_channels_to_remove_for_streaming()

        # Phase 3: Real-time streaming with iterative trial selection
        if start_from <= Phase.REALTIME_STREAMING:
            # Get sampling rate from first recorded trial
            if not recorded_trials:
                raise ValueError("No recorded trials available for sampling rate.")
            fsamp = recorded_trials[0]["srate"]
            n_trials = len(all_results)
            iteration_count = 0

            if n_trials == 0:
                raise ValueError(
                    "No decomposition results available. The checkpoint may be from an older version "
                    "that didn't save all_results. Please re-run from phase STREAMING_DECOMPOSITION."
                )

            # Iterative trial selection loop
            while True:
                iteration_count += 1

                # Step 1: Trial selection (skip if only one trial)
                if n_trials > 1:
                    selected_indices = prompt_trial_selection(n_trials, all_metadata)
                else:
                    selected_indices = [0]
                    print(f"\nSingle trial available, using Trial 1")

                selected_results = [all_results[i] for i in selected_indices]

                # Step 2: Stack MU filters from selected trials
                stacked_mu_filters = self._stack_mu_filters(selected_results)
                n_total_mus = stacked_mu_filters.shape[1]
                print(f"\nStacked {n_total_mus} motor units from {len(selected_indices)} trial(s)")

                # Step 3: Compute combined whitening, recalibrate filters via STA, and compute normalization
                (Z_combined, recalibrated_filters, recalibrated_centroids, norm_factors,
                 recalibrated_sil, recalibrated_spikes, sources, original_filters) = \
                    compute_combined_whitening_and_centroids(
                        recorded_trials=recorded_trials, selected_indices=selected_indices, stacked_mu_filters=stacked_mu_filters, fsamp=fsamp, config_path=self.config_path
                    )

                # Step 4: Build combined results with recalibrated filters and normalization (Farina 2025)
                combined_results = {
                    "sources": sources,
                    "spikes": recalibrated_spikes,
                    "silhouette": recalibrated_sil,
                    "mu_filters": recalibrated_filters,           # Primary: STA-recalibrated filters
                    "mu_filters_original": original_filters,      # Backup: original peel-off filters
                    "Z": Z_combined,
                    "centroids": recalibrated_centroids,
                    "norm_factors": norm_factors,                 # Normalization for [0,1] range
                    "selected_trial_indices": selected_indices,
                    "individual_results": all_results,
                    "channels_to_remove": channels_to_remove,     # Channels removed during calibration
                }

                combined_metadata = {
                    "n_trials": len(selected_indices),
                    "n_mus": n_total_mus,
                    "selected_trial_indices": selected_indices,
                    "individual_metadata": [all_metadata[i] for i in selected_indices],
                }

                # Step 5: Save combined model for streaming
                combined_path = os.path.join(
                    self.output_folder,
                    f"results_{self.trial_name}_combined_iter{iteration_count}.pkl"
                )
                with open(combined_path, "wb") as f:
                    pkl.dump((combined_results, combined_metadata), f)
                print(f"\nCombined model saved to: {combined_path}")

                # Step 6: Stream with the new combined model
                self._prompt_user(
                    f"PHASE 3: REAL-TIME DECOMPOSITION (Iteration {iteration_count})\n\n"
                    f"Streaming with {n_total_mus} motor units from trials {[i+1 for i in selected_indices]}.\n"
                    "Please prepare to perform the movement.\n\n"
                    "Instructions:\n"
                    "  1. Get into position for the movement\n"
                    "  2. Press ENTER to start real-time streaming\n"
                    "  3. Perform the movement - observe motor unit activity\n"
                    "  4. Close the plot window when finished"
                )

                print("\nStarting real-time decomposition stream...")
                print(">>> Close the plot window when you want to stop <<<\n")

                streaming_results = stream_with_decomposition(
                    decomp_path=combined_path,
                    display_seconds=50,
                    enable_filtering=True,
                    sil_threshold=0.5,
                    max_display_mus=8
                )

                if streaming_results is not None:
                    # Step 7: Save streaming results
                    output_path = os.path.join(
                        self.output_folder,
                        f"streaming_results_{self.trial_name}_iter{iteration_count}.pkl"
                    )
                    with open(output_path, "wb") as f:
                        pkl.dump(streaming_results, f)
                    print(f"Streaming results saved to: {output_path}")

                    # Save raster plots
                    raster_save_path = os.path.join(
                        self.output_folder,
                        f"{self.trial_name}_iter{iteration_count}"
                    )
                    plot_raster_summary(
                        streaming_results,
                        emg_channel=0,
                        units_per_plot=4,
                        save_path=raster_save_path
                    )
                    print(f"Raster plots saved to: {raster_save_path}")

                    # Save checkpoint after streaming
                    self._save_checkpoint(
                        Phase.REALTIME_STREAMING,
                        recorded_trials=recorded_trials,
                        combined_path=combined_path,
                        combined_results=combined_results,
                        combined_metadata=combined_metadata,
                        streaming_results=streaming_results,
                        all_results=all_results,
                        all_metadata=all_metadata,
                        selected_indices=selected_indices,
                        channels_to_remove=channels_to_remove
                    )
                else:
                    print("\nWarning: No streaming data was collected.")

                # Step 8: Prompt for next action
                print(f"\n" + "=" * 60)
                print("STREAMING COMPLETE - What would you like to do?")
                print("=" * 60)
                print("  1. Re-select trials and stream again")
                print("  2. Proceed to motor unit selection")
                print("  3. Exit pipeline")

                while True:
                    choice = input("\nEnter choice (1/2/3): ").strip()
                    if choice in ['1', '2', '3']:
                        break
                    print("Invalid choice. Enter 1, 2, or 3")

                if choice == '1':
                    print("\n--- Returning to trial selection ---")
                    continue  # Loop back to trial selection
                elif choice == '3':
                    print("\nExiting pipeline.")
                    return streaming_results
                else:  # choice == '2'
                    print("\n--- Proceeding to motor unit selection ---")
                    break  # Exit loop and continue to Phase 4

            # Save final combined results (for use in subsequent phases)
            final_combined_path = os.path.join(
                self.output_folder,
                f"results_{self.trial_name}_combined.pkl"
            )
            with open(final_combined_path, "wb") as f:
                pkl.dump((combined_results, combined_metadata), f)
            combined_path = final_combined_path

        else:
            # Load streaming results from checkpoint or file_data
            if checkpoint:
                streaming_results = checkpoint.get("streaming_results")
                all_results = checkpoint.get("all_results", [])
                all_metadata = checkpoint.get("all_metadata", [])
                if streaming_results is None:
                    # Try to find latest streaming results file
                    output_path = os.path.join(self.output_folder, f"streaming_results_{self.trial_name}.pkl")
                    if os.path.exists(output_path):
                        with open(output_path, "rb") as f:
                            streaming_results = pkl.load(f)
            elif file_data:
                streaming_results = file_data.get("streaming_results")
                all_results = []
                all_metadata = []

        # Phase 4: MU Selection
        if start_from <= Phase.MU_SELECTION:
            n_mus = streaming_results['n_mus']
            print(f"\n" + "-" * 40)
            print(f"Available motor units: 0 to {n_mus - 1}")

            # Ask how many MUs to select
            while True:
                try:
                    n_select = int(input(f"How many motor units to select? (1-{n_mus}): ").strip())
                    if 1 <= n_select <= n_mus:
                        break
                    else:
                        print(f"Please enter a number between 1 and {n_mus}")
                except ValueError:
                    print("Please enter a valid integer")

            selected_units = []
            for i in range(n_select):
                while True:
                    try:
                        unit = int(input(f"Select motor unit {i + 1} of {n_select}: ").strip())
                        if 0 <= unit < n_mus:
                            if unit in selected_units:
                                print(f"Motor unit {unit} already selected, choose another")
                            else:
                                selected_units.append(unit)
                                break
                        else:
                            print(f"Please enter a number between 0 and {n_mus - 1}")
                    except ValueError:
                        print("Please enter a valid integer")

            print(f"Selected motor units: {selected_units}")

            # Prompt for DOF configuration
            dof_configs = prompt_dof_configuration(n_selected_mus=len(selected_units), n_dofs=2)
            dof_configs_serialized = serialize_dof_configs(dof_configs)

            selected_results = self._extract_selected_mus(combined_results, selected_units)
            selected_metadata = {
                "n_mus": len(selected_units),
                "original_mu_indices": selected_units,
                "dof_configs": dof_configs_serialized,
            }

            selected_path = os.path.join(self.output_folder, f"results_{self.trial_name}_selected.pkl")
            with open(selected_path, "wb") as f:
                pkl.dump((selected_results, selected_metadata), f)

            print(f"\nSelected MU filters saved to: {selected_path}")

            # Save checkpoint after MU selection
            self._save_checkpoint(
                Phase.MU_SELECTION,
                recorded_trials=recorded_trials,
                combined_path=combined_path,
                combined_results=combined_results,
                combined_metadata=combined_metadata,
                streaming_results=streaming_results,
                selected_units=selected_units,
                selected_path=selected_path,
                dof_configs=dof_configs_serialized,
                channels_to_remove=channels_to_remove
            )
        else:
            # Load selected MUs from checkpoint or file_data
            if checkpoint:
                selected_units = checkpoint.get("selected_units")
                selected_path = checkpoint.get("selected_path")
                dof_configs_serialized = checkpoint.get("dof_configs")
                if selected_path is None:
                    selected_path = os.path.join(self.output_folder, f"results_{self.trial_name}_selected.pkl")
                    # Try to load DOF configs from file if not in checkpoint
                    if dof_configs_serialized is None and os.path.exists(selected_path):
                        with open(selected_path, "rb") as f:
                            _, sel_meta = pkl.load(f)
                        dof_configs_serialized = sel_meta.get("dof_configs")
            elif file_data:
                selected_units = file_data.get("selected_units")
                selected_path = file_data.get("selected_path")
                dof_configs_serialized = file_data.get("dof_configs")
            print(f"Loaded selected motor units: {selected_units}")
            if dof_configs_serialized:
                print(f"Loaded DOF configs: {len(dof_configs_serialized)} DOFs")

        # Phase 5: Selected MU streaming
        if start_from <= Phase.SELECTED_STREAMING:
            self._prompt_user(
                "PHASE 5: STREAMING WITH SELECTED MOTOR UNITS\n\n"
                f"You will now stream with only the {len(selected_units)} selected motor units.\n"
                "Please prepare to perform the movement again.\n\n"
                "Instructions:\n"
                "  1. Get into position for the movement\n"
                "  2. Press ENTER to start streaming\n"
                "  3. Perform the movement\n"
                "  4. Close the plot window when finished"
            )

            print("\nStarting streaming with selected motor units...")
            print(">>> Close the plot window when you want to stop <<<\n")

            selected_streaming_results = stream_with_decomposition(
                decomp_path=selected_path,
                display_seconds=50,
                enable_filtering=True,
                sil_threshold=0.2,
                max_display_mus=None
            )

            if selected_streaming_results is not None:
                selected_output_path = os.path.join(
                    self.output_folder,
                    f"streaming_results_{self.trial_name}_selected.pkl"
                )
                with open(selected_output_path, "wb") as f:
                    pkl.dump(selected_streaming_results, f)
                print(f"Selected streaming results saved to: {selected_output_path}")

            # Save checkpoint after selected streaming
            self._save_checkpoint(
                Phase.SELECTED_STREAMING,
                recorded_trials=recorded_trials,
                combined_path=combined_path,
                combined_results=combined_results,
                combined_metadata=combined_metadata,
                streaming_results=streaming_results,
                selected_units=selected_units,
                selected_path=selected_path,
                channels_to_remove=channels_to_remove
            )

        # Phase 6: Video game / Cursor control
        if start_from <= Phase.VIDEO_GAME:
            print(f"\n" + "-" * 40)
            print("Game options:")
            print("  1. Video control (MU0=forward, MU1=backward)")
            print("  2. Cursor control (2D target acquisition task)")
            print("  3. Cue game (left/right cue response)")
            print("  4. Skip")
            game_choice = input("Select game mode (1/2/3/4): ").strip()

            if game_choice == '1':
                video_path = input("Enter path to video file: ").strip().strip('"').strip("'")

                if os.path.exists(video_path):
                    self._prompt_user(
                        "PHASE 6: VIDEO GAME\n\n"
                        "Control the video with your motor units!\n"
                        "  - MU 0 firing -> video plays FORWARD\n"
                        "  - MU 1 firing -> video plays BACKWARD\n"
                        "  - Speed is proportional to firing rate (max at 70 Hz)\n\n"
                        "Close the window when finished."
                    )

                    game_results = stream_video_game(
                        decomp_path=selected_path,
                        video_path=video_path,
                        target_fr=30.0,
                        enable_filtering=True
                    )

                    if game_results:
                        print(f"\nGame ended at frame {game_results['final_frame']}/{game_results['total_frames']}")
                else:
                    print(f"Video file not found: {video_path}")

            elif game_choice == '2':
                # Build DOF description for prompt
                dof_description = ""
                if dof_configs_serialized:
                    dof_configs = deserialize_dof_configs(dof_configs_serialized)
                    dof_description = "\n".join([f"  - {cfg.describe()}" for cfg in dof_configs])
                else:
                    dof_description = "  - MU 0 firing -> X position\n  - MU 1 firing -> Y position"
                    dof_configs = None

                self._prompt_user(
                    "PHASE 6: CURSOR CONTROL TASK\n\n"
                    "Control a 2D cursor with your motor units!\n"
                    f"{dof_description}\n"
                    "  - Move cursor to targets and hold to acquire\n\n"
                    "Close the window when finished."
                )

                cursor_results = stream_cursor_control(
                    decomp_path=selected_path,
                    target_fr=30.0,
                    n_targets=10,
                    target_tolerance=0.1,
                    target_hold_time=0.5,
                    enable_filtering=True,
                    dof_configs=dof_configs
                )

                if cursor_results:
                    # Save results (includes EMG data like stream_with_decomposition)
                    cursor_output_path = os.path.join(
                        self.output_folder,
                        f"cursor_control_results_{self.trial_name}.pkl"
                    )
                    with open(cursor_output_path, "wb") as f:
                        pkl.dump(cursor_results, f)

                    print(f"\nTask complete!")
                    print(f"  Targets: {cursor_results['targets_completed']}/{cursor_results['total_targets']}")
                    if cursor_results['total_time']:
                        print(f"  Total time: {cursor_results['total_time']:.1f}s")
                        print(f"  Mean acquisition: {cursor_results['mean_acquisition_time']:.2f}s/target")
                    print(f"  Results saved to: {cursor_output_path}")

            elif game_choice == '3':
                self._prompt_user(
                    "PHASE 6: CUE GAME\n\n"
                    "Respond to left/right cues with your motor units!\n"
                    "  - MU0 dominant -> move RIGHT\n"
                    "  - MU1 dominant -> move LEFT\n"
                    "  - Hold correct direction to score\n\n"
                    "Settings: 50 reps, 20 chunks for FR calculation\n"
                    "Close the window when finished."
                )

                cue_results = stream_cue_game(
                    decomp_path=selected_path,
                    n_reps=50,
                    target_fr=30.0,
                    threshold=0.3,
                    hold_time=0.5,
                    enable_filtering=True,
                    fr_window_size=20
                )

                if cue_results:
                    # Save results
                    cue_output_path = os.path.join(
                        self.output_folder,
                        f"cue_game_results_{self.trial_name}.pkl"
                    )
                    with open(cue_output_path, "wb") as f:
                        pkl.dump(cue_results, f)

                    print(f"\nGame complete!")
                    print(f"  Successful: {cue_results['successful']}/{cue_results['n_reps']}")
                    print(f"  Success rate: {cue_results['success_rate']:.1f}%")
                    print(f"  Total time: {cue_results['total_time']:.1f}s")
                    print(f"  Results saved to: {cue_output_path}")

        print(f"\n" + "=" * 60)
        print("SESSION COMPLETE")
        print("=" * 60)

        return streaming_results





