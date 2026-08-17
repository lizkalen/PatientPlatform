"""
Test script for the interactive EMG decomposition pipeline.

Usage:
    python run_pipeline.py                    # Start from beginning
    python run_pipeline.py --start-from 3     # Start from phase 3 (real-time streaming)
    python run_pipeline.py --list             # List available checkpoint info

Phases:
    1 - RECORDING: Record EMG trials
    2 - DECOMPOSITION: Decompose EMG to extract MUs
    3 - REALTIME_STREAMING: Stream with all MUs
    4 - MU_SELECTION: Select MUs for control
    5 - SELECTED_STREAMING: Stream with selected MUs
    6 - VIDEO_GAME: Video game control

"""

import os
import argparse
from legacy_pipeline.pipeline_manager import PipelineManager, Phase

# Configuration
OUTPUT_FOLDER = r"C:\Users\velar\SynologyDrive\Personal\Thesis Code\output4"
TRIAL_NAME = "test_trial4"
CONFIG_PATH = r"C:\Users\velar\SynologyDrive\Personal\Thesis Code\src\configs\cbss.json"

# Create output folder if it doesn't exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="EMG Decomposition Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Phases:
  1 - RECORDING           Record EMG trials
  2 - DECOMPOSITION       Decompose EMG to extract MUs
  3 - REALTIME_STREAMING  Stream with all MUs
  4 - MU_SELECTION        Select MUs for control
  5 - SELECTED_STREAMING  Stream with selected MUs
  6 - VIDEO_GAME          Video game control
        """
    )
    parser.add_argument(
        "--start-from", "-s",
        type=int,
        choices=[1, 2, 3, 4, 5, 6],
        default=1,
        help="Phase number to start from (default: 1)"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List checkpoint info and exit"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("EMG DECOMPOSITION PIPELINE")
    print("=" * 60)
    print(f"\nOutput folder: {OUTPUT_FOLDER}")
    print(f"Trial name: {TRIAL_NAME}")
    print(f"Config: {CONFIG_PATH}\n")

    pipeline = PipelineManager(
        output_folder=OUTPUT_FOLDER,
        trial_name=TRIAL_NAME,
        config_path=CONFIG_PATH
    )

    if args.list:
        pipeline.list_checkpoints()
    else:
        start_phase = Phase(args.start_from)
        print(f"Starting from phase: {start_phase.name}\n")

        results = pipeline.run(start_from=start_phase)

        if results:
            print("\nPipeline completed successfully!")
        else:
            print("\nPipeline ended without results.")
