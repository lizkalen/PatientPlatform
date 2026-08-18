# PatientGUI

A real-time EMG visualization and patient exercise guidance system with 3D WebGL rendering.

## Overview

PatientGUI consists of two main components:

- **Backend (bend/)**: Python server that streams EMG data via WebSocket
- **Frontend (fend/)**: Three.js-based 3D visualization interface

The system supports two modes:
1. **Live Mode**: Connects to Ripple Trellis hardware for real-time EMG acquisition
2. **Simulation Mode**: Plays back pre-recorded EMG data from `.npz` files (for testing/development)

---

## Prerequisites

Before using PatientGUI, ensure you have:

- **Miniconda or Anaconda** installed (default path: `%USERPROFILE%\miniconda3`)
- **Node.js** (v18 or later recommended)
- **npm** (comes with Node.js)

For live mode only:
- **Ripple Trellis hardware** connected, with the Trellis app open
- **xipppy** library (included in conda environment)

---

## Quick Start (Using Batch Files)

### Option A: Simulation Mode 

Use this mode to test the system without Ripple hardware.

1. **Double-click `start_sim.bat`**

2. On first run, you will be prompted to set up the environment:
   - Press `Y` when asked "Run setup?"
   - The script will:
     - Create the `patientgui` conda environment
     - Install Python dependencies
     - Install frontend npm packages

3. Once setup completes, two windows will open:
   - **Backend Server**: Streams simulated EMG data
   - **Frontend Dev Server**: Serves the web interface

4. **Open your browser** to `http://localhost:8080`

> **Note**: The simulation mode uses a pre-configured data file. To use your own data, edit the path in `start_sim.bat` line 70.

### Option B: Live Mode (Ripple Hardware)

Use this mode with real Ripple Trellis hardware.

1. **Connect your Ripple hardware** and ensure it's powered on

2. **Double-click `start.bat`**

3. Complete first-time setup if prompted (same as simulation mode)

4. **Open your browser** to `http://localhost:8080`

---

## Manual Setup (Without Batch Files)

### Step 1: Create the Conda Environment

```bash
# Navigate to the project directory
cd PatientPlatform

# Create the environment from the YAML file
conda env create -f bend/patientgui.yml

# Activate the environment
conda activate patientgui
```

### Step 2: Install the Backend Package

**Do not skip this.** The backend is imported as `server.*`, so without this
step every launch fails with `No module named server`.

```bash
conda activate patientgui
pip install -e ./bend --no-deps
```

`--no-deps` is deliberate: the environment is owned by `patientgui.yml`, and
letting pip resolve dependencies too makes pip and conda fight over the same
packages (see the comment in `bend/pyproject.toml`).

To confirm it points at *this* checkout — an editable install from another
clone of the repo will silently win otherwise:

```bash
python -c "import server, os; print(os.path.dirname(list(server.__path__)[0]))"
# expect: .../PatientPlatform/bend/src
```

If it prints a different folder, re-run the `pip install -e` above from this
one. The `.bat` launchers perform this check automatically and offer to repair.

### Step 3: Install Frontend Dependencies

```bash
# Navigate to frontend directory
cd fend

# Install npm packages
npm install
```

### Step 4: Start the Backend Server

Open a terminal, activate the conda environment, and run one of the following
from the repository root. These are `python -m` module invocations — they work
from anywhere once Step 2 is done, and they are what the `.bat` files run.

**For Simulation Mode:**
```bash
conda activate patientgui
python -m server.simulate_server_cli "path/to/your/data.npz" --port 8765 --pre-trigger 10
```

No data file? Generate a synthetic one (64 channels, 60 s, 2000 Hz, ~30 MB):

```bash
python -m server.make_sim_data sim/datafile1_filtered.npz
```

That file is band-limited noise with burst envelopes — enough to exercise the
streaming, filtering, recording and plotting paths, and **not** physiological
data. Never use it to validate an algorithm, a threshold or a model.

**For Live Mode (Ripple Hardware):**
```bash
conda activate patientgui
python -m server.server_cli --port 8765 --pre-trigger 10
```

**For Quattrocento / OTBioLab+:**
```bash
conda activate patientgui
python -m server.quattrocento_server_cli --config "path/to/config.otb+stp" --port 8765
```

### Step 5: Start the Frontend

Open a separate terminal:

```bash
cd fend
npm run dev
```

### Step 6: Access the Application

Open your browser to `http://localhost:8080`

---

## Backend Server Options

### Simulation Server (`simulate_server_cli.py`)

| Option | Default | Description |
|--------|---------|-------------|
| `npz_path` | (required) | Path to `.npz` file with EMG data |
| `--host`, `-h` | localhost | Server host address |
| `--port`, `-p` | 8765 | WebSocket server port |
| `--srate`, `-r` | 2000.0 | Sample rate in Hz |
| `--loop`, `-l` | False | Loop data continuously when simulating |
| `--pre-trigger` | 10.0 | Seconds of pre-trigger data for recordings |
| `--output`, `-o` | ./recordings | Output folder for recordings |
| `--no-filter` | False | Disable bandpass/notch filtering |
| `--no-lsl` | False | Disable LSL outlet |
| `--chunk-interval` | 50 | Milliseconds between data chunks |
| `--lowcut` | 20.0 | Bandpass low cutoff (Hz) |
| `--highcut` | 500.0 | Bandpass high cutoff (Hz) |
| `--notch` | 50.0 | Notch filter frequency (Hz) |

### Live Server (`server_cli.py`)

| Option | Default | Description |
|--------|---------|-------------|
| `--host`, `-h` | localhost | Server host address |
| `--port`, `-p` | 8765 | WebSocket server port |
| `--stream-type`, `-s` | hi-res | Ripple stream type: hi-res, raw, or lfp |
| `--pre-trigger` | 10.0 | Seconds of pre-trigger data for recordings |
| `--output`, `-o` | ./recordings | Output folder for recordings |
| `--no-filter` | False | Disable bandpass/notch filtering |
| `--no-lsl` | False | Disable LSL outlet |
| `--chunk-interval` | 50 | Milliseconds between data chunks |
| `--lowcut` | 20.0 | Bandpass low cutoff (Hz) |
| `--highcut` | 500.0 | Bandpass high cutoff (Hz) |
| `--notch` | 50.0 | Notch filter frequency (Hz) |

---

## Using the Application

### Interface Modes

The application has two interface modes:

1. **Config Mode**: For configuring sequences and settings
2. **Patient Mode**: Simplified view for patient exercises with tutorial overlays

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `H` | Reset camera to home position |
| `G` or `P` | Toggle GUI panel |

### Features

- **3D Visualization**: Real-time WebGL rendering with Three.js
- **EMG Channel View**: Live visualization of EMG channels
- **Sequence Player**: Play pre-defined exercise sequences
- **Timeline View**: Navigate through recordings
- **Phase Overlay**: Visual cues during exercises
- **Tutorial Overlay**: Guidance for patients before each exercise
- **Progress Sidebar**: Track exercise completion in patient mode

---

## Preparing Your Own Data

To use your own EMG data in simulation mode:

1. Save your data as a NumPy `.npz` file with the EMG array under the key
   `data`, shaped `(channels, samples)` — that is what the simulated device
   reads
2. Put it at `sim/datafile1_filtered.npz` (what `start_sim.bat` expects) or
   pass it explicitly:
   ```bash
   python -m server.simulate_server_cli "C:\path\to\your\data.npz" --srate 2000
   ```

To generate a synthetic file in that format instead, see
`python -m server.make_sim_data --help`.

---

## Troubleshooting

### Conda not found
- Ensure Miniconda/Anaconda is installed
- The launchers probe `%USERPROFILE%` and `%LOCALAPPDATA%` for `miniconda3` and
  `anaconda3`, then `%PROGRAMDATA%\miniconda3`
- If yours is somewhere else, set `CONDA_OVERRIDE` near the top of the `.bat`
  to the full path of its `Scripts\activate.bat`

### "No module named server"
The backend package was never installed into the environment. Run
`pip install -e ./bend --no-deps` with `patientgui` active (Manual Setup,
Step 2). If it is installed but the error persists, an editable install from a
*different* clone of this repo is winning — the launchers detect this and offer
to repair it; see Step 2 for the manual check.

### Port 8765 already in use
A backend from an earlier run is still alive. The frontend would silently
connect to it instead of the new one. Close the old **Backend Server** window,
or find the owner with `netstat -ano | findstr :8765`.

### WebSocket connection failed
- Verify the backend server is running
- Check that port 8765 is not in use by another application
- Ensure frontend and backend are using the same port

### Ripple hardware not detected (Live Mode)
- Verify hardware connections
- Ensure Ripple software/drivers are installed
- Check that xipppy can detect the device

### Frontend build errors
- Delete `fend/node_modules` and run `npm install` again
- Ensure you have Node.js v18 or later

---

## Project Structure

```
PatientGUI/
├── start.bat              # Quick start - Live mode
├── start_sim.bat          # Quick start - Simulation mode
├── bend/                  # Backend (Python)
│   ├── patientgui.yml     # Conda environment specification
│   └── src/
│       └── server/
│           ├── server_cli.py           # Live server CLI
│           └── simulate_server_cli.py  # Simulation server CLI
└── fend/                  # Frontend (JavaScript)
    ├── package.json       # npm dependencies
    └── src/
        ├── App.js         # Main application
        ├── emg/           # EMG client and visualization
        ├── webgl/         # Three.js WebGL rendering
        ├── ui/            # UI components
        └── gui/           # Tweakpane GUI
```

---

## License

[Add license information here, if we ever publish this for some reason]
