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
cd PatientGUI

# Create the environment from the YAML file
conda env create -f bend/patientgui.yml

# Activate the environment
conda activate patientgui
```

### Step 2: Install Frontend Dependencies

```bash
# Navigate to frontend directory
cd fend

# Install npm packages
npm install
```

### Step 3: Start the Backend Server

Open a terminal, activate the conda environment, and run one of the following:

**For Simulation Mode:**
```bash
conda activate patientgui
cd bend
python src/server/simulate_server_cli.py "path/to/your/data.npz" --port 8765 --pre-trigger 10
```

**For Live Mode (Ripple Hardware):**
```bash
conda activate patientgui
cd bend
python src/server/server_cli.py --port 8765 --pre-trigger 10
```

### Step 4: Start the Frontend

Open a separate terminal:

```bash
cd fend
npm run dev
```

### Step 5: Access the Application

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

1. Save your data as a NumPy `.npz` file with the EMG array
2. Update the path in `start_sim.bat` or provide it as an argument:
   ```bash
   python src/server/simulate_server_cli.py "C:\path\to\your\data.npz" --srate 2000
   ```

---

## Troubleshooting

### Conda not found
- Ensure Miniconda/Anaconda is installed
- If installed in a non-default location, edit the `CONDA_PATH` in `start.bat` or `start_sim.bat`

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
