# WebSocket Decomposition Streaming

Real-time motor unit (MU) decomposition and classification over WebSocket.

## Table of Contents

- [WebSocket Decomposition Streaming](#websocket-decomposition-streaming)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Architecture](#architecture)
  - [Backend Files](#backend-files)
  - [Signal Processing Pipeline](#signal-processing-pipeline)
    - [Per-Chunk Processing (`DecompositionManager.process()`)](#per-chunk-processing-decompositionmanagerprocess)
    - [Key Concepts](#key-concepts)
  - [WebSocket Protocol Reference](#websocket-protocol-reference)
    - [Connection](#connection)
    - [Decomposition Commands (Client → Server)](#decomposition-commands-client--server)
      - [`load_model`](#load_model)
      - [`unload_model`](#unload_model)
    - [Decomposition Messages (Server → Client)](#decomposition-messages-server--client)
      - [`decomposition_status`](#decomposition_status)
      - [`decomposition_chunk`](#decomposition_chunk)
    - [Classification Commands](#classification-commands)
      - [`start_classification`](#start_classification)
      - [`stop_classification`](#stop_classification)
      - [`classification_status`](#classification_status)
    - [Existing Commands (unchanged)](#existing-commands-unchanged)
  - [Frontend API (EMGClient.js)](#frontend-api-emgclientjs)
    - [Methods](#methods)
      - [Decomposition](#decomposition)
      - [Classification](#classification)
      - [State Properties](#state-properties)
    - [Callbacks](#callbacks)
    - [Data Structures](#data-structures)
      - [Decomposition Data Object](#decomposition-data-object)
  - [Frontend Integration Guide](#frontend-integration-guide)
    - [Displaying Source Signals](#displaying-source-signals)
    - [Displaying Spike Rasters](#displaying-spike-rasters)
    - [Displaying Firing Rates \& SIL Scores](#displaying-firing-rates--sil-scores)
    - [Classification-Driven UI](#classification-driven-ui)
    - [Buffering Strategy](#buffering-strategy)
  - [Model File Format](#model-file-format)
  - [Testing](#testing)
    - [Test with Simulated Server](#test-with-simulated-server)
    - [Test from Browser Console](#test-from-browser-console)
    - [Test with websocat](#test-with-websocat)
  - [Performance Considerations](#performance-considerations)
    - [Timing Budget](#timing-budget)
    - [Bandwidth](#bandwidth)
    - [Threading](#threading)
    - [Scaling](#scaling)
  - [Glossary](#glossary)

---

## Overview

The WebSocket server (`RippleWebSocketServer` / `SimulatedWebSocketServer`) streams raw or filtered EMG data to the frontend. The **decomposition extension** adds the ability to:

1. **Load a pretrained decomposition model** at runtime (via frontend command)
2. **Decompose each EMG chunk** into individual motor unit source signals and spike trains
3. **Stream the results** (`decomposition_chunk` messages) alongside the existing `emg_chunk` messages
4. **Classify movements** from pairs of MU firing rates (optional)

All decomposition logic is encapsulated in `DecompositionManager` — the server simply delegates to it and broadcasts results.

---

## Architecture

```
┌─────────────────────┐      ┌──────────────────────────────┐      ┌──────────────┐
│  Ripple Hardware    │      │  WebSocket Server             │      │  Frontend    │
│  (or SimulatedDevice)├─────►│                              │      │              │
│                     │ fetch│  ┌────────────────────────┐   │  WS  │  EMGClient   │
│  EMG (64ch, 2kHz)   │──────►  │  Filter (bandpass+notch)│  ├──────►             │
└─────────────────────┘      │  └──────────┬─────────────┘   │      │  emg_chunk   │
                             │             │                 │      │              │
                             │             ▼                 │      │              │
                             │  ┌────────────────────────┐   │      │              │
                             │  │  DecompositionManager   │  ├──────►             │
                             │  │                        │   │      │  decomp_chunk│
                             │  │  channel select        │   │      │              │
                             │  │  → extension (R=16)    │   │      │  sources     │
                             │  │  → whitening (Z)       │   │      │  spikes      │
                             │  │  → separation (B^T)    │   │      │  sil_scores  │
                             │  │  → spike detection     │   │      │  firing_rates│
                             │  │  → firing rates        │   │      │  classif.    │
                             │  │  → classification      │   │      │              │
                             │  └────────────────────────┘   │      └──────────────┘
                             └──────────────────────────────┘
```

**Data flow per chunk (~50ms):**

1. Server fetches raw EMG from device → `(n_channels, n_samples)`
2. Bandpass + notch filtering (if enabled)
3. Broadcast `emg_chunk` to all clients
4. If a decomposition model is loaded:
   - `DecompositionManager.process(samples)` runs in a background thread (`asyncio.to_thread`)
   - Result is serialized to JSON and broadcast as `decomposition_chunk`
5. Both messages arrive at the frontend nearly simultaneously

---

## Backend Files

| File | Purpose |
|------|---------|
| `bend/src/server/decomposition_manager.py` | `DecompositionManager` class — model loading, chunk processing, classification, JSON serialization |
| `bend/src/server/websocket_server.py` | `RippleWebSocketServer` — hardware server, integrates `DecompositionManager` |
| `bend/src/server/simulated_websocket_server.py` | `SimulatedWebSocketServer` — playback server for testing, same decomposition integration |
| `bend/src/online/processing.py` | `process_chunk()`, `calculate_firing_rates()`, `design_filters()`, `apply_filters()` |
| `bend/src/online/loading.py` | `load_pretrained_model()` — loads `.pkl` model files |
| `bend/src/muniverse/algorithms/core.py` | `est_spike_times()`, `extension()` — spike detection and signal extension |
| `fend/src/emg/EMGClient.js` | Frontend WebSocket client — handles all message types, exposes callbacks |

---

## Signal Processing Pipeline

Understanding the decomposition pipeline is essential for interpreting the data on the frontend.

### Per-Chunk Processing (`DecompositionManager.process()`)

```
Input: samples (n_channels × n_samples), e.g. (64 × 100) at 2 kHz / 50 ms

1. Channel Selection
   └─ Keep only channels_to_keep (exclude channels removed during calibration)
   └─ Result: (n_keep × n_samples), e.g. (60 × 100)

2. Tail Prepend (chunk continuity)
   └─ Prepend last R-1 = 15 samples from previous chunk
   └─ Result: (60 × 115)

3. Toeplitz Extension (R = 16 lags)
   └─ extension(samples, R=16)
   └─ Result: (n_keep × 16, n_extended) = (960 × 115)

4. Mean Centering
   └─ Subtract row means

5. Whitening
   └─ Z @ extended_signal
   └─ Z is (960 × 960), precomputed offline

6. Source Separation
   └─ B^T @ whitened_signal
   └─ B is (960 × n_mus), precomputed offline
   └─ Result: sources (n_mus × 115)

7. Normalization (optional, Farina 2025)
   └─ If norm_factors available: (source - offset) / scale, clipped to [0, 1]

8. Trim Overlap
   └─ Remove the first 15 prepended samples
   └─ Rebase spike indices to align with the chunk's n_samples
   └─ Result: sources (n_mus × 100)

9. Spike Detection (per MU)
   └─ est_spike_times(source, fs, cluster="centroid", centroids=...)
   └─ Asymmetric power law → peak detection → centroid-based clustering
   └─ Returns: spike indices, SIL score [0, 1], updated centroids

10. Firing Rates
    └─ Rolling window of 5 chunks (~250 ms)
    └─ spike_count / window_duration → Hz
```

### Key Concepts

- **Sources**: The separated motor unit pulse trains. Each row is one MU's activity over time. Values represent the instantaneous activation level.
- **Spikes**: Discrete firing events detected from sources. Array of sample indices within the chunk.
- **SIL Score** (Silhouette Index): Quality metric [0, 1] for each MU's spike detection. Higher = more reliable separation between spike and noise clusters. Typical threshold: ≥ 0.75.
- **Firing Rate**: Spikess per second (Hz), averaged over a rolling window of 5 chunks.
- **Centroids**: Per-MU `[noise_centroid, spike_centroid]` values used for real-time spike classification. Updated every chunk.

---

## WebSocket Protocol Reference

All messages are JSON. The server broadcasts to **all connected clients** unless specified otherwise.

### Connection

When a client connects, the server sends:

```jsonc
{
  "type": "connected",
  "sample_rate": 2000,
  "n_channels": 64,
  "pre_trigger_seconds": 10.0,
  "stream_type": "hi-res",         // or "simulated"
  "filtering_enabled": true,
  "lsl_enabled": true,

  // Decomposition status fields (always present)
  "decomposition_active": false,
  "n_mus": 0,
  "model_path": null,
  "channels_to_remove": [],
  "total_samples_processed": 0,
  "classification_active": false
}
```

### Decomposition Commands (Client → Server)

#### `load_model`

Load a pretrained decomposition model. Decomposition starts immediately on the next chunk.

```jsonc
{
  "command": "load_model",
  "model_path": "C:/path/to/results_combined.pkl"
}
```

**On success** → broadcasts `decomposition_status` to all clients.
**On failure** → sends `error` to the requesting client only.

> **Note:** `model_path` is a filesystem path on the **server machine**. The model `.pkl` file must be accessible from where the server process runs.

#### `unload_model`

Stop decomposition and unload the model.

```jsonc
{
  "command": "unload_model"
}
```

**Response** → broadcasts `decomposition_status` with `active: false`.

### Decomposition Messages (Server → Client)

#### `decomposition_status`

Sent when a model is loaded or unloaded. Broadcast to all clients.

```jsonc
// Model loaded
{
  "type": "decomposition_status",
  "active": true,
  "n_mus": 5,
  "model_path": "C:/path/to/results_combined.pkl",
  "channels_to_remove": [12, 45],
  "channels_to_keep": [0, 1, 2, /* ... */, 63],   // excluding 12, 45
  "has_centroids": true,
  "has_norm_factors": false
}

// Model unloaded
{
  "type": "decomposition_status",
  "active": false,
  "n_mus": 0
}
```

#### `decomposition_chunk`

Sent every ~50 ms (same rate as `emg_chunk`) while a model is loaded.

```jsonc
{
  "type": "decomposition_chunk",

  // Source signals: n_mus × n_samples (same n_samples as corresponding emg_chunk)
  "sources": [
    [0.12, 0.05, 0.34, /* ... */],    // MU 0
    [0.01, 0.89, 0.02, /* ... */],    // MU 1
    // ... one row per MU
  ],

  // Spike indices within this chunk (0-based sample indices)
  "spikes": {
    "0": [23, 67],      // MU 0 fired at samples 23 and 67
    "1": [45],           // MU 1 fired at sample 45
    "2": [],             // MU 2 did not fire
    "3": [12, 56, 89]   // MU 3 fired 3 times
  },

  // Silhouette scores per MU (quality of spike detection, 0-1)
  "sil_scores": [0.85, 0.72, 0.91, 0.68],

  // Firing rates per MU (Hz, rolling average over ~250 ms)
  "firing_rates": [12.5, 8.0, 15.3, 6.2],

  // Number of new samples in this chunk
  "n_samples": 100,

  // Cumulative samples processed since model load
  "total_samples": 50000,

  // Timestamp (same as corresponding emg_chunk)
  "timestamp": 1234567890.123,

  // Classification (only present if classification is active)
  "classification": {
    "label": 1,                       // 0=rest, 1=movement1, 2=movement2
    "mu1_firing_rate": 12.5,          // Hz
    "mu2_firing_rate": 15.3,          // Hz
    "transition": "rest_to_active"    // or "active_to_rest" or null
  }
}
```

### Classification Commands

#### `start_classification`

Requires a model to be loaded first.

```jsonc
{
  "command": "start_classification",
  "mu1_idx": 0,          // optional, default 0
  "mu2_idx": 1,          // optional, default 1
  "threshold": 10.0,     // optional, default 10.0 Hz
  "window_sec": 1.0      // optional, default 1.0 s
}
```

**Classification labels:**

| Label | Condition | Meaning |
|-------|-----------|---------|
| `0` | Neither MU above threshold | Rest |
| `1` | Both MU1 **and** MU2 above threshold | Movement type 1 |
| `2` | Only MU2 above threshold | Movement type 2 |

**Response** → broadcasts `classification_status`.

#### `stop_classification`

```jsonc
{
  "command": "stop_classification"
}
```

**Response** → broadcasts `classification_status` with `active: false`.

#### `classification_status`

```jsonc
{
  "type": "classification_status",
  "active": true,
  "mu1_idx": 0,
  "mu2_idx": 1,
  "threshold": 10.0,
  "window_sec": 1.0
}
```

### Existing Commands (unchanged)

These commands remain exactly as before:

| Command | Direction | Description |
|---------|-----------|-------------|
| `start_recording` | Client → Server | Start recording with optional metadata |
| `stop_recording` | Client → Server | Stop recording with optional timeline |
| `get_status` | Client → Server | Query server status (now includes decomposition state) |
| `emg_chunk` | Server → Client | Raw/filtered EMG chunk (unchanged) |
| `recording_status` | Server → Client | Recording started/stopped notification |
| `recording_saved` | Server → Client | Recording file saved notification |
| `error` | Server → Client | Error message |

The `get_status` response now includes all decomposition/classification fields.

---

## Frontend API (EMGClient.js)

### Methods

#### Decomposition

```javascript
// Load a decomposition model (server-side path)
emgClient.loadModel('/path/to/results_combined.pkl');

// Unload the current model (stops decomposition)
emgClient.unloadModel();
```

#### Classification

```javascript
// Start classification with custom config
emgClient.startClassification({
  mu1Idx: 0,          // default: 0
  mu2Idx: 1,          // default: 1
  threshold: 10.0,    // default: 10.0 Hz
  windowSec: 1.0,     // default: 1.0 s
});

// Stop classification
emgClient.stopClassification();
```

#### State Properties

```javascript
emgClient.decompositionActive  // boolean — is a model loaded?
emgClient.nMUs                 // number  — how many motor units?
emgClient.classificationActive // boolean — is classification running?
```

### Callbacks

```javascript
// Fires when a model is loaded or unloaded
emgClient.onDecompositionStatus((status) => {
  // status.active: boolean
  // status.nMUs: number
  // status.modelPath: string
  // status.channelsToRemove: number[]
  // status.channelsToKeep: number[]
  // status.hasCentroids: boolean
  // status.hasNormFactors: boolean
});

// Fires every ~50 ms with decomposition results
emgClient.onDecompositionData((data) => {
  // data.sources: number[][]     — [n_mus][n_samples]
  // data.spikes: object          — { "0": [23, 67], "1": [45], ... }
  // data.silScores: number[]     — per-MU SIL scores
  // data.firingRates: number[]   — per-MU firing rates (Hz)
  // data.nSamples: number        — samples in this chunk
  // data.totalSamples: number    — cumulative since model load
  // data.timestamp: number       — server timestamp
});

// Fires when classification is started or stopped
emgClient.onClassificationStatus((status) => {
  // status.active: boolean
  // status.mu1Idx: number
  // status.mu2Idx: number
  // status.threshold: number
  // status.windowSec: number
});

// Fires every chunk while classification is active
emgClient.onClassification((result) => {
  // result.label: number            — 0 (rest), 1 (move1), 2 (move2)
  // result.mu1FiringRate: number    — Hz
  // result.mu2FiringRate: number    — Hz
  // result.transition: string|null  — "rest_to_active" | "active_to_rest" | null
  // result.timestamp: number
});
```

### Data Structures

#### Decomposition Data Object

```javascript
{
  sources: Float64Array[][] // [muIdx][sampleIdx] — source signals
  spikes: {                 // spike indices within this chunk
    "0": [23, 67],
    "1": [45],
    // ...
  },
  silScores: [0.85, 0.72, ...],   // per-MU quality [0, 1]
  firingRates: [12.5, 8.0, ...],  // per-MU Hz
  nSamples: 100,                  // samples in this chunk
  totalSamples: 50000,            // cumulative
  timestamp: 1234567890.123
}
```

> **Note:** `sources` arrive as plain JS arrays (from JSON). If you need typed arrays for canvas rendering, convert on arrival:
> ```javascript
> const sourceBuffers = data.sources.map(row => new Float32Array(row));
> ```

---

## Frontend Integration Guide

This section provides concrete guidance for building visualization components.

### Displaying Source Signals

Source signals are continuous waveforms, one per MU. Display them like EMG channels but with fewer traces (typically 3–8 MUs vs 64 EMG channels).

**Recommended approach:**

```javascript
class DecompositionView {
  constructor(emgClient, options = {}) {
    this.bufferSize = options.bufferSize || 4096; // ~2s at 2kHz
    this.silThreshold = options.silThreshold || 0.75;
    this.sourceBuffers = []; // Float32Array per MU
    this.spikeHistory = [];  // per-MU array of buffer-relative indices

    emgClient.onDecompositionStatus((status) => {
      if (status.active) {
        this._initBuffers(status.nMUs);
      }
    });

    emgClient.onDecompositionData((data) => {
      this._updateBuffers(data);
      this._draw();
    });
  }

  _initBuffers(nMUs) {
    this.sourceBuffers = [];
    this.spikeHistory = [];
    for (let i = 0; i < nMUs; i++) {
      this.sourceBuffers.push(new Float32Array(this.bufferSize));
      this.spikeHistory.push([]);
    }
  }

  _updateBuffers(data) {
    const nNew = data.nSamples;

    for (let mu = 0; mu < data.sources.length; mu++) {
      const buf = this.sourceBuffers[mu];

      // Shift left and append new samples (same pattern as EMGClient)
      buf.copyWithin(0, nNew);
      const src = data.sources[mu];
      for (let i = 0; i < nNew; i++) {
        buf[this.bufferSize - nNew + i] = src[i];
      }

      // Update spike history: shift existing by nNew, add new spikes
      this.spikeHistory[mu] = this.spikeHistory[mu]
        .map(t => t - nNew)
        .filter(t => t >= 0);

      // Only add spikes if SIL score meets threshold
      if (data.silScores[mu] >= this.silThreshold) {
        const newSpikes = data.spikes[String(mu)] || [];
        for (const idx of newSpikes) {
          this.spikeHistory[mu].push(this.bufferSize - nNew + idx);
        }
      }
    }
  }
}
```

### Displaying Spike Rasters

Spikes are best displayed as a raster plot: vertical tick marks on a horizontal time axis, one row per MU.

**Key considerations:**

- Spike indices in `data.spikes` are **relative to the current chunk** (0 to `n_samples - 1`)
- Convert to buffer position: `bufferPos = bufferSize - nSamples + spikeIndex`
- Filter by SIL score before displaying — MUs with `sil_scores[mu] < 0.75` produce unreliable spikes
- Color-code by MU index for visual distinction

```javascript
// In your canvas draw loop:
for (let mu = 0; mu < nMUs; mu++) {
  const y = mu * rowHeight + rowHeight / 2;
  const spikes = this.spikeHistory[mu];

  ctx.strokeStyle = MU_COLORS[mu % MU_COLORS.length];
  ctx.lineWidth = 2;

  for (const bufIdx of spikes) {
    const x = (bufIdx / this.bufferSize) * canvasWidth;
    ctx.beginPath();
    ctx.moveTo(x, y - tickHeight / 2);
    ctx.lineTo(x, y + tickHeight / 2);
    ctx.stroke();
  }
}
```

### Displaying Firing Rates & SIL Scores

Firing rates and SIL scores are **per-MU scalars** that update every chunk (~50 ms). Display as:

- **Text labels**: `"MU0: 12.5 Hz | SIL: 0.85"` next to each source/raster row
- **Bar chart**: horizontal bars per MU, length proportional to firing rate
- **Color-coded**: green if SIL ≥ threshold, gray/red if below

```javascript
emgClient.onDecompositionData((data) => {
  for (let mu = 0; mu < data.firingRates.length; mu++) {
    const fr = data.firingRates[mu].toFixed(1);
    const sil = data.silScores[mu].toFixed(2);
    const reliable = data.silScores[mu] >= 0.75;

    updateLabel(mu, `${fr} Hz | SIL: ${sil}${reliable ? '' : ' (low)'}`);
    updateBar(mu, data.firingRates[mu], reliable ? '#00ff88' : '#666');
  }
});
```

### Classification-Driven UI

When classification is active, each chunk includes a `label` (0, 1, or 2). Use this to drive UI elements:

```javascript
emgClient.onClassification((result) => {
  switch (result.label) {
    case 0:
      setIndicator('REST', '#888');
      break;
    case 1:
      setIndicator('MOVEMENT 1', '#00ff88');
      break;
    case 2:
      setIndicator('MOVEMENT 2', '#ff8800');
      break;
  }

  // React to transitions for animations/sounds
  if (result.transition === 'rest_to_active') {
    playActivationSound();
  }
});
```

### Buffering Strategy

| Data Type | Buffer Approach | Suggested Size |
|-----------|----------------|----------------|
| Source signals | Circular `Float32Array` per MU (shift-left + append) | 4096 samples (~2s) |
| Spike history | Array of buffer-relative indices, shift by `nNew` each chunk | Keep last ~2s worth |
| Firing rates | Overwrite each chunk (already rolling-averaged server-side) | Single value per MU |
| SIL scores | Overwrite each chunk | Single value per MU |
| Classification | Overwrite each chunk | Single label |

---

## Model File Format

The model `.pkl` file is produced by the offline decomposition pipeline (`run_cli_pipeline.py` → decomposition phase). It contains a tuple `(results, metadata)`:

```python
results = {
    "mu_filters": np.ndarray,      # (n_extended_channels, n_mus) — separation matrix B
    "Z": np.ndarray,                # (n_ext_ch, n_ext_ch) — whitening matrix
    "centroids": {                  # per-MU spike/noise centroids (optional)
        0: [noise_centroid, spike_centroid],
        1: [noise_centroid, spike_centroid],
        # ...
    },
    "norm_factors": {               # per-MU normalization (optional, Farina 2025)
        0: (offset, scale),
        1: (offset, scale),
        # ...
    },
    "channels_to_remove": [12, 45],  # channel indices excluded during calibration
}

metadata = {
    "srate": 2000,
    # ... other trial metadata
}
```

The model is typically found at paths like:
```
output/results_{trial_name}_combined.pkl
```

---

## Testing

### Test with Simulated Server

No hardware required. Uses `.npz` playback.

**Terminal 1 — Start the simulated server:**
```bash
cd bend
python src/server/simulate_server_cli.py ../datafile1_filtered.npz --loop --no-lsl --port 8765
```

**Terminal 2 — Run the test client:**
```bash
cd bend
python test_decomp_ws.py --model "path/to/results_combined.pkl" --duration 30
```

With classification:
```bash
python test_decomp_ws.py --model "path/to/results_combined.pkl" --classify --duration 30
```

### Test from Browser Console

With the frontend running (`npm run dev` in `fend/`):

```javascript
// Access the EMG client
const client = app.emgClient;

// Load model
client.loadModel('C:/path/to/results_combined.pkl');

// Listen for decomposition data
client.onDecompositionData((data) => {
  console.log('Firing rates:', data.firingRates);
  console.log('Spikes:', data.spikes);
});

// Listen for classification
client.onClassification((r) => {
  console.log('Label:', r.label, 'FR1:', r.mu1FiringRate, 'FR2:', r.mu2FiringRate);
});

// Start classification
client.startClassification({ mu1Idx: 0, mu2Idx: 1, threshold: 10 });

// Clean up
client.stopClassification();
client.unloadModel();
```

### Test with websocat

```bash
# Connect and send commands interactively
websocat ws://localhost:8765

# Then type JSON commands:
{"command": "load_model", "model_path": "C:/path/to/model.pkl"}
{"command": "start_classification", "mu1_idx": 0, "mu2_idx": 1}
{"command": "get_status"}
{"command": "stop_classification"}
{"command": "unload_model"}
```

---

## Performance Considerations

### Timing Budget

At 50 ms chunk interval with 2 kHz sample rate (100 samples/chunk):

| Step | Typical Duration |
|------|-----------------|
| Extension + whitening + separation | ~1–3 ms |
| Spike detection (per MU, centroid mode) | ~0.5–1 ms |
| Total for 5 MUs | ~4–8 ms |
| JSON serialization (5 MUs × 100 samples) | ~1–2 ms |
| **Total** | **~5–10 ms** (well within 50 ms budget) |

### Bandwidth

Per `decomposition_chunk` message (5 MUs, 100 samples):
- `sources`: 5 × 100 × ~8 chars = ~4 KB
- `spikes` + `sil_scores` + `firing_rates`: ~200 bytes
- **Total**: ~4–5 KB per chunk → ~80–100 KB/s

This is comparable to the existing `emg_chunk` bandwidth (64 channels × 100 samples ≈ 50 KB/chunk).

### Threading

`DecompositionManager.process()` runs in `asyncio.to_thread()` so the event loop is not blocked during CPU-intensive spike detection. This means:
- EMG chunks continue being broadcast on time
- Decomposition chunks may arrive slightly after their paired EMG chunk
- The `timestamp` field is shared, so the frontend can correlate them

### Scaling

- **More MUs**: Processing time scales linearly with `n_mus`. Up to ~15 MUs should fit comfortably in the 50 ms budget.
- **Higher sample rates**: Extension and matrix multiplication scale with `n_samples`. At 30 kHz, consider increasing `chunk_interval_ms` or running decomposition on every Nth chunk.

---

## Glossary

| Term | Definition |
|------|-----------|
| **MU** | Motor Unit — the smallest functional unit of voluntary movement (a motor neuron + the muscle fibers it innervates) |
| **Source signal** | The separated activation trace for a single MU, extracted from multi-channel EMG |
| **Spike** | A single firing event of a motor unit, detected as a peak in the source signal |
| **SIL score** | Silhouette-like quality index [0, 1] for spike detection. Measures cluster separation between spike and noise peaks |
| **Firing rate** | Number of spikes per second (Hz), averaged over a rolling window |
| **Whitening matrix (Z)** | Decorrelates and normalizes the extended EMG signal. Precomputed offline |
| **Separation matrix (B)** | Motor unit filters that project whitened signal into individual source signals. Precomputed offline |
| **Extension** | Toeplitz time-delay embedding — each channel is expanded into R=16 time-lagged copies to capture temporal structure |
| **Centroids** | Per-MU `[noise, spike]` cluster centers used for real-time spike classification (avoids re-fitting k-means each chunk) |
| **Norm factors** | Per-MU `(offset, scale)` values to constrain source signals to `[0, 1]` range (Farina 2025 method) |
