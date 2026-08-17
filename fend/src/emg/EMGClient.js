/**
 * EMG WebSocket Client
 * ====================
 *
 * Connects to the EMG WebSocket server and handles:
 * - Receiving EMG data chunks
 * - Sending start/stop recording commands
 * - Connection state management
 */
export default class EMGClient {
	constructor(options = {}) {
		this.host = options.host || 'localhost';
		this.port = options.port || 8765;
		this.autoReconnect = options.autoReconnect !== false;
		this.reconnectInterval = options.reconnectInterval || 3000;

		// Connection state
		this.ws = null;
		this.isConnected = false;
		this.isRecording = false;

		// Server info (received on connect)
		this.sampleRate = null;
		this.nChannels = null;
		this.preTriggerSeconds = null;

		// Decomposition state
		this.decompositionActive = false;
		this.nMUs = 0;
		this.classificationActive = false;

		// Movement-classification state (independent of decomposition)
		this.movementActive = false;
		this.movementClassOrder = ['close', 'trp', 'ext', 'rest'];
		this.movementStimBinary = false;   // STIM head reporting a move/rest gate (vs 4-class)

		// Whether the server streams live EMG chunks for plotting. Disabled for
		// high-channel devices (Quattrocento) where serialization stalls the
		// server; recording is unaffected. Defaults true for older servers.
		this.liveEmgEnabled = true;

		// Data buffers - store recent samples for visualization
		this.channelBuffers = [];
		this.bufferSize = 2048; // ~1 second at 2048 Hz

		// Callbacks
		this.onConnectCallbacks = [];
		this.onDisconnectCallbacks = [];
		this.onDataCallbacks = [];
		this.onRecordingStatusCallbacks = [];
		this.onRecordingSavedCallbacks = [];
		this.onErrorCallbacks = [];
		this.onDecompositionStatusCallbacks = [];
		this.onDecompositionDataCallbacks = [];
		this.onClassificationStatusCallbacks = [];
		this.onClassificationCallbacks = [];
		this.onConfigStatusCallbacks = [];
		this.onMovementStatusCallbacks = [];
		this.onMovementDecisionCallbacks = [];
		this.onMvTrainStatusCallbacks = [];
		this.onTriggerChunkCallbacks = [];
		this.onStimulationStatusCallbacks = [];
		this.onDeviceStatusCallbacks = [];
		this.onStreamEndedCallbacks = [];

		// Reconnection timer
		this._reconnectTimer = null;
	}

	/**
	 * Get the WebSocket URL
	 */
	get url() {
		return `ws://${this.host}:${this.port}`;
	}

	/**
	 * Connect to the EMG server
	 */
	connect() {
		if (this.ws && this.ws.readyState === WebSocket.OPEN) {
			console.log('EMGClient: Already connected');
			return;
		}

		console.log(`EMGClient: Connecting to ${this.url}...`);

		try {
			this.ws = new WebSocket(this.url);

			this.ws.onopen = () => {
				console.log('EMGClient: Connected');
				this.isConnected = true;
				this._clearReconnectTimer();
			};

			this.ws.onmessage = (event) => {
				this._handleMessage(event.data);
			};

			this.ws.onclose = () => {
				console.log('EMGClient: Disconnected');
				this.isConnected = false;
				this._notifyDisconnect();

				if (this.autoReconnect) {
					this._scheduleReconnect();
				}
			};

			this.ws.onerror = (error) => {
				console.error('EMGClient: WebSocket error', error);
				this._notifyError('WebSocket connection error');
			};
		} catch (error) {
			console.error('EMGClient: Failed to connect', error);
			this._notifyError(`Failed to connect: ${error.message}`);

			if (this.autoReconnect) {
				this._scheduleReconnect();
			}
		}
	}

	/**
	 * Disconnect from the EMG server
	 */
	disconnect() {
		this.autoReconnect = false;
		this._clearReconnectTimer();

		if (this.ws) {
			this.ws.close();
			this.ws = null;
		}
	}

	/**
	 * Send start recording command with optional metadata
	 * @param {object} metadata - Optional trial metadata to include with recording
	 *
	 * Metadata structure (all fields optional):
	 * {
	 *   subjectId: string,      // Subject/participant identifier
	 *   sessionId: string,      // Session identifier
	 *   notes: string,          // Free-form notes
	 *   sequenceName: string,   // Name of the sequence being recorded
	 *   totalItems: number,     // Total items in sequence
	 *   settings: {
	 *     pauseBetweenItems: number,
	 *     restBetweenReps: number,
	 *     prepTime: number,
	 *     playbackSpeed: number
	 *   },
	 *   items: [{               // Array of sequence items
	 *     index: number,
	 *     model: string,
	 *     animation: number,
	 *     repetitions: number
	 *   }],
	 *   startTime: string       // ISO timestamp
	 * }
	 */
	startRecording(metadata = null) {
		const message = { command: 'start_recording' };
		if (metadata) {
			message.metadata = metadata;
		}
		this._send(message);
	}

	/**
	 * Send stop recording command with optional timeline data
	 * @param {object} timeline - Optional session timeline with phase events
	 *
	 * Timeline structure (if provided):
	 * {
	 *   sessionStart: string,       // ISO timestamp
	 *   sessionStartMs: number,     // Unix timestamp in ms
	 *   sessionEnd: string,         // ISO timestamp
	 *   totalDurationMs: number,    // Total session duration
	 *   events: [{                  // Array of timeline events
	 *     type: string,             // 'phase_change', 'session_end', etc.
	 *     time: string,             // ISO timestamp
	 *     offsetMs: number,         // Offset from session start
	 *     phase: string,            // Phase name (for phase_change events)
	 *     item: number,             // Item index
	 *     rep: number               // Repetition number
	 *   }],
	 *   summary: {
	 *     totalTutorialTimeMs: number,
	 *     totalPrepTimeMs: number,
	 *     totalMoveTimeMs: number,
	 *     totalRestTimeMs: number,
	 *     totalTransitionTimeMs: number,
	 *     itemsCompleted: number,
	 *     totalRepetitions: number
	 *   }
	 * }
	 */
	stopRecording(timeline = null) {
		const message = { command: 'stop_recording' };
		if (timeline) {
			message.timeline = timeline;
		}
		this._send(message);
	}

	/**
	 * Request current status
	 */
	getStatus() {
		this._send({ command: 'get_status' });
	}

	/**
	 * Point the (Quattrocento) server at a different OTBioLab+ configuration file.
	 * The server reloads it, rebuilds the device for the new channel count /
	 * sample rate, and reconnects. `path` is a path on the server machine.
	 * @param {string} path - Path to the OTBioLab+ config file (.otb+stp / XML)
	 */
	setOtbConfig(path) {
		this._send({ command: 'set_otb_config', path });
	}

	// -------------------------------------------------------------------------
	// STIMULATION COMMANDS
	// -------------------------------------------------------------------------

	/**
	 * Start an infinite train of stimulation pulses on the external stimulator.
	 * The server forwards this to the stimulator controller; the train runs
	 * until stimulateStop() is called (i.e. until the movement ends).
	 *
	 * @param {object} opts
	 * @param {Array} opts.channels - Channel configs, e.g.
	 *   [{ id: 1, amplitude: 8.0, pulse_width: 200, frequency: 50.0, is_biphasic: true }]
	 * @param {string} [opts.stimulatorType] - 'science_mode3' | 'science_mode4' | 'mock'
	 * @param {string} [opts.port] - Device COM port (optional)
	 * @param {string} [opts.controllerUrl] - Override stimulator controller base URL
	 */
	stimulateStart({ channels, stimulatorType, port, controllerUrl } = {}) {
		const message = { command: 'stimulate_start' };
		if (channels) message.channels = channels;
		if (stimulatorType) message.stimulator_type = stimulatorType;
		if (port) message.port = port;
		if (controllerUrl) message.controller_url = controllerUrl;
		this._send(message);
	}

	/**
	 * Stop any active stimulation on the external stimulator.
	 * @param {object} [opts]
	 * @param {string} [opts.controllerUrl] - Override stimulator controller base URL
	 */
	stimulateStop({ controllerUrl } = {}) {
		const message = { command: 'stimulate_stop' };
		if (controllerUrl) message.controller_url = controllerUrl;
		this._send(message);
	}

	// -------------------------------------------------------------------------
	// DECOMPOSITION COMMANDS
	// -------------------------------------------------------------------------

	/**
	 * Load a decomposition model on the server
	 * @param {string} modelPath - Path to the .pkl model file on the server
	 */
	loadModel(modelPath) {
		this._send({ command: 'load_model', model_path: modelPath });
	}

	/**
	 * Unload the current decomposition model
	 */
	unloadModel() {
		this._send({ command: 'unload_model' });
	}

	// -------------------------------------------------------------------------
	// MOVEMENT-CLASSIFICATION COMMANDS
	// -------------------------------------------------------------------------

	/**
	 * Load a movement-discrimination model on the server (frozen artifact from
	 * online_sim --save-model). Independent of the decomposition model.
	 * @param {string} modelPath - Path to the .pkl artifact on the server machine
	 */
	loadMovementModel(modelPath) {
		this._send({ command: 'load_movement_model', model_path: modelPath });
	}

	/**
	 * Unload the current movement-discrimination model
	 */
	unloadMovementModel() {
		this._send({ command: 'unload_movement_model' });
	}

	/**
	 * Toggle the STIM head's live output between full 4-class (close/trp/ext/rest)
	 * and a move/rest gate (1 - P(rest)). Runtime-safe; the clean head is unaffected.
	 * @param {boolean} on - true = move/rest gate, false = 4-class
	 */
	setStimBinary(on) {
		this._send({ command: 'mv_set_stim_binary', on: !!on });
	}

	/** Live-plot a channel (e.g. the stim trigger) on the frontend. null = stop. */
	monitorTrigger(channel) {
		this._send({ command: 'monitor_trigger', channel: channel == null ? null : channel });
	}

	onTriggerChunk(callback) { this.onTriggerChunkCallbacks.push(callback); }

	// -------------------------------------------------------------------------
	// MOVEMENT-MODEL TRAINING FLOW (record raw -> train -> calibrate)
	// -------------------------------------------------------------------------

	/** Start capturing a raw recording on the server for movement training. `meta`
	 * (label, session, class_order, class_sequence, trig_ch, grids) is saved with the
	 * recording and drives the informative filename + per-movement segment files. */
	mvRecordStart(meta = {}) {
		this._send({ command: 'mv_record_start', ...meta });
	}

	/** Stop capturing; the server saves the raw block (+ per-movement files) and keeps
	 * it in memory for the next train/calibrate. */
	mvRecordStop() {
		this._send({ command: 'mv_record_stop' });
	}

	/** Pause/resume capture WITHOUT ending the recording — used to exclude the tutorial
	 * phase (still shown to the patient) from the recorded/trained data. */
	mvRecordPause() {
		this._send({ command: 'mv_record_pause' });
	}

	mvRecordResume() {
		this._send({ command: 'mv_record_resume' });
	}

	/** Begin capturing the ONLINE run (raw stream + decisions), saved on stop with an
	 * informative name. `meta`: { label, session, class_order }. */
	mvOnlineStart(meta = {}) {
		this._send({ command: 'mv_online_start', ...meta });
	}

	/** Stop + save the current online run. */
	mvOnlineStop() {
		this._send({ command: 'mv_online_stop' });
	}

	/**
	 * Train the base movement model from the last captured recording.
	 * @param {object} cfg
	 * @param {number[]} cfg.classSequence - class index per trigger bout, in order
	 * @param {string[]} cfg.classOrder - [...movement names, 'rest']
	 * @param {object}   [cfg.params] - { clf, norm, pca, winMs, stepMs, smooth, trigCh }
	 */
	mvTrain({ classSequence, classOrder, params = {} }) {
		this._send({
			command: 'mv_train',
			class_sequence: classSequence,
			class_order: classOrder,
			clf: params.clf ?? 'qda',
			norm: params.norm ?? 'coral',
			pca: params.pca ?? 30,
			win_ms: params.winMs ?? 256,
			step_ms: params.stepMs ?? 64,
			smooth: params.smooth ?? 0.3,
			trig_ch: params.trigCh ?? 192,
			grids: params.grids ?? null,      // [[start,end],...] EMG grids; null = default
		});
	}

	/**
	 * Train the CLEAN (nostim) specialist from the last captured NO-stim recording.
	 * The patient gets no stimulation, but the stimulator still drives the trigger
	 * (ch3), so the server trigger-segments the bouts exactly like the stim block —
	 * same payload as mvTrain. Requires a prior mvTrain (shares its channel mask).
	 * @param {object} cfg
	 * @param {number[]} cfg.classSequence - class index per trigger bout
	 * @param {string[]} [cfg.classOrder] - [movements..., 'rest']
	 */
	mvTrainClean({ classSequence, classOrder }) {
		this._send({
			command: 'mv_train_clean',
			class_sequence: classSequence,
			class_order: classOrder,
		});
	}

	/**
	 * Finalize with the last captured (1-rep-each) calibration recording; the
	 * server CORAL-aligns, freezes the artifact, and auto-loads it live.
	 * @param {object} cfg
	 * @param {number[]} cfg.classSequence - class index per calibration bout
	 * @param {string}   [cfg.savePath] - server-side path for the artifact
	 */
	mvCalibrate({ classSequence, savePath }) {
		const msg = { command: 'mv_calibrate', class_sequence: classSequence };
		if (savePath) msg.save_path = savePath;
		this._send(msg);
	}

	/**
	 * Start 2-MU movement classification
	 * @param {object} config - Classification configuration
	 * @param {number} config.mu1Idx - Index of motor unit 1 (default: 0)
	 * @param {number} config.mu2Idx - Index of motor unit 2 (default: 1)
	 * @param {number} config.threshold - Firing rate threshold in Hz (default: 10.0)
	 * @param {number} config.windowSec - Sliding window duration in seconds (default: 1.0)
	 */
	startClassification(config = {}) {
		const msg = {
			command: 'start_classification',
			mu1_idx: config.mu1Idx ?? 0,
			mu2_idx: config.mu2Idx ?? 1,
			threshold: config.threshold ?? 10.0,
			window_sec: config.windowSec ?? 1.0,
		};
		if (config.btPort) msg.bt_port = config.btPort;
		console.log('[EMGClient] startClassification sending:', JSON.stringify(msg));
		this._send(msg);
	}

	/**
	 * Stop movement classification
	 */
	stopClassification() {
		this._send({ command: 'stop_classification' });
	}

	/**
	 * Get data for a specific channel
	 * @param {number} channelIndex - Channel index (0-based)
	 * @returns {Float32Array} - Recent samples for the channel
	 */
	getChannelData(channelIndex) {
		if (channelIndex < 0 || channelIndex >= this.channelBuffers.length) {
			return new Float32Array(this.bufferSize);
		}
		return this.channelBuffers[channelIndex];
	}

	/**
	 * Get data for multiple channels
	 * @param {number} startChannel - First channel index
	 * @param {number} count - Number of channels
	 * @returns {Float32Array[]} - Array of channel data
	 */
	getChannelsData(startChannel, count) {
		const result = [];
		for (let i = 0; i < count; i++) {
			result.push(this.getChannelData(startChannel + i));
		}
		return result;
	}

	// -------------------------------------------------------------------------
	// CALLBACKS
	// -------------------------------------------------------------------------

	onConnect(callback) {
		this.onConnectCallbacks.push(callback);
	}

	onDisconnect(callback) {
		this.onDisconnectCallbacks.push(callback);
	}

	onData(callback) {
		this.onDataCallbacks.push(callback);
	}

	onRecordingStatus(callback) {
		this.onRecordingStatusCallbacks.push(callback);
	}

	onRecordingSaved(callback) {
		this.onRecordingSavedCallbacks.push(callback);
	}

	onError(callback) {
		this.onErrorCallbacks.push(callback);
	}

	onDecompositionStatus(callback) {
		this.onDecompositionStatusCallbacks.push(callback);
	}

	onDecompositionData(callback) {
		this.onDecompositionDataCallbacks.push(callback);
	}

	onClassificationStatus(callback) {
		this.onClassificationStatusCallbacks.push(callback);
	}

	onClassification(callback) {
		this.onClassificationCallbacks.push(callback);
	}

	onConfigStatus(callback) {
		this.onConfigStatusCallbacks.push(callback);
	}

	onMovementStatus(callback) {
		this.onMovementStatusCallbacks.push(callback);
	}

	onMovementDecision(callback) {
		this.onMovementDecisionCallbacks.push(callback);
	}

	onMvTrainStatus(callback) {
		this.onMvTrainStatusCallbacks.push(callback);
	}

	/**
	 * Ack for stimulate_start/stimulate_stop.
	 * @param {Function} callback - ({ active, status, ...result })
	 */
	onStimulationStatus(callback) {
		this.onStimulationStatusCallbacks.push(callback);
	}

	/**
	 * Acquisition device lost / regained (live server only).
	 * @param {Function} callback - ({ connected, message })
	 */
	onDeviceStatus(callback) {
		this.onDeviceStatusCallbacks.push(callback);
	}

	/**
	 * Simulated playback reached the end of the file.
	 * @param {Function} callback - (msg)
	 */
	onStreamEnded(callback) {
		this.onStreamEndedCallbacks.push(callback);
	}

	// -------------------------------------------------------------------------
	// PRIVATE
	// -------------------------------------------------------------------------

	_send(data) {
		if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
			console.warn('EMGClient: Cannot send, not connected');
			return false;
		}

		this.ws.send(JSON.stringify(data));
		return true;
	}

	_handleMessage(rawData) {
		let msg;
		try {
			msg = JSON.parse(rawData);
		} catch (e) {
			console.error('EMGClient: Invalid JSON message', e);
			return;
		}

		switch (msg.type) {
			case 'connected':
				this._handleConnected(msg);
				break;

			case 'emg_chunk':
				this._handleEMGChunk(msg);
				break;

			case 'decomposition_chunk':
				this._handleDecompositionChunk(msg);
				break;

			case 'decomposition_status':
				this._handleDecompositionStatus(msg);
				break;

			case 'classification_status':
				this._handleClassificationStatus(msg);
				break;

			case 'movement_status':
				this._handleMovementStatus(msg);
				break;

			case 'movement_decision':
				this._handleMovementDecision(msg);
				break;

			case 'mv_train_status':
				this._handleMvTrainStatus(msg);
				break;

			case 'trigger_chunk':
				for (const cb of this.onTriggerChunkCallbacks) cb(msg);
				break;

			case 'trigger_monitor_status':
				break;

			case 'stimulation_status':
				this._handleStimulationStatus(msg);
				break;

			case 'device_status':
				this._handleDeviceStatus(msg);
				break;

			case 'stream_ended':
				console.log('EMGClient: stream ended (simulated playback finished)');
				for (const cb of this.onStreamEndedCallbacks) cb(msg);
				break;

			case 'recording_status':
				this._handleRecordingStatus(msg);
				break;

			case 'recording_saved':
				this._handleRecordingSaved(msg);
				break;

			case 'status':
				this._handleStatus(msg);
				break;

			case 'config_status':
				this._handleConfigStatus(msg);
				break;

			case 'error':
				this._notifyError(msg.message);
				break;

			default:
				console.warn('EMGClient: Unknown message type', msg.type);
		}
	}

	_handleConnected(msg) {
		this.sampleRate = msg.sample_rate;
		this.nChannels = msg.n_channels;
		this.preTriggerSeconds = msg.pre_trigger_seconds;
		this.decompositionActive = msg.decomposition_active || false;
		this.nMUs = msg.n_mus || 0;
		this.classificationActive = msg.classification_active || false;
		this.liveEmgEnabled = msg.live_emg_enabled !== false;
		this.movementActive = msg.movement_active || false;
		if (Array.isArray(msg.movement_class_order)) {
			this.movementClassOrder = msg.movement_class_order;
		}

		// Initialize channel buffers
		this.channelBuffers = [];
		for (let i = 0; i < this.nChannels; i++) {
			this.channelBuffers.push(new Float32Array(this.bufferSize));
		}

		console.log(`EMGClient: Server info - ${this.nChannels} channels @ ${this.sampleRate} Hz` +
			(this.decompositionActive ? `, decomposition active (${this.nMUs} MUs)` : ''));

		for (const cb of this.onConnectCallbacks) {
			cb({
				sampleRate: this.sampleRate,
				nChannels: this.nChannels,
				preTriggerSeconds: this.preTriggerSeconds,
				decompositionActive: this.decompositionActive,
				nMUs: this.nMUs,
				classificationActive: this.classificationActive,
				liveEmgEnabled: this.liveEmgEnabled,
			});
		}

		this._rehydrateStatusFromConnect(msg);
	}

	/**
	 * Re-announce decomposition / classification / movement state after a connect.
	 *
	 * The `connected` payload embeds decomp.get_status() and movement.get_status(),
	 * but under namespaced keys (`decomposition_active`, `movement_active`) rather
	 * than the flat `active` the standalone status messages use. Without this, only
	 * onConnect fires: reconnecting to a server that already has models loaded left
	 * DecompositionView with no MU rows, MovementStimController un-armed, and the
	 * stim-binary toggle showing a stale label.
	 *
	 * Runs after onConnect so listeners have channel count / sample rate first.
	 * Every consumer handles `active: false`, so a fresh connect is a safe no-op.
	 */
	_rehydrateStatusFromConnect(msg) {
		this._handleDecompositionStatus({
			active: this.decompositionActive,
			n_mus: this.nMUs,
			model_path: msg.model_path,
			channels_to_remove: msg.channels_to_remove,
		});

		this._handleClassificationStatus({
			active: this.classificationActive,
			...(msg.classification_config || {}),
		});

		this._handleMovementStatus({
			active: this.movementActive,
			movement_model_path: msg.movement_model_path,
			movement_class_order: msg.movement_class_order,
			movement_n_classes: msg.movement_n_classes,
			movement_stim_binary: msg.movement_stim_binary,
			movement_config: msg.movement_config,
			movement_two_stage: msg.movement_two_stage,
		});
	}

	_handleStimulationStatus(msg) {
		if (msg.status && msg.status !== 'success') {
			console.warn('EMGClient: stimulation command failed -', msg.status, msg.message || '');
		}
		for (const cb of this.onStimulationStatusCallbacks) {
			cb(msg);
		}
	}

	_handleDeviceStatus(msg) {
		if (msg.connected) {
			console.log('EMGClient: device reconnected -', msg.message);
		} else {
			console.warn('EMGClient: device lost -', msg.message);
		}
		for (const cb of this.onDeviceStatusCallbacks) {
			cb({ connected: !!msg.connected, message: msg.message });
		}
	}

	_handleEMGChunk(msg) {
		// msg.data is [n_channels][n_samples]
		const data = msg.data;
		const nSamples = msg.n_samples;

		// Update channel buffers (circular buffer approach)
		for (let ch = 0; ch < data.length && ch < this.channelBuffers.length; ch++) {
			const buffer = this.channelBuffers[ch];
			const samples = data[ch];

			// Shift existing data and append new samples
			if (nSamples < this.bufferSize) {
				// Shift left
				buffer.copyWithin(0, nSamples);
				// Copy new samples to the end
				for (let i = 0; i < nSamples; i++) {
					buffer[this.bufferSize - nSamples + i] = samples[i];
				}
			} else {
				// Replace entire buffer with last bufferSize samples
				for (let i = 0; i < this.bufferSize; i++) {
					buffer[i] = samples[nSamples - this.bufferSize + i];
				}
			}
		}

		// Notify listeners
		for (const cb of this.onDataCallbacks) {
			cb({
				data: this.channelBuffers,
				timestamp: msg.timestamp,
				nSamples,
			});
		}
	}

	_handleRecordingStatus(msg) {
		this.isRecording = msg.recording;

		for (const cb of this.onRecordingStatusCallbacks) {
			cb({
				recording: msg.recording,
				message: msg.message,
			});
		}
	}

	_handleRecordingSaved(msg) {
		for (const cb of this.onRecordingSavedCallbacks) {
			cb({
				filepath: msg.filepath,
				durationSeconds: msg.duration_seconds,
				nChannels: msg.n_channels,
				nSamples: msg.n_samples,
			});
		}
	}

	_handleStatus(msg) {
		// Update local state from status response
		if (msg.decomposition_active !== undefined) {
			this.decompositionActive = msg.decomposition_active;
			this.nMUs = msg.n_mus || 0;
		}
		if (msg.classification_active !== undefined) {
			this.classificationActive = msg.classification_active;
		}
		console.log('EMGClient: Status', msg);
	}

	_handleConfigStatus(msg) {
		console.log('EMGClient: config status -', msg.message);
		for (const cb of this.onConfigStatusCallbacks) {
			cb(msg);
		}
	}

	_handleDecompositionStatus(msg) {
		this.decompositionActive = msg.active;
		this.nMUs = msg.n_mus || 0;

		console.log(`EMGClient: Decomposition ${msg.active ? 'active' : 'inactive'}` +
			(msg.active ? ` (${msg.n_mus} MUs)` : ''));

		for (const cb of this.onDecompositionStatusCallbacks) {
			cb({
				active: msg.active,
				nMUs: msg.n_mus || 0,
				modelPath: msg.model_path,
				channelsToRemove: msg.channels_to_remove,
				channelsToKeep: msg.channels_to_keep,
				hasCentroids: msg.has_centroids,
				hasNormFactors: msg.has_norm_factors,
			});
		}
	}

	_handleDecompositionChunk(msg) {
		// msg.sources: [n_mus][n_samples] - MU source signals
		// msg.spikes: { "0": [12, 48], "1": [5], ... } - spike indices per MU
		// msg.sil_scores: [n_mus] - silhouette quality scores
		// msg.firing_rates: [n_mus] - firing rates in Hz
		// msg.n_samples: number of new samples in this chunk
		// msg.total_samples: cumulative sample count
		// msg.classification: { label, mu1_firing_rate, mu2_firing_rate, transition } (optional)

		const decompositionData = {
			sources: msg.sources,
			spikes: msg.spikes,
			silScores: msg.sil_scores,
			firingRates: msg.firing_rates,
			nSamples: msg.n_samples,
			totalSamples: msg.total_samples,
			timestamp: msg.timestamp,
		};

		for (const cb of this.onDecompositionDataCallbacks) {
			cb(decompositionData);
		}

		// If classification data is included, also notify classification listeners
		if (msg.classification) {
			for (const cb of this.onClassificationCallbacks) {
				cb({
					label: msg.classification.label,
					mu1FiringRate: msg.classification.mu1_firing_rate,
					mu2FiringRate: msg.classification.mu2_firing_rate,
					transition: msg.classification.transition,
					timestamp: msg.timestamp,
				});
			}
		}
	}

	_handleClassificationStatus(msg) {
		this.classificationActive = msg.active;

		console.log(`[EMGClient] classification_status received:`, JSON.stringify(msg));

		for (const cb of this.onClassificationStatusCallbacks) {
			cb({
				active: msg.active,
				mu1Idx: msg.mu1_idx,
				mu2Idx: msg.mu2_idx,
				threshold: msg.threshold,
				windowSec: msg.window_sec,
			});
		}
	}

	_handleMovementStatus(msg) {
		this.movementActive = msg.active;
		if (Array.isArray(msg.movement_class_order)) {
			this.movementClassOrder = msg.movement_class_order;
		}
		if (msg.movement_stim_binary !== undefined) {
			this.movementStimBinary = !!msg.movement_stim_binary;
		}
		console.log(`[EMGClient] movement ${msg.active ? 'active' : 'inactive'}` +
			(msg.active ? ` (${msg.movement_n_classes} classes)` : ''));

		for (const cb of this.onMovementStatusCallbacks) {
			cb({
				active: msg.active,
				modelPath: msg.movement_model_path,
				classOrder: msg.movement_class_order || this.movementClassOrder,
				nClasses: msg.movement_n_classes,
				fixedLatencyMs: msg.fixed_latency_ms,
				config: msg.movement_config,
				stimBinary: msg.movement_stim_binary,
			});
		}
	}

	_handleMovementDecision(msg) {
		// msg: { label, pred, probs:[close,trp,ext,rest], class_order, n_decisions,
		//        end_sample, n_samples, total_samples, timestamp,
		//        stim (bool: which head produced it), route_stim, route_clean }
		for (const cb of this.onMovementDecisionCallbacks) {
			cb({
				label: msg.label,
				pred: msg.pred,
				probs: msg.probs,
				classOrder: msg.class_order || this.movementClassOrder,
				nDecisions: msg.n_decisions,
				endSample: msg.end_sample,
				totalSamples: msg.total_samples,
				timestamp: msg.timestamp,
				// Which specialist head classified this window: true = stim head
				// (trigger fired within the window), false = clean/no-stim head.
				stim: msg.stim,
				routeStim: msg.route_stim,
				routeClean: msg.route_clean,
				// STIM-head move/rest gate readout: `binary` set when the stim head is in
				// gate mode; `moving` = p_move >= 0.5; `pMove` = 1 - P(rest) (smoothed).
				binary: msg.binary,
				moving: msg.moving,
				pMove: msg.p_move,
			});
		}
	}

	_handleMvTrainStatus(msg) {
		// msg.state: recording | recorded | training | trained | calibrating |
		//            calibrated | error  (+ per-state summary fields)
		console.log('[EMGClient] mv_train_status:', msg.state, msg.message || '');
		for (const cb of this.onMvTrainStatusCallbacks) {
			cb(msg);
		}
	}

	_notifyDisconnect() {
		for (const cb of this.onDisconnectCallbacks) {
			cb();
		}
	}

	_notifyError(message) {
		console.error('[EMGClient] Error received from server:', message);
		for (const cb of this.onErrorCallbacks) {
			cb(message);
		}
	}

	_scheduleReconnect() {
		this._clearReconnectTimer();
		this._reconnectTimer = setTimeout(() => {
			console.log('EMGClient: Attempting to reconnect...');
			this.connect();
		}, this.reconnectInterval);
	}

	_clearReconnectTimer() {
		if (this._reconnectTimer) {
			clearTimeout(this._reconnectTimer);
			this._reconnectTimer = null;
		}
	}
}
