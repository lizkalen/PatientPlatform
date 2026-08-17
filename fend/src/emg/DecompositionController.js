import * as THREE from 'three';

/**
 * Decomposition Controller
 * ========================
 *
 * Manages decomposition model loading/unloading, classification
 * configuration, and drives the 3D hand animation from classification labels.
 *
 * State machine (label transitions):
 *   REST (0) → MOVE1 (1): load label1 model if needed, play forward
 *   REST (0) → MOVE2 (2): load label2 model if needed, play forward
 *   MOVE (1|2) → REST (0): play current animation in reverse back to start
 *   MOVE1 ↔ MOVE2: do nothing (for now)
 * 
 * NOTE: we are not doing any transitions between MOVE1 and MOVE2 as that would require blending between two different animations, which is more complex to implement. 
 * Instead, if the user goes from MOVE1 to MOVE2 (or vice versa), we just keep playing the current animation until they return to REST, 
 * at which point we can load the new model for the next MOVE.
 * 
 * - Gian
 */
export default class DecompositionController {
	constructor(emgClient, webgl) {
		this.emgClient = emgClient;
		this.webgl = webgl;

		// Decomposition model config
		this.modelPath = '';

		// Classification config
		this.mu1Idx = 0;
		this.mu2Idx = 1;
		this.threshold = 10.0;
		this.windowSec = 1.0;
		this.btPort = '';  // Optional COM port for exo hand (e.g. 'COM3')

		// 3D model files for each movement label
		this.label1ModelFile = null;
		this.label2ModelFile = null;

		// State tracking
		this.currentLabel = -1; // -1 = uninitialized
		this.currentlyLoadedFile = null; // which .glb is in the scene right now
		this.isDecompositionActive = false;
		this.isClassificationActive = false;
		this.nMUs = 0;

		// Animation target tracking (for update-loop based stopping)
		this._playTargetTime = null;  // time to stop at (null = not tracking)
		this._playDirection = null;   // 'forward' | 'backward'

		// State change callbacks (for GUIView refresh)
		this._stateChangeCallbacks = [];

		// Subscribe to EMG client events
		this._subscribeToClient();
	}

	// -------------------------------------------------------------------------
	// PUBLIC API
	// -------------------------------------------------------------------------

	/**
	 * Load a decomposition model on the server
	 * @param {string} path - Server-side path to .pkl file
	 */
	loadModel(path) {
		if (!path) return;
		this.modelPath = path;
		this.emgClient.loadModel(path);
	}

	/**
	 * Unload the current decomposition model
	 */
	unloadModel() {
		this.emgClient.unloadModel();
		this.currentLabel = -1;
		this._notifyStateChange();
	}

	/**
	 * Start 2-MU movement classification (also starts recording)
	 */
	startClassification() {
		console.log(`[DecompCtrl] startClassification: mu1=${this.mu1Idx} mu2=${this.mu2Idx} ` +
			`threshold=${this.threshold} window=${this.windowSec} btPort=${JSON.stringify(this.btPort)}`);
		// Start recording so classification data is saved
		this.emgClient.startRecording({ source: 'classification' });

		this.emgClient.startClassification({
			mu1Idx: this.mu1Idx,
			mu2Idx: this.mu2Idx,
			threshold: this.threshold,
			windowSec: this.windowSec,
			btPort: this.btPort || undefined,
		});
	}

	/**
	 * Stop movement classification (also stops recording & saves data)
	 */
	stopClassification() {
		this.emgClient.stopClassification();
		// Stop recording so everything is saved to disk
		this.emgClient.stopRecording();
		// Return hand to rest state
		if (this.currentLabel !== 0) {
			this._transitionToRest();
		}
	}

	/**
	 * Set the 3D model file to play for classification label 1
	 * @param {string} modelFile - e.g. 'wrist.glb'
	 */
	setLabel1Model(modelFile) {
		this.label1ModelFile = modelFile;
	}

	/**
	 * Set the 3D model file to play for classification label 2
	 * @param {string} modelFile - e.g. 'tripodpinch.glb'
	 */
	setLabel2Model(modelFile) {
		this.label2ModelFile = modelFile;
	}

	/**
	 * Register a callback for controller state changes (for GUIView refresh)
	 * @param {function} cb
	 */
	onStateChange(cb) {
		this._stateChangeCallbacks.push(cb);
	}

	// -------------------------------------------------------------------------
	// PRIVATE: EMG client subscriptions
	// -------------------------------------------------------------------------

	_subscribeToClient() {
		this.emgClient.onDecompositionStatus((status) => {
			this.isDecompositionActive = status.active;
			this.nMUs = status.nMUs;
			if (!status.active) {
				this.currentLabel = -1;
				this.isClassificationActive = false;
			}
			this._notifyStateChange();
		});

		this.emgClient.onClassificationStatus((status) => {
			this.isClassificationActive = status.active;
			if (status.active) {
				this.mu1Idx = status.mu1Idx;
				this.mu2Idx = status.mu2Idx;
				this.threshold = status.threshold;
				this.windowSec = status.windowSec;
			}
			this._notifyStateChange();
		});

		this.emgClient.onClassification((result) => {
			this._onClassification(result);
		});
	}

	// -------------------------------------------------------------------------
	// PUBLIC: Per-frame update (called from App.update())
	// -------------------------------------------------------------------------

	/**
	 * Called every animation frame. Stops playback when the target time is reached.
	 */
	update() {
		if (this._playTargetTime === null) return;
		if (!this.webgl.isPlaying) {
			console.warn('[DecompCtrl] update: have target but webgl.isPlaying=false');
			return;
		}

		const t = this.webgl.animationTime;
		const action = this.webgl.action;

		// Log every ~60 frames to avoid flooding
		if (!this._debugFrameCount) this._debugFrameCount = 0;
		this._debugFrameCount++;
		if (this._debugFrameCount % 60 === 0) {
			console.log(`[DecompCtrl] update: dir=${this._playDirection} t=${t.toFixed(3)} target=${this._playTargetTime.toFixed(3)} action.time=${action?.time?.toFixed(3)} action.enabled=${action?.enabled} action.paused=${action?.paused} timeScale=${action?.timeScale}`);
		}

		if (this._playDirection === 'forward' && t >= this._playTargetTime) {
			console.log(`[DecompCtrl] forward target reached: t=${t.toFixed(3)}, stopping at ${this._playTargetTime.toFixed(3)}`);
			this.webgl.pause();
			this.webgl.setAnimationTime(this._playTargetTime);
			this._playTargetTime = null;
		} else if (this._playDirection === 'backward' && t <= this._playTargetTime) {
			console.log(`[DecompCtrl] backward target reached: t=${t.toFixed(3)}, stopping at ${this._playTargetTime.toFixed(3)}`);
			this.webgl.pause();
			this.webgl.setAnimationTime(this._playTargetTime);
			this._playTargetTime = null;
		}
	}

	// -------------------------------------------------------------------------
	// PRIVATE: Classification-driven hand animation
	// -------------------------------------------------------------------------

	_onClassification(result) {
		const newLabel = result.label;
		console.log(`[DecompCtrl] _onClassification: label=${newLabel} transition=${result.transition} ` +
			`mu1FR=${result.mu1FiringRate?.toFixed(1)} mu2FR=${result.mu2FiringRate?.toFixed(1)}`);

		// No change
		if (newLabel === this.currentLabel) return;

		const prevLabel = this.currentLabel;
		this.currentLabel = newLabel;
		console.log(`[DecompCtrl] label change: ${prevLabel} → ${newLabel}`);

		// MOVE1 ↔ MOVE2: do nothing (same model stays loaded, keep playing)
		if (prevLabel >= 1 && newLabel >= 1) {
			console.log('[DecompCtrl] MOVE↔MOVE: ignoring');
			return;
		}

		if (newLabel === 0) {
			console.log('[DecompCtrl] → REST transition');
			this._transitionToRest();
		} else {
			// REST → MOVE (1 or 2)
			const targetFile = newLabel === 1 ? this.label1ModelFile : this.label2ModelFile;
			if (!targetFile) {
				console.warn(`[DecompCtrl] No model file configured for label ${newLabel}`);
				return;
			}
			console.log(`[DecompCtrl] → MOVE${newLabel} transition, file=${targetFile}`);
			this._transitionToMove(targetFile);
		}
	}

	/**
	 * Returns the start time of the first 'movement' segment in the SegmentManager.
	 * Falls back to animationDuration if no segments are configured (play full clip).
	 */
	_getMovementStartTime() {
		const sm = this.webgl.segmentManager;
		if (sm?.segments?.length) {
			const moveSeg = sm.segments.find(s => s.type === 'movement');
			if (moveSeg) return moveSeg.start;
		}
		return this.webgl.animationDuration || 1;
	}

	_transitionToRest() {
		const action = this.webgl.action;
		if (!action) {
			console.warn('[DecompCtrl] _transitionToRest: no action, just pausing');
			this.webgl.pause();
			return;
		}

		// Save current position before reset (reset sets time to 0)
		const currentTime = this.webgl.animationTime;
		console.log(`[DecompCtrl] _transitionToRest: currentTime=${currentTime.toFixed(3)}`, {
			actionEnabled: action.enabled, actionPaused: action.paused, actionTime: action.time,
			timeScale: action.timeScale, isPlaying: this.webgl.isPlaying,
		});

		// Reset action to re-enable it (LoopOnce disables the action when finished)
		action.reset();
		action.setLoop(THREE.LoopOnce);
		action.clampWhenFinished = true;
		action.timeScale = -1;
		action.time = currentTime; // Restore position so we play backward from here

		console.log(`[DecompCtrl] _transitionToRest: after reset+configure`, {
			actionEnabled: action.enabled, actionPaused: action.paused, actionTime: action.time,
			timeScale: action.timeScale,
		});

		this._playTargetTime = 0;
		this._playDirection = 'backward';
		this.webgl.play();
	}

	_transitionToMove(targetFile) {
		const doPlay = () => {
			const action = this.webgl.action;
			if (!action) {
				console.warn('[DecompCtrl] _transitionToMove doPlay: no action!');
				return;
			}

			console.log(`[DecompCtrl] _transitionToMove doPlay BEFORE reset`, {
				actionEnabled: action.enabled, actionPaused: action.paused, actionTime: action.time,
				timeScale: action.timeScale, isPlaying: this.webgl.isPlaying,
				animDuration: this.webgl.animationDuration,
			});

			// Reset action to re-enable it (LoopOnce disables the action when finished)
			// reset() also sets time back to 0, which is where we want to start
			action.reset();
			action.setLoop(THREE.LoopOnce);
			action.clampWhenFinished = true;
			action.timeScale = 1;

			this._playTargetTime = this._getMovementStartTime();
			this._playDirection = 'forward';
			this.webgl.animationTime = 0;

			console.log(`[DecompCtrl] _transitionToMove doPlay AFTER reset`, {
				actionEnabled: action.enabled, actionPaused: action.paused, actionTime: action.time,
				timeScale: action.timeScale, targetTime: this._playTargetTime,
				segments: this.webgl.segmentManager?.segments,
			});

			this.webgl.play();
		};

		if (targetFile === this.currentlyLoadedFile) {
			doPlay();
		} else {
			this.currentlyLoadedFile = targetFile;
			this.webgl.onceModelLoaded(() => doPlay());
			this.webgl.loadModel(targetFile);
		}
	}

	// -------------------------------------------------------------------------
	// PRIVATE: Notifications
	// -------------------------------------------------------------------------

	_notifyStateChange() {
		for (const cb of this._stateChangeCallbacks) {
			cb({
				isDecompositionActive: this.isDecompositionActive,
				isClassificationActive: this.isClassificationActive,
				nMUs: this.nMUs,
			});
		}
	}
}
