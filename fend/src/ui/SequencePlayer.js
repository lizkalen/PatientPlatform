/**
 * Manages playback of animation sequences with configurable repetitions.
 * Coordinates with WebGLView to play animations in order.
 *
 * Phases:
 * - 'idle': Not playing
 * - 'prep': Preparation phase before movement (shows "Get Ready")
 * - 'move': Animation is playing (shows "Move")
 * - 'rest': Rest period between repetitions (shows "Rest/Hold")
 * - 'transition': Transitioning between items
 */
import * as THREE from 'three';

// Phase constants
export const PHASE = {
	IDLE: 'idle',
	TUTORIAL: 'tutorial',
	PREP: 'prep',
	MOVE: 'move',
	REST: 'rest',
	TRANSITION: 'transition',
};

export default class SequencePlayer {
	constructor() {
		// Reference to WebGLView (set via setWebGLView)
		this.webglView = null;

		// Reference to EMGClient (set via App.js)
		this.onEMGClient = null;

		// Current sequence configuration
		this.config = null;

		// Playback state
		this.isPlaying = false;
		this.currentItemIndex = 0;
		this.currentRepetition = 0;

		// Phase tracking for visual cues
		this.currentPhase = PHASE.IDLE;
		this.phaseTimeRemaining = 0;
		this._phaseTimer = null;

		// Bound handler for animation finished event
		this._onAnimationFinished = this._handleAnimationFinished.bind(this);

		// Callbacks for state changes
		this.onChangeCallbacks = [];
		this.onSequenceCompleteCallbacks = [];
		this.onPhaseChangeCallbacks = [];
		this.onTutorialRequestCallbacks = [];

		// Tutorial settings
		this._tutorialEnabled = false;
		this._tutorialResolve = null;
		this._hasCompletedFirstRun = false; // Track if sequence has looped

		// Session timeline tracking
		this._timeline = null;
		this._lastPhaseStartTime = null;

		// Whether a stimulation train we started is currently running. Cleared ONLY on a
		// confirmed send, so a stop that never left the socket cannot be recorded as a
		// success (plan A2) — `_pendingStop` latches that retry for the next connect.
		this._stimActive = false;
		this._pendingStop = false;
		this._alarmListeners = [];   // operator-visible stim alarms -> status banner

		// When true, an external driver owns the EMG recording lifecycle and the
		// player must not start/stop recordings itself. MovementSessionController
		// sets this while it runs the session's own mv_record_* capture.
		this._externalRecording = false;
	}

	/**
	 * Hand EMG recording control to an external driver.
	 *
	 * The movement session brackets each block with its own mv_record_start/stop,
	 * then drives this player to cue the movements. Without this flag the player
	 * would also fire start_recording/stop_recording, producing two independent,
	 * mutually unaware recordings per block.
	 *
	 * @param {boolean} enabled - true while an external driver owns recording
	 */
	setExternalRecording(enabled) {
		this._externalRecording = !!enabled;
	}

	/**
	 * Set the WebGLView reference
	 * @param {WebGLView} webglView
	 */
	setWebGLView(webglView) {
		this.webglView = webglView;
	}

	/**
	 * Load a sequence configuration
	 * @param {object} config - Sequence configuration object
	 */
	loadConfig(config) {
		this.config = config;
		this.reset();
		this._notifyChange();
	}

	/**
	 * Register a callback for state changes
	 * @param {Function} callback - Called with (state) on changes
	 */
	onChange(callback) {
		this.onChangeCallbacks.push(callback);
	}

	/**
	 * Register a callback for sequence completion
	 * @param {Function} callback - Called when sequence finishes
	 */
	onSequenceComplete(callback) {
		this.onSequenceCompleteCallbacks.push(callback);
	}

	/**
	 * Register a callback for phase changes
	 * @param {Function} callback - Called with (phase, timeRemaining) on phase changes
	 */
	onPhaseChange(callback) {
		this.onPhaseChangeCallbacks.push(callback);
	}

	/**
	 * Register a callback for tutorial requests
	 * @param {Function} callback - Called with (item, confirmFn) when tutorial should be shown
	 */
	onTutorialRequest(callback) {
		this.onTutorialRequestCallbacks.push(callback);
	}

	/**
	 * Enable or disable tutorial phase before each exercise
	 * @param {boolean} enabled
	 */
	setTutorialEnabled(enabled) {
		this._tutorialEnabled = enabled;
	}

	/**
	 * Check if tutorial is enabled
	 * @returns {boolean}
	 */
	isTutorialEnabled() {
		return this._tutorialEnabled;
	}

	/**
	 * Called when patient confirms they are ready after tutorial
	 */
	confirmTutorial() {
		if (this._tutorialResolve) {
			this._tutorialResolve();
			this._tutorialResolve = null;
		}
	}

	/**
	 * Start playing the sequence from current position
	 */
	play() {
		if (!this.config || !this.webglView) {
			console.warn('SequencePlayer: No config or webglView set');
			return;
		}

		if (this.config.items.length === 0) {
			console.warn('SequencePlayer: Sequence has no items');
			return;
		}

		this.isPlaying = true;

		// Initialize timeline tracking for this session
		this._initTimeline();

		// Start EMG recording when sequence starts, with metadata.
		// Skipped when an external driver owns recording (see setExternalRecording).
		if (!this._externalRecording && this.onEMGClient && this.onEMGClient.isConnected) {
			const metadata = this._buildTrialMetadata();
			this.onEMGClient.startRecording(metadata);
		}

		// Start with prep phase if configured
		// Skip initial prep if tutorial will be shown (tutorial prepares the user)
		const prepTime = this.config.settings?.prepTime || 0;
		const willShowTutorial = this._tutorialEnabled && this.currentRepetition === 0 && !this._hasCompletedFirstRun;
		if (prepTime > 0 && !willShowTutorial) {
			this._setPhase(PHASE.PREP, prepTime, () => {
				this._playCurrentItem();
			});
		} else {
			this._playCurrentItem();
		}

		this._notifyChange();
	}

	/**
	 * Pause sequence playback
	 */
	pause() {
		this.isPlaying = false;
		this._clearPhaseTimer();
		// pause() does not emit a phase change, so stop stimulation explicitly.
		this._stopStimulation();
		if (this.webglView) {
			this.webglView.pause();
		}
		this._notifyChange();
	}

	/**
	 * Stop and reset to beginning
	 */
	stop() {
		this._removeAnimationListener();
		this._clearPhaseTimer();
		this.isPlaying = false;
		this.currentItemIndex = 0;
		this.currentRepetition = 0;
		this._stopStimulation();
		this._setPhase(PHASE.IDLE, 0);
		this._notifyChange();

		// Finalize timeline and stop EMG recording.
		// Skipped when an external driver owns recording (see setExternalRecording).
		if (!this._externalRecording && this.onEMGClient
			&& this.onEMGClient.isConnected && this.onEMGClient.isRecording) {
			const timeline = this._finalizeTimeline();
			this.onEMGClient.stopRecording(timeline);
		}
	}

	/**
	 * Reset to beginning without changing playing state
	 */
	reset() {
		this._removeAnimationListener();
		this._clearPhaseTimer();
		this._stopStimulation();
		this.currentItemIndex = 0;
		this.currentRepetition = 0;
		this.currentPhase = PHASE.IDLE;
		this.isPlaying = false;
		this._hasCompletedFirstRun = false;
	}

	/**
	 * Skip to next item in sequence
	 */
	skipToNext() {
		if (!this.config) return;

		this._clearPhaseTimer();
		// Leaving the current item is leaving its MOVE phase, but no phase change is
		// emitted here — stop stimulation explicitly, as pause()/stop()/reset() do.
		this._stopStimulation();
		this.currentRepetition = 0;
		this.currentItemIndex++;

		if (this.currentItemIndex >= this.config.items.length) {
			if (this.config.settings?.loopSequence) {
				this.currentItemIndex = 0;
			} else {
				this._onSequenceComplete();
				return;
			}
		}

		if (this.isPlaying) {
			this._playCurrentItem();
		}
		this._notifyChange();
	}

	/**
	 * Skip to previous item in sequence
	 */
	skipToPrevious() {
		if (!this.config) return;

		this._clearPhaseTimer();
		// See skipToNext: no phase change is emitted, so stop stimulation explicitly.
		this._stopStimulation();
		this.currentRepetition = 0;
		this.currentItemIndex = Math.max(0, this.currentItemIndex - 1);

		if (this.isPlaying) {
			this._playCurrentItem();
		}
		this._notifyChange();
	}

	/**
	 * Jump to a specific item index
	 * @param {number} index
	 */
	jumpToItem(index) {
		if (!this.config || index < 0 || index >= this.config.items.length) return;

		this._clearPhaseTimer();
		// See skipToNext: no phase change is emitted, so stop stimulation explicitly.
		this._stopStimulation();
		this.currentItemIndex = index;
		this.currentRepetition = 0;

		if (this.isPlaying) {
			this._playCurrentItem();
		}
		this._notifyChange();
	}

	/**
	 * Get current playback state
	 * @returns {object}
	 */
	getState() {
		const currentItem = this.config?.items[this.currentItemIndex] || null;
		return {
			isPlaying: this.isPlaying,
			currentItemIndex: this.currentItemIndex,
			currentRepetition: this.currentRepetition,
			totalItems: this.config?.items.length || 0,
			totalRepetitions: currentItem?.repetitions || 0,
			currentItem,
			sequenceName: this.config?.name || '',
			phase: this.currentPhase,
			phaseTimeRemaining: this.phaseTimeRemaining,
		};
	}

	// -------------------------------------------------------------------------
	// PRIVATE
	// -------------------------------------------------------------------------

	/**
	 * Build trial metadata for EMG recording
	 */
	_buildTrialMetadata() {
		const item = this.config?.items[this.currentItemIndex];
		return {
			// Provenance: which code produced this recording. APP_NAME/APP_VERSION/
			// GIT_SHA are Vite build-time defines (see vite.config.js). A GIT_SHA
			// ending in `-dirty` means the build had uncommitted changes.
			appName: APP_NAME,
			appVersion: APP_VERSION,
			gitSha: GIT_SHA,

			// Config metadata
			subjectId: this.config?.metadata?.subjectId || 'unknown',
			sessionId: this.config?.metadata?.sessionId || 'unknown',
			notes: this.config?.metadata?.notes || '',

			// Sequence info
			sequenceName: this.config?.name || '',
			totalItems: this.config?.items?.length || 0,

			// Settings
			settings: {
				pauseBetweenItems: this.config?.settings?.pauseBetweenItems || 0,
				restBetweenReps: this.config?.settings?.restBetweenReps || 0,
				prepTime: this.config?.settings?.prepTime || 0,
				playbackSpeed: this.config?.settings?.playbackSpeed || 1.0,
			},

			// Items summary
			items: this.config?.items?.map((item, idx) => ({
				index: idx,
				model: item.model,
				animation: item.animation,
				repetitions: item.repetitions,
			})) || [],

			// Timestamp
			startTime: new Date().toISOString(),
		};
	}

	/**
	 * Initialize timeline tracking for a new session
	 */
	_initTimeline() {
		this._timeline = {
			sessionStart: new Date().toISOString(),
			sessionStartMs: Date.now(),
			sessionEnd: null,
			totalDurationMs: 0,
			events: [],
			summary: {
				totalTutorialTimeMs: 0,
				totalPrepTimeMs: 0,
				totalMoveTimeMs: 0,
				totalRestTimeMs: 0,
				totalTransitionTimeMs: 0,
				itemsCompleted: 0,
				totalRepetitions: 0,
			},
		};
		this._lastPhaseStartTime = null;
	}

	/**
	 * Record a timeline event
	 * @param {string} type - Event type (e.g., 'phase_start', 'rep_complete')
	 * @param {object} data - Additional event data
	 */
	_recordEvent(type, data = {}) {
		if (!this._timeline) return;

		const now = Date.now();
		const event = {
			type,
			time: new Date().toISOString(),
			offsetMs: now - this._timeline.sessionStartMs,
			...data,
		};
		this._timeline.events.push(event);
	}

	/**
	 * Record phase transition and track duration of previous phase
	 * @param {string} phase - New phase
	 */
	_recordPhaseChange(phase) {
		if (!this._timeline) return;

		const now = Date.now();

		// Close out previous phase duration
		if (this._lastPhaseStartTime && this.currentPhase !== PHASE.IDLE) {
			const duration = now - this._lastPhaseStartTime;
			this._addPhaseDuration(this.currentPhase, duration);
		}

		// Drive stimulation on move enter/leave. This is the single chokepoint
		// for every phase transition (including the combined rest+prep phase),
		// so comparing the old phase (this.currentPhase) to the new one here
		// catches all movement boundaries.
		this._handleStimulationTransition(this.currentPhase, phase);

		// Record the phase change event
		this._recordEvent('phase_change', {
			phase,
			item: this.currentItemIndex,
			rep: this.currentRepetition,
		});

		this._lastPhaseStartTime = now;
	}

	/**
	 * Add duration to the appropriate summary field
	 * @param {string} phase - Phase type
	 * @param {number} durationMs - Duration in milliseconds
	 */
	_addPhaseDuration(phase, durationMs) {
		if (!this._timeline) return;

		switch (phase) {
			case PHASE.TUTORIAL:
				this._timeline.summary.totalTutorialTimeMs += durationMs;
				break;
			case PHASE.PREP:
				this._timeline.summary.totalPrepTimeMs += durationMs;
				break;
			case PHASE.MOVE:
				this._timeline.summary.totalMoveTimeMs += durationMs;
				break;
			case PHASE.REST:
				this._timeline.summary.totalRestTimeMs += durationMs;
				break;
			case PHASE.TRANSITION:
				this._timeline.summary.totalTransitionTimeMs += durationMs;
				break;
		}
	}

	/**
	 * Finalize timeline with end time and return the complete timeline
	 * @returns {object|null} The finalized timeline
	 */
	_finalizeTimeline() {
		if (!this._timeline) return null;

		// Close out final phase if needed
		if (this._lastPhaseStartTime && this.currentPhase !== PHASE.IDLE) {
			const duration = Date.now() - this._lastPhaseStartTime;
			this._addPhaseDuration(this.currentPhase, duration);
		}

		this._timeline.sessionEnd = new Date().toISOString();
		this._timeline.totalDurationMs = Date.now() - this._timeline.sessionStartMs;

		// Record session end event
		this._recordEvent('session_end', {
			completed: this.currentItemIndex >= (this.config?.items?.length || 0),
		});

		return this._timeline;
	}

	/**
	 * Start or stop stimulation in response to a phase transition.
	 * Stimulation runs while the current item is in its MOVE phase, provided
	 * stimulation is enabled in the config and the item opts in via `stimulate`.
	 * @param {string} oldPhase - Phase being left
	 * @param {string} newPhase - Phase being entered
	 */
	_handleStimulationTransition(oldPhase, newPhase) {
		const stim = this.config?.stimulation;
		if (!stim || !stim.enabled) return;
		if (!this.onEMGClient) return;

		const enteringMove = newPhase === PHASE.MOVE && oldPhase !== PHASE.MOVE;
		const leavingMove = oldPhase === PHASE.MOVE && newPhase !== PHASE.MOVE;

		if (enteringMove) {
			if (!this.onEMGClient.isConnected) return;   // nothing to start over a dead socket
			const item = this.config?.items?.[this.currentItemIndex];
			if (item?.stimulate && !this._stimActive) {
				// Per-movement channels if defined for this item, else the global ones.
				const channels = item.stim?.channels?.length ? item.stim.channels : stim.channels;
				const sent = this.onEMGClient.stimulateStart({
					channels,
					stimulatorType: stim.stimulatorType,
					port: stim.port,
					controllerUrl: stim.controllerUrl,
				});
				// Claim the train only if the command actually left the socket.
				if (sent) this._stimActive = true;
			}
		} else if (leavingMove) {
			// Runs even while disconnected: _stopStimulation latches the retry itself,
			// and skipping it here would leave a live train with nothing tracking it.
			this._stopStimulation();
		}
	}

	/**
	 * Stop stimulation if we believe a train is running. Idempotent and safe to
	 * call from any teardown path (pause/stop/complete) so stimulation never
	 * outlives the movement, even on paths that don't emit a phase change.
	 *
	 * `_stimActive` is cleared ONLY on a confirmed send. A dead socket leaves the
	 * train running server-side, so the belief is retained (a second Stop press must
	 * still send), the retry is latched for the next connect, and the operator is
	 * alarmed rather than shown a silent "stopped" (plan A2).
	 *
	 * @returns {boolean} whether the command left the socket
	 */
	_stopStimulation() {
		if (!this._stimActive && !this._pendingStop) return true;
		if (!this.onEMGClient) { this._stimActive = false; this._pendingStop = false; return true; }
		const sent = this.onEMGClient.stimulateStop({
			controllerUrl: this.config?.stimulation?.controllerUrl,
		});
		if (sent) {
			this._stimActive = false;
			this._pendingStop = false;
		} else {
			const first = !this._pendingStop;
			this._pendingStop = true;
			if (first) {
				this._raiseAlarm('STOP NOT SENT — no connection to the server; '
					+ 'sequence stimulation may still be running');
			}
		}
		return sent;
	}

	/** Retry a stop that never left the socket. Wired to the client's connect hook. */
	flushPendingStimStop() {
		if (this._pendingStop) this._stopStimulation();
	}

	/** Subscribe to operator-visible stim alarms. Called with a message string. */
	onAlarm(cb) { this._alarmListeners.push(cb); }

	_raiseAlarm(message) {
		console.warn('[SequencePlayer] ALARM:', message);
		for (const cb of this._alarmListeners) {
			try { cb(message); } catch (e) { console.error('SequencePlayer alarm listener error', e); }
		}
	}

	/**
	 * Set the current phase and notify listeners
	 */
	_setPhase(phase, duration, onComplete) {
		// Record phase change for timeline tracking
		this._recordPhaseChange(phase);

		this.currentPhase = phase;
		this.phaseTimeRemaining = duration;

		// Notify phase change
		for (const cb of this.onPhaseChangeCallbacks) {
			cb(phase, duration);
		}

		// Set up timer if duration > 0
		if (duration > 0 && onComplete) {
			this._clearPhaseTimer();

			// Update countdown every 100ms
			const startTime = Date.now();
			const endTime = startTime + duration * 1000;

			this._phaseTimer = setInterval(() => {
				const remaining = Math.max(0, (endTime - Date.now()) / 1000);
				this.phaseTimeRemaining = remaining;

				// Notify phase update for countdown
				for (const cb of this.onPhaseChangeCallbacks) {
					cb(phase, remaining);
				}

				if (remaining <= 0) {
					this._clearPhaseTimer();
					if (this.isPlaying) {
						onComplete();
					}
				}
			}, 100);
		}
	}

	/**
	 * Clear the phase timer
	 */
	_clearPhaseTimer() {
		if (this._phaseTimer) {
			clearInterval(this._phaseTimer);
			this._phaseTimer = null;
		}
	}

	/**
	 * Play the current sequence item
	 */
	async _playCurrentItem() {
		if (!this.config || !this.webglView) return;

		const item = this.config.items[this.currentItemIndex];
		if (!item) {
			this._onSequenceComplete();
			return;
		}

		// Check if we need to load a different model
		const currentModelFile = this.webglView.constructor.AVAILABLE_MODELS[
			this.webglView.currentModelIndex
		]?.file;

		if (currentModelFile !== item.model) {
			// Transition phase while loading
			this._setPhase(PHASE.TRANSITION, 0);
			await this._loadModelAndWait(item.model);
		}

		// Play the specified animation. A stim-rest item advances the animation (so phase
		// timing + rep counting still fire) but at zero visual weight, so the hand holds
		// its neutral pose — the patient is cued to stay still, not to move.
		this.webglView.playAnimation(item.animation, true, item.restClass ? 0 : 1);

		// Show tutorial if enabled (before each new item, first rep only, first loop only)
		if (this._tutorialEnabled && this.currentRepetition === 0 && !this._hasCompletedFirstRun) {
			await this._showTutorial(item);
		}

		// Set up loop mode for repetitions
		this._setupAnimationLoop(item.repetitions);

		// Apply playback speed from config
		if (this.config.settings?.playbackSpeed) {
			this.webglView.playbackSpeed = this.config.settings.playbackSpeed;
		}

		// Reset animation to start position (important after tutorial where animation may have been playing)
		if (this.webglView.action) {
			this.webglView.action.time = 0;
			this.webglView.mixer.update(0);
		}

		// Show PREP phase before movement (traffic light countdown)
		const prepTime = this.config.settings?.prepTime || 0;
		if (prepTime > 0) {
			this._setPhase(PHASE.PREP, prepTime, () => {
				if (!this.isPlaying) return;
				this._setPhase(PHASE.MOVE, 0);
				this.webglView.play();
				this._notifyChange();
			});
		} else {
			// Same re-check the prepTime > 0 branch does inside its callback: this method
			// awaited the model load and the tutorial, so playback may have been stopped
			// while we were suspended. Without it, a Stop pressed during the tutorial is
			// undone the moment the patient presses "I'm Ready" (prepTime === 0 only).
			if (!this.isPlaying) return;
			this._setPhase(PHASE.MOVE, 0);
			this.webglView.play();
			this._notifyChange();
		}
	}

	/**
	 * Show tutorial overlay and wait for user confirmation
	 * @param {object} item - The sequence item to show tutorial for
	 * @returns {Promise<void>}
	 */
	_showTutorial(item) {
		return new Promise((resolve) => {
			this._tutorialResolve = resolve;
			this._setPhase(PHASE.TUTORIAL, 0);

			// Notify all tutorial request callbacks
			for (const cb of this.onTutorialRequestCallbacks) {
				cb(item, () => this.confirmTutorial());
			}

			// If no callbacks registered, resolve immediately
			if (this.onTutorialRequestCallbacks.length === 0) {
				resolve();
				this._tutorialResolve = null;
			}
		});
	}

	/**
	 * Load a model and wait for it to complete
	 * @param {string} modelFile
	 * @returns {Promise<void>}
	 */
	_loadModelAndWait(modelFile) {
		return new Promise((resolve) => {
			// Find model index
			const models = this.webglView.constructor.AVAILABLE_MODELS;
			const modelIndex = models.findIndex((m) => m.file === modelFile);

			if (modelIndex === -1) {
				console.warn(`SequencePlayer: Model '${modelFile}' not found`);
				resolve();
				return;
			}

			// One-shot: coexists with any other waiter (e.g. DecompositionController)
			// instead of the old save/restore dance over a single shared slot.
			this.webglView.onceModelLoaded(() => resolve());

			// Load the model
			this.webglView.currentModelIndex = modelIndex;
			this.webglView.loadModel(modelFile);
		});
	}

	/**
	 * Set up animation looping for the specified number of repetitions
	 * @param {number} repetitions
	 */
	_setupAnimationLoop(repetitions) {
		if (!this.webglView.action || !this.webglView.mixer) return;

		const action = this.webglView.action;

		// Remove any existing listener
		this._removeAnimationListener();

		// Set to play once (we'll handle repetitions manually)
		action.setLoop(THREE.LoopOnce, 1);
		action.clampWhenFinished = true;
		action.reset();
		// Keep paused until PREP finishes (reset() unpauses the action)
		action.paused = true;

		// Add listener for animation completion
		this.webglView.mixer.addEventListener('finished', this._onAnimationFinished);
	}

	/**
	 * Handle animation finished event
	 */
	_handleAnimationFinished() {
		if (!this.isPlaying) return;

		const item = this.config?.items[this.currentItemIndex];
		if (!item) return;

		this.currentRepetition++;
		this._notifyChange();

		// Track repetition completion in timeline
		if (this._timeline) {
			this._timeline.summary.totalRepetitions++;
			this._recordEvent('rep_complete', {
				item: this.currentItemIndex,
				rep: this.currentRepetition,
			});
		}

		if (this.currentRepetition < item.repetitions) {
			// Rest period before next repetition
			this._playRepetition();
		} else {
			// Track item completion in timeline
			if (this._timeline) {
				this._timeline.summary.itemsCompleted++;
				this._recordEvent('item_complete', {
					item: this.currentItemIndex,
				});
			}
			// Move to next item
			this._advanceToNextItem();
		}
	}

	/**
	 * Play another repetition of the current animation
	 */
	_playRepetition() {
		if (!this.webglView.action) return;

		const item = this.config?.items[this.currentItemIndex];
		// Use item-specific rest duration or fall back to global setting
		const restDuration = item?.restBetweenReps ?? this.config.settings?.restBetweenReps ?? 0;
		const prepTime = this.config.settings?.prepTime || 0;

		if (restDuration > 0 || prepTime > 0) {
			// Combined REST+PREP countdown: starts with REST (red), switches to PREP (yellow)
			this._setCombinedRestPrepPhase(restDuration, prepTime, () => {
				this._startMovement();
			});
		} else {
			this._startMovement();
		}
	}

	/**
	 * Set a combined REST+PREP phase with a single countdown
	 * Starts with REST (red light), transitions to PREP (yellow light) when prepTime remains
	 * @param {number} restDuration - Duration of REST phase in seconds
	 * @param {number} prepDuration - Duration of PREP phase in seconds
	 * @param {Function} onComplete - Callback when both phases complete
	 */
	_setCombinedRestPrepPhase(restDuration, prepDuration, onComplete) {
		const totalDuration = restDuration + prepDuration;

		if (totalDuration <= 0) {
			onComplete?.();
			return;
		}

		// Start with REST phase if restDuration > 0, otherwise start with PREP
		const initialPhase = restDuration > 0 ? PHASE.REST : PHASE.PREP;
		// Record the phase change so rest/prep periods appear in the timeline
		// (closes out the previous phase's duration before switching).
		this._recordPhaseChange(initialPhase);
		this.currentPhase = initialPhase;
		this.phaseTimeRemaining = totalDuration;

		// Notify initial phase
		for (const cb of this.onPhaseChangeCallbacks) {
			cb(initialPhase, totalDuration);
		}

		this._clearPhaseTimer();

		const startTime = Date.now();
		const endTime = startTime + totalDuration * 1000;
		const switchTime = endTime - prepDuration * 1000; // When to switch from REST to PREP

		this._phaseTimer = setInterval(() => {
			const now = Date.now();
			const remaining = Math.max(0, (endTime - now) / 1000);
			this.phaseTimeRemaining = remaining;

			// Check if we should switch from REST to PREP
			if (restDuration > 0 && prepDuration > 0 && now >= switchTime && this.currentPhase === PHASE.REST) {
				// Record the boundary so REST and PREP are split into separate timeline regions
				this._recordPhaseChange(PHASE.PREP);
				this.currentPhase = PHASE.PREP;
			}

			// Notify phase update for countdown
			for (const cb of this.onPhaseChangeCallbacks) {
				cb(this.currentPhase, remaining);
			}

			if (remaining <= 0) {
				this._clearPhaseTimer();
				if (this.isPlaying) {
					onComplete?.();
				}
			}
		}, 100);
	}

	/**
	 * Start the movement (reset and play animation)
	 */
	_startMovement() {
		if (!this.isPlaying || !this.webglView.action) return;

		this._setPhase(PHASE.MOVE, 0);
		this.webglView.action.reset();
		this.webglView.action.play();
		this._notifyChange();
	}

	/**
	 * Advance to the next item in the sequence
	 */
	_advanceToNextItem() {
		this.currentRepetition = 0;
		this.currentItemIndex++;

		if (this.currentItemIndex >= this.config.items.length) {
			if (this.config.settings?.loopSequence) {
				this.currentItemIndex = 0;
				this._hasCompletedFirstRun = true; // Skip tutorials on subsequent loops
				this._transitionToNextItem();
			} else {
				this._onSequenceComplete();
			}
		} else {
			this._transitionToNextItem();
		}
	}

	/**
	 * Handle transition between items
	 * Goes directly to the next item (which handles tutorial + prep internally)
	 */
	_transitionToNextItem() {
		const pauseDuration = this.config.settings?.pauseBetweenItems || 0;

		if (pauseDuration > 0) {
			this._setPhase(PHASE.TRANSITION, pauseDuration, () => {
				if (!this.isPlaying) return;
				this._playCurrentItem();
			});
		} else {
			this._playCurrentItem();
		}
	}

	/**
	 * Remove animation finished listener
	 */
	_removeAnimationListener() {
		if (this.webglView?.mixer) {
			this.webglView.mixer.removeEventListener('finished', this._onAnimationFinished);
		}
	}

	/**
	 * Called when sequence completes
	 */
	_onSequenceComplete() {
		this._removeAnimationListener();
		this._clearPhaseTimer();
		this.isPlaying = false;
		this._stopStimulation();
		this._setPhase(PHASE.IDLE, 0);
		this._notifyChange();

		// Finalize timeline and stop EMG recording.
		// Skipped when an external driver owns recording (see setExternalRecording).
		if (!this._externalRecording && this.onEMGClient
			&& this.onEMGClient.isConnected && this.onEMGClient.isRecording) {
			const timeline = this._finalizeTimeline();
			this.onEMGClient.stopRecording(timeline);
		}

		for (const cb of this.onSequenceCompleteCallbacks) {
			cb();
		}

		console.log('SequencePlayer: Sequence complete');
	}

	/**
	 * Notify all change callbacks
	 */
	_notifyChange() {
		const state = this.getState();
		for (const cb of this.onChangeCallbacks) {
			cb(state);
		}
	}
}
