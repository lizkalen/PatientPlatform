/**
 * Animation Sequence Configuration
 *
 * Define sequences of animations to play with specified repetitions.
 * Each sequence item specifies a model, animation index, and how many times to repeat.
 */

const sequenceConfig = {
	// Name of this sequence (for display/identification)
	name: 'Default Training Sequence',

	// Trial metadata (sent to EMG server when recording starts)
	metadata: {
		subjectId: 'subject_001',
		sessionId: 'session_001',
		notes: '',
	},

	// Global settings
	settings: {
		// Pause duration (in seconds) between sequence items
		pauseBetweenItems: 0.5,
		// Rest duration (in seconds) between repetitions within an item
		restBetweenReps: 2.0,
		// Preparation time (in seconds) before movement starts (shows "Get Ready")
		prepTime: 1.0,
		// Whether to loop the entire sequence when finished
		loopSequence: false,
		// Playback speed multiplier (1.0 = normal)
		playbackSpeed: 1.0,
	},

	// Electrical stimulation settings.
	// When `enabled` is true, an infinite pulse train is sent to the external
	// stimulator at the start of each MOVE phase (for items with `stimulate: true`)
	// and stopped when the movement ends. Params are defined here and passed
	// straight through to the stimulator controller.
	stimulation: {
		// Master switch for automatic stimulation during movements
		enabled: false,
		// Stimulator backend: 'science_mode3' | 'science_mode4' | 'mock'
		stimulatorType: 'science_mode3',
		// Base URL of the stimulator controller (samolator)
		controllerUrl: 'http://127.0.0.1:11051',
		// Device COM port (optional; leave null to let the controller decide)
		port: null,
		// Per-channel stimulation parameters. `duration_sec` is set automatically
		// to infinite by the backend (the movement controls how long it lasts).
		channels: [
			{ id: 1, amplitude: 8.0, pulse_width: 200, frequency: 50.0, is_biphasic: true },
		],
		// TRIGGER channel for the guided session's NO-STIM training block. The
		// stimulator's ch3 is wired to the trigger input, not the patient, so during
		// that block the patient channels fire at amplitude 0 while THIS channel fires
		// normally — the trigger still records for bout segmentation, but no current
		// reaches the muscle. Set `id`/params to your actual trigger-wired channel.
		nostimTrigger: { id: 3, amplitude: 8.0, pulse_width: 200, frequency: 30.0, is_biphasic: true },
	},

	// Closed-loop stimulation for the ONLINE recognition phase. When enabled, the
	// recognized movement drives its OWN stim pattern (recognition -> stimulation).
	// `patterns` is keyed by the movement LABEL derived from the items below (the
	// model base name, or `<base>-<animation>` when a model is reused for several
	// movements). Off by default — opt in per session, and configure a pattern per
	// movement. A global EMERGENCY STOP in the session overlay halts all stim.
	onlineStimulation: {
		enabled: false,
		stimulatorType: 'science_mode3',
		controllerUrl: 'http://127.0.0.1:11051',
		port: null,
		patterns: {
			// e.g. wrist:       { channels: [{ id: 1, amplitude: 8.0, pulse_width: 200, frequency: 50.0, is_biphasic: true }] },
			//      tripodpinch: { channels: [{ id: 2, amplitude: 6.0, pulse_width: 200, frequency: 50.0, is_biphasic: true }] },
		},
	},

	// The sequence of animations to play
	// Each item: { model, animation, repetitions, restBetweenReps (optional override), stimulate (optional) }
	// - model: filename from WebGLView.AVAILABLE_MODELS (e.g., 'wrist.glb')
	// - animation: index of the animation within that model (0-based)
	// - repetitions: how many times to play this animation
	// - restBetweenReps: (optional) override the global rest duration for this item
	// - stimulate: (optional) when true and stimulation.enabled is true, deliver a
	//   pulse train during this item's MOVE phases
	items: [
		{ model: 'all.glb', animation: 0, repetitions: 3 },
		{ model: 'all.glb', animation: 1, repetitions: 2 },
		{ model: 'all.glb', animation: 2, repetitions: 2 },
		{ model: 'wrist.glb', animation: 0, repetitions: 5 },
		{ model: 'tripodpinch.glb', animation: 0, repetitions: 3 },
	],
};

export default sequenceConfig;
