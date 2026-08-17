/**
 * Sequence Config Store
 * =====================
 *
 * Persistence for the sequence config the operator edits in SequenceConfigPanel.
 *
 * There is ONE live config: the one loaded into SequencePlayer. This module is
 * only its durable backing store, so a reload restores what was last applied
 * instead of silently reverting to the static default in `sequenceConfig.js`.
 *
 * Read it at boot (App.initSequencePlayer) and write it on Apply. Everything
 * downstream — the panel, the movement session, the patient view — should read
 * `sequencePlayer.config`, never this module directly.
 */

const STORAGE_KEY = 'sequenceConfig';

/**
 * Fill in the structural fields the editor and player assume exist.
 * Applied on load and on import, so a hand-written or older JSON can't reach
 * the rest of the app half-formed.
 *
 * @param {object} config
 * @returns {object} the same object, mutated
 */
export function normalizeConfig(config) {
	if (!config || typeof config !== 'object') return config;

	if (!config.metadata) config.metadata = {};
	if (!config.settings) config.settings = {};
	if (!Array.isArray(config.items)) config.items = [];
	if (!config.stimulation) config.stimulation = {};
	if (!Array.isArray(config.stimulation.channels)) {
		config.stimulation.channels = [
			{ id: 1, amplitude: 8.0, pulse_width: 200, frequency: 50.0, is_biphasic: true },
		];
	}

	return config;
}

/**
 * The last applied config, or null if none was ever saved (or it's unreadable).
 * @returns {object|null}
 */
export function loadSavedConfig() {
	try {
		const saved = localStorage.getItem(STORAGE_KEY);
		if (!saved) return null;
		return normalizeConfig(JSON.parse(saved));
	} catch (e) {
		console.warn('configStore: failed to load saved config, using defaults:', e);
		return null;
	}
}

/**
 * Persist a config as the one to restore on next load.
 * @param {object} config
 */
export function saveConfig(config) {
	try {
		localStorage.setItem(STORAGE_KEY, JSON.stringify(config));
	} catch (e) {
		console.warn('configStore: failed to save config:', e);
	}
}

/** Forget the saved config; the next load falls back to the static default. */
export function clearSavedConfig() {
	try {
		localStorage.removeItem(STORAGE_KEY);
	} catch (e) {
		console.warn('configStore: failed to clear saved config:', e);
	}
}
