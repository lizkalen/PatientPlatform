/**
 * Patient Mode Controller
 * =======================
 *
 * Central controller for managing Patient/Config mode switching.
 * Notifies subscribed components when mode changes so they can
 * show/hide themselves appropriately.
 *
 * Modes:
 * - 'config': Full admin interface (Tweakpane, Timeline, Segment editor)
 * - 'patient': Simplified patient view (3D viewport, Progress sidebar, Phase overlay)
 */

export const MODE = {
	CONFIG: 'config',
	PATIENT: 'patient',
};

export default class PatientModeController {
	constructor(app) {
		this.app = app;
		this._currentMode = MODE.CONFIG;
		this._listeners = [];

		// Keyboard shortcut
		this._handleKeydown = this._handleKeydown.bind(this);
		window.addEventListener('keydown', this._handleKeydown);

		// Set initial body class
		this._updateBodyClass();
	}

	/**
	 * Get the current mode
	 * @returns {'config' | 'patient'}
	 */
	getMode() {
		return this._currentMode;
	}

	/**
	 * Check if currently in patient mode
	 * @returns {boolean}
	 */
	isPatientMode() {
		return this._currentMode === MODE.PATIENT;
	}

	/**
	 * Check if currently in config mode
	 * @returns {boolean}
	 */
	isConfigMode() {
		return this._currentMode === MODE.CONFIG;
	}

	/**
	 * Set the current mode
	 * @param {'config' | 'patient'} mode
	 */
	setMode(mode) {
		if (mode === this._currentMode) return;
		if (mode !== MODE.CONFIG && mode !== MODE.PATIENT) {
			console.warn(`PatientModeController: Invalid mode '${mode}'`);
			return;
		}

		const previousMode = this._currentMode;
		this._currentMode = mode;

		this._updateBodyClass();
		this._notifyListeners(mode, previousMode);

		// Trigger resize after mode change (for viewport adjustments)
		if (this.app?.resize) {
			// Small delay to allow CSS transitions to start
			requestAnimationFrame(() => {
				this.app.resize();
			});
		}

		console.log(`PatientModeController: Mode changed to '${mode}'`);
	}

	/**
	 * Re-broadcast the current mode to all listeners without changing it.
	 *
	 * `setMode()` early-returns when the mode is unchanged, so it cannot be used to
	 * align subscribers at boot — every component would be left relying on its own
	 * constructor defaults happening to match the initial mode. Call this once after
	 * all subscribers are constructed to put them in a known state.
	 */
	syncMode() {
		this._updateBodyClass();
		this._notifyListeners(this._currentMode, this._currentMode);
	}

	/**
	 * Toggle between patient and config modes
	 */
	toggleMode() {
		this.setMode(this._currentMode === MODE.CONFIG ? MODE.PATIENT : MODE.CONFIG);
	}

	/**
	 * Subscribe to mode changes
	 * @param {Function} callback - Called with (newMode, previousMode)
	 * @returns {Function} Unsubscribe function
	 */
	onModeChange(callback) {
		this._listeners.push(callback);

		// Return unsubscribe function
		return () => {
			const index = this._listeners.indexOf(callback);
			if (index > -1) {
				this._listeners.splice(index, 1);
			}
		};
	}

	/**
	 * Update body class based on current mode
	 */
	_updateBodyClass() {
		document.body.classList.remove('config-mode', 'patient-mode');
		document.body.classList.add(`${this._currentMode}-mode`);
	}

	/**
	 * Notify all listeners of mode change
	 */
	_notifyListeners(newMode, previousMode) {
		for (const listener of this._listeners) {
			try {
				listener(newMode, previousMode);
			} catch (err) {
				console.error('PatientModeController: Error in listener', err);
			}
		}
	}

	/**
	 * Handle keyboard shortcuts
	 */
	_handleKeydown(e) {
		// Ctrl+M to toggle mode
		if (e.ctrlKey && e.key === 'm') {
			e.preventDefault();
			this.toggleMode();
		}
	}

	/**
	 * Cleanup
	 */
	destroy() {
		window.removeEventListener('keydown', this._handleKeydown);
		this._listeners = [];
	}
}
