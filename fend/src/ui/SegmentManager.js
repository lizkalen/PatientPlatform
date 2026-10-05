import segmentConfig from '../config/segmentConfig.js';

/**
 * Manages animation segments (movement/rest) with independent speed controls.
 * Handles persistence via LocalStorage and JSON export/import.
 * Falls back to segmentConfig defaults when no localStorage data exists.
 */
export default class SegmentManager {
	constructor() {
		// Array of { start: number, end: number, type: 'movement'|'rest' }
		this.segments = [];

		// Speed multipliers for each segment type
		this.speeds = {
			movement: 1.0,
			rest: 3.0,
		};

		// Current model/animation context for storage
		this.currentModelFile = null;
		this.currentAnimationIndex = 0;

		// Callbacks for UI updates
		this.onChangeCallbacks = [];
	}

	/**
	 * Register a callback to be called when segments change
	 * @param {Function} callback
	 */
	onChange(callback) {
		this.onChangeCallbacks.push(callback);
	}

	/**
	 * Notify all registered callbacks of a change
	 */
	_notifyChange() {
		for (const cb of this.onChangeCallbacks) {
			cb(this.segments, this.speeds);
		}
	}

	/**
	 * Set the current model/animation context
	 * @param {string} modelFile
	 * @param {number} animationIndex
	 */
	setContext(modelFile, animationIndex = 0) {
		this.currentModelFile = modelFile;
		this.currentAnimationIndex = animationIndex;
	}

	/**
	 * Add a new segment
	 * @param {number} start - Start time in seconds
	 * @param {number} end - End time in seconds
	 * @param {string} type - 'movement' or 'rest'
	 * @returns {number} Index of the new segment
	 */
	addSegment(start, end, type = 'movement') {
		// Ensure start < end
		if (start > end) {
			[start, end] = [end, start];
		}

		const segment = { start, end, type };
		this.segments.push(segment);

		// Sort by start time
		this.segments.sort((a, b) => a.start - b.start);

		this._autoSave();
		this._notifyChange();

		return this.segments.indexOf(segment);
	}

	/**
	 * Remove a segment by index
	 * @param {number} index
	 */
	removeSegment(index) {
		if (index >= 0 && index < this.segments.length) {
			this.segments.splice(index, 1);
			this._autoSave();
			this._notifyChange();
		}
	}

	/**
	 * Update a segment's properties
	 * @param {number} index
	 * @param {object} props - { start?, end?, type? }
	 */
	updateSegment(index, props) {
		if (index >= 0 && index < this.segments.length) {
			const segment = this.segments[index];

			if (props.start !== undefined) segment.start = props.start;
			if (props.end !== undefined) segment.end = props.end;
			if (props.type !== undefined) segment.type = props.type;

			// Ensure start < end
			if (segment.start > segment.end) {
				[segment.start, segment.end] = [segment.end, segment.start];
			}

			// Re-sort by start time
			this.segments.sort((a, b) => a.start - b.start);

			this._autoSave();
			this._notifyChange();
		}
	}

	/**
	 * Clear all segments
	 */
	clearSegments() {
		this.segments = [];
		this._autoSave();
		this._notifyChange();
	}

	/**
	 * Get the segment at a specific time
	 * @param {number} time
	 * @returns {object|null} The segment or null if not in any segment
	 */
	getSegmentAtTime(time) {
		for (const segment of this.segments) {
			if (time >= segment.start && time <= segment.end) {
				return segment;
			}
		}
		return null;
	}

	/**
	 * Get the speed multiplier for a specific time
	 * @param {number} time
	 * @returns {number} Speed multiplier (1.0 if not in any segment)
	 */
	getSpeedAtTime(time) {
		const segment = this.getSegmentAtTime(time);
		if (segment) {
			return this.speeds[segment.type] ?? 1.0;
		}
		return 1.0; // Default speed when not in a segment
	}

	/**
	 * Set speed for a segment type
	 * @param {string} type - 'movement' or 'rest'
	 * @param {number} speed - Speed multiplier
	 */
	setSpeed(type, speed) {
		if (type === 'movement' || type === 'rest') {
			this.speeds[type] = speed;
			this._autoSave();
			this._notifyChange();
		}
	}

	/**
	 * Generate a unique storage key for current context
	 * @returns {string}
	 */
	_getStorageKey() {
		if (!this.currentModelFile) return null;
		return `anim-segments:${this.currentModelFile}:${this.currentAnimationIndex}`;
	}

	/**
	 * Auto-save to localStorage if context is set
	 */
	_autoSave() {
		const key = this._getStorageKey();
		if (!key) return;

		try {
			const data = {
				segments: this.segments,
				speeds: this.speeds,
			};
			localStorage.setItem(key, JSON.stringify(data));
		} catch (e) {
			console.warn('Failed to save segments to localStorage:', e);
		}
	}

	/**
	 * Save current configuration to localStorage
	 * @returns {boolean} Success
	 */
	save() {
		this._autoSave();
		return true;
	}

	/**
	 * Load configuration from localStorage for current context.
	 * Falls back to segmentConfig defaults if no localStorage data exists.
	 * @returns {boolean} True if data was loaded
	 */
	load() {
		const key = this._getStorageKey();
		if (!key) return false;

		try {
			const json = localStorage.getItem(key);

			if (json) {
				// Load from localStorage
				const data = JSON.parse(json);

				if (Array.isArray(data.segments)) {
					this.segments = data.segments;
				}
				if (data.speeds) {
					if (typeof data.speeds.movement === 'number') {
						this.speeds.movement = data.speeds.movement;
					}
					if (typeof data.speeds.rest === 'number') {
						this.speeds.rest = data.speeds.rest;
					}
				}

				this._notifyChange();
				return true;
			} else {
				// Fall back to segmentConfig defaults
				return this._loadFromConfig();
			}
		} catch (e) {
			console.warn('Failed to load segments from localStorage:', e);
			// Try config as fallback
			return this._loadFromConfig();
		}
	}

	/**
	 * Load default segments from segmentConfig
	 * @returns {boolean} True if defaults were loaded
	 */
	_loadFromConfig() {
		if (!this.currentModelFile) return false;

		const configSegments = segmentConfig.getSegments(
			this.currentModelFile,
			this.currentAnimationIndex
		);

		if (configSegments && configSegments.length > 0) {
			// Convert config format to internal format
			// Config uses: { name, start, end }
			// Internal uses: { start, end, type }
			this.segments = configSegments.map((seg) => ({
				start: seg.start,
				end: seg.end,
				type: seg.name.toLowerCase() === 'movement' ? 'movement' : 'rest',
			}));

			this._notifyChange();
			console.log(`Loaded default segments for ${this.currentModelFile}:${this.currentAnimationIndex}`);
			return true;
		}

		return false;
	}

	/**
	 * Export current configuration as JSON string
	 * @returns {string}
	 */
	exportJSON() {
		const data = {
			modelFile: this.currentModelFile,
			animationIndex: this.currentAnimationIndex,
			segments: this.segments,
			speeds: this.speeds,
			exportedAt: new Date().toISOString(),
		};
		return JSON.stringify(data, null, 2);
	}

	/**
	 * Import configuration from JSON string
	 * @param {string} json
	 * @returns {boolean} Success
	 */
	importJSON(json) {
		try {
			const data = JSON.parse(json);

			if (Array.isArray(data.segments)) {
				this.segments = data.segments;
			}
			if (data.speeds) {
				if (typeof data.speeds.movement === 'number') {
					this.speeds.movement = data.speeds.movement;
				}
				if (typeof data.speeds.rest === 'number') {
					this.speeds.rest = data.speeds.rest;
				}
			}

			this._autoSave();
			this._notifyChange();
			return true;
		} catch (e) {
			console.error('Failed to import segments JSON:', e);
			return false;
		}
	}

	/**
	 * Download configuration as a JSON file
	 */
	downloadJSON() {
		const json = this.exportJSON();
		const blob = new Blob([json], { type: 'application/json' });
		const url = URL.createObjectURL(blob);

		const a = document.createElement('a');
		a.href = url;
		a.download = `segments-${this.currentModelFile || 'unknown'}-${this.currentAnimationIndex}.json`;
		a.click();

		URL.revokeObjectURL(url);
	}
}
