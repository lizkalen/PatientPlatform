export default class TimelineView {
	constructor(app, { heightPx = 72 } = {}) {
		this.app = app;
		this.heightPx = heightPx;

		this._isScrubbing = false;
		this._wasPlayingBeforeScrub = false;

		this.keyframeTimes = [];
		this.tracks = [];
		this.selectedTrack = 'ALL';
		this.duration = 0;

		// Segment editing state
		this.segmentManager = null;
		this.isAddingSegment = false;
		this.addingSegmentType = 'movement';
		this.selectedSegmentIndex = -1;
		this._segmentDragStart = null;
		this._segmentDragHandle = null;

		this.initDOM();
		this.attachEvents();
		this.subscribeToModeChanges();
	}

	/**
	 * Subscribe to patient mode changes to auto-hide in patient mode
	 */
	subscribeToModeChanges() {
		if (this.app.modeController) {
			this.app.modeController.onModeChange((mode) => {
				this.setVisible(mode !== 'patient');
			});
		}
	}

	/**
	 * Show or hide the timeline
	 * @param {boolean} visible
	 */
	setVisible(visible) {
		this.el.style.display = visible ? '' : 'none';
		if (this.app.resize) {
			this.app.resize();
		}
	}

	initDOM() {
		this.el = document.createElement('div');
		this.el.className = 'timeline';
		this.el.style.height = `${this.heightPx}px`;

		this.el.innerHTML = `
			<div class="timeline__controls">
				<button class="timeline__btn" data-action="toggle">Play</button>

				<label class="timeline__label">
					<span>Track</span>
					<select class="timeline__track"></select>
				</label>

				<label class="timeline__label">
					<span>Speed</span>
					<input class="timeline__speed" type="range" min="0.1" max="3" step="0.1" value="1" />
					<span class="timeline__speedValue">1.0×</span>
				</label>

				<div class="timeline__legend">
					<div class="timeline__legend-item">
						<div class="timeline__legend-color timeline__legend-color--movement"></div>
						<span>Movement</span>
					</div>
					<div class="timeline__legend-item">
						<div class="timeline__legend-color timeline__legend-color--rest"></div>
						<span>Rest</span>
					</div>
				</div>

				<div class="timeline__mode-indicator" style="display: none;">
					<div class="timeline__mode-dot"></div>
					<span class="timeline__mode-text">Adding...</span>
				</div>

				<div class="timeline__time">0.00 / 0.00</div>
				<div class="timeline__kf" title="Unique keyframe times across all tracks">KF: 0</div>
			</div>

			<div class="timeline__scrub">
				<div class="timeline__segments" aria-hidden="true"></div>
				<div class="timeline__markers" aria-hidden="true"></div>
				<input class="timeline__range" type="range" min="0" max="0" step="0.001" value="0" />
			</div>
		`;

		this.btnToggle = this.el.querySelector('[data-action="toggle"]');
		this.range = this.el.querySelector('.timeline__range');
		this.trackSelect = this.el.querySelector('.timeline__track');
		this.speed = this.el.querySelector('.timeline__speed');
		this.speedValue = this.el.querySelector('.timeline__speedValue');
		this.timeEl = this.el.querySelector('.timeline__time');
		this.kfEl = this.el.querySelector('.timeline__kf');
		this.markersEl = this.el.querySelector('.timeline__markers');
		this.segmentsEl = this.el.querySelector('.timeline__segments');
		this.scrubEl = this.el.querySelector('.timeline__scrub');
		this.modeIndicator = this.el.querySelector('.timeline__mode-indicator');
		this.modeText = this.el.querySelector('.timeline__mode-text');
	}

	attachEvents() {
		this.btnToggle.addEventListener('click', () => {
			const webgl = this.app.webgl;
			if (!webgl) return;

			if (webgl.isPlaying) webgl.pause();
			else webgl.play();

			this.syncControlsFromWebGL();
		});

		this.trackSelect.addEventListener('change', () => {
			this.selectedTrack = this.trackSelect.value;
			this.updateMarkersForSelection();
		});

		this.speed.addEventListener('input', () => {
			const webgl = this.app.webgl;
			if (!webgl) return;

			const v = Number(this.speed.value);
			webgl.playbackSpeed = v;
			this.speedValue.textContent = `${v.toFixed(1)}×`;
		});

		// Scrubbing behavior: pause while scrubbing, optionally resume.
		const onScrubStart = () => {
			const webgl = this.app.webgl;
			if (!webgl) return;

			this._isScrubbing = true;
			this._wasPlayingBeforeScrub = !!webgl.isPlaying;
			webgl.pause();
			this.syncControlsFromWebGL();
		};

		const onScrubEnd = () => {
			const webgl = this.app.webgl;
			if (!webgl) return;

			this._isScrubbing = false;
			if (this._wasPlayingBeforeScrub) webgl.play();
			this.syncControlsFromWebGL();
		};

		this.range.addEventListener('pointerdown', onScrubStart);
		window.addEventListener('pointerup', onScrubEnd);

		this.range.addEventListener('input', () => {
			const webgl = this.app.webgl;
			if (!webgl) return;
			webgl.setAnimationTime(Number(this.range.value));
			this.updateTimeLabel(Number(this.range.value), this.duration);
		});
	}

	syncControlsFromWebGL() {
		const webgl = this.app.webgl;
		if (!webgl) return;

		this.btnToggle.textContent = webgl.isPlaying ? 'Pause' : 'Play';

		const v = Number(webgl.playbackSpeed ?? 1);
		this.speed.value = String(v);
		this.speedValue.textContent = `${v.toFixed(1)}×`;
	}

	setClipInfo({ duration = 0, keyframeTimes = [], tracks = [] } = {}) {
		this.duration = duration;
		this.keyframeTimes = Array.isArray(keyframeTimes) ? keyframeTimes : [];
		this.tracks = Array.isArray(tracks) ? tracks : [];

		this.range.max = String(duration || 0);
		this.populateTrackSelect();
		this.updateMarkersForSelection();
		this.update(0);
	}

	populateTrackSelect() {
		if (!this.trackSelect) return;
		this.trackSelect.innerHTML = '';

		const addOption = (value, label) => {
			const opt = document.createElement('option');
			opt.value = value;
			opt.textContent = label;
			this.trackSelect.appendChild(opt);
		};

		addOption('ALL', `All (${this.tracks.length})`);
		for (const t of this.tracks) {
			addOption(t.name, t.name);
		}

		// Keep current selection if possible.
		const exists = Array.from(this.trackSelect.options).some(o => o.value === this.selectedTrack);
		this.trackSelect.value = exists ? this.selectedTrack : 'ALL';
		this.selectedTrack = this.trackSelect.value;
	}

	updateMarkersForSelection() {
		let times;
		if (this.selectedTrack === 'ALL') {
			times = this.keyframeTimes;
		} else {
			const track = this.tracks.find(t => t.name === this.selectedTrack);
			times = track?.times || [];
			// Dedupe within the track (typed arrays can include repeats across segments).
			times = this.dedupeTimes(times);
		}
		if (this.kfEl) this.kfEl.textContent = `KF: ${times?.length || 0}`;
		this.renderMarkers(times);
	}

	dedupeTimes(times, { epsilon = 1e-4 } = {}) {
		if (!times || times.length === 0) return [];
		// times should already be sorted; if not, copy+sort.
		let sorted = times;
		for (let i = 1; i < times.length; i++) {
			if (times[i] < times[i - 1]) {
				sorted = Array.from(times).sort((a, b) => a - b);
				break;
			}
		}
		const out = [sorted[0]];
		for (let i = 1; i < sorted.length; i++) {
			const t = sorted[i];
			if (Math.abs(t - out[out.length - 1]) > epsilon) out.push(t);
		}
		return out;
	}

	renderMarkers(keyframeTimes) {
		this.markersEl.innerHTML = '';
		const timesIn = keyframeTimes || [];
		if (!this.duration || !timesIn.length) return;

		// Cap markers to keep DOM reasonable.
		const MAX_MARKERS = 2000;
		let times = timesIn;
		if (times.length > MAX_MARKERS) {
			const step = Math.ceil(times.length / MAX_MARKERS);
			times = times.filter((_, i) => i % step === 0);
			console.warn(`[Timeline] Too many keyframes (${timesIn.length}); showing ${times.length} markers.`);
		}

		const frag = document.createDocumentFragment();
		for (const t of times) {
			const x = Math.min(100, Math.max(0, (t / this.duration) * 100));
			const m = document.createElement('div');
			m.className = 'timeline__marker';
			m.style.left = `${x}%`;
			frag.appendChild(m);
		}
		this.markersEl.appendChild(frag);
	}

	update(currentTime) {
		if (!this._isScrubbing) {
			this.range.value = String(currentTime || 0);
		}
		this.updateTimeLabel(currentTime || 0, this.duration);
		this.syncControlsFromWebGL();
	}

	updateTimeLabel(t, duration) {
		const a = Number.isFinite(t) ? t : 0;
		const b = Number.isFinite(duration) ? duration : 0;
		this.timeEl.textContent = `${a.toFixed(2)} / ${b.toFixed(2)}`;
	}

	// -----------------------------------------------------------------------------------------
	// SEGMENT MANAGEMENT
	// -----------------------------------------------------------------------------------------

	/**
	 * Set the segment manager and subscribe to changes
	 * @param {SegmentManager} manager
	 */
	setSegmentManager(manager) {
		this.segmentManager = manager;
		if (manager) {
			manager.onChange(() => this.renderSegments());
			this.renderSegments();
			this.attachSegmentEvents();
		}
	}

	/**
	 * Attach segment-related event handlers
	 */
	attachSegmentEvents() {
		// Handle segment creation via click-drag on segments area
		this.segmentsEl.addEventListener('pointerdown', (e) => this._onSegmentAreaPointerDown(e));
		window.addEventListener('pointermove', (e) => this._onSegmentAreaPointerMove(e));
		window.addEventListener('pointerup', (e) => this._onSegmentAreaPointerUp(e));
	}

	/**
	 * Enable/disable segment adding mode
	 * @param {boolean} enabled
	 * @param {string} type - 'movement' or 'rest'
	 */
	setAddingSegment(enabled, type = 'movement') {
		this.isAddingSegment = enabled;
		this.addingSegmentType = type;

		// Update scrub area classes
		this.scrubEl.classList.toggle('timeline__scrub--adding', enabled);
		this.scrubEl.classList.toggle('timeline__scrub--adding-movement', enabled && type === 'movement');
		this.scrubEl.classList.toggle('timeline__scrub--adding-rest', enabled && type === 'rest');

		// Update mode indicator
		if (enabled) {
			this.modeIndicator.style.display = 'flex';
			this.modeIndicator.className = `timeline__mode-indicator timeline__mode-indicator--${type}`;
			this.modeText.textContent = type === 'movement' ? 'Click & drag to add MOVEMENT' : 'Click & drag to add REST';
		} else {
			this.modeIndicator.style.display = 'none';
		}
	}

	/**
	 * Convert pixel X position to time
	 * @param {number} clientX
	 * @returns {number}
	 */
	_xToTime(clientX) {
		const rect = this.segmentsEl.getBoundingClientRect();
		const x = Math.max(0, Math.min(clientX - rect.left, rect.width));
		const ratio = x / rect.width;
		return ratio * this.duration;
	}

	/**
	 * Handle pointer down on segment area
	 */
	_onSegmentAreaPointerDown(e) {
		if (!this.segmentManager || !this.duration) return;

		const target = e.target;

		// Check if clicking on a segment handle
		if (target.classList.contains('timeline__segment-handle')) {
			e.preventDefault();
			e.stopPropagation();
			const segmentEl = target.closest('.timeline__segment');
			const index = parseInt(segmentEl.dataset.index, 10);
			const isStart = target.classList.contains('timeline__segment-handle--start');

			this._segmentDragHandle = { index, isStart };
			this.selectedSegmentIndex = index;
			this.renderSegments();
			return;
		}

		// Check if clicking on a segment (to select it)
		if (target.classList.contains('timeline__segment')) {
			const index = parseInt(target.dataset.index, 10);
			this.selectedSegmentIndex = this.selectedSegmentIndex === index ? -1 : index;
			this.renderSegments();
			return;
		}

		// If in adding mode, start creating a new segment
		if (this.isAddingSegment) {
			e.preventDefault();
			const time = this._xToTime(e.clientX);
			this._segmentDragStart = time;
			this._showSegmentPreview(time, time);
		}
	}

	/**
	 * Handle pointer move for segment creation/editing
	 */
	_onSegmentAreaPointerMove(e) {
		if (!this.segmentManager || !this.duration) return;

		// Handle segment handle dragging
		if (this._segmentDragHandle) {
			const { index, isStart } = this._segmentDragHandle;
			const time = this._xToTime(e.clientX);
			const segment = this.segmentManager.segments[index];

			if (isStart) {
				this.segmentManager.updateSegment(index, { start: time });
			} else {
				this.segmentManager.updateSegment(index, { end: time });
			}
			return;
		}

		// Handle segment creation preview
		if (this._segmentDragStart !== null) {
			const endTime = this._xToTime(e.clientX);
			this._showSegmentPreview(this._segmentDragStart, endTime);
		}
	}

	/**
	 * Handle pointer up for segment creation/editing
	 */
	_onSegmentAreaPointerUp(e) {
		// Finish handle dragging
		if (this._segmentDragHandle) {
			this._segmentDragHandle = null;
			return;
		}

		// Finish segment creation
		if (this._segmentDragStart !== null && this.segmentManager) {
			const endTime = this._xToTime(e.clientX);
			const start = Math.min(this._segmentDragStart, endTime);
			const end = Math.max(this._segmentDragStart, endTime);

			// Only create if segment has meaningful duration
			if (end - start > 0.01) {
				this.segmentManager.addSegment(start, end, this.addingSegmentType);
			}

			this._segmentDragStart = null;
			this._hideSegmentPreview();
		}
	}

	/**
	 * Show preview segment while dragging
	 */
	_showSegmentPreview(start, end) {
		let preview = this.segmentsEl.querySelector('.timeline__segment-preview');
		if (!preview) {
			preview = document.createElement('div');
			preview.className = `timeline__segment-preview timeline__segment-preview--${this.addingSegmentType}`;
			this.segmentsEl.appendChild(preview);
		}

		// Update type class if it changed
		preview.className = `timeline__segment-preview timeline__segment-preview--${this.addingSegmentType}`;

		const left = Math.min(start, end) / this.duration * 100;
		const width = Math.abs(end - start) / this.duration * 100;
		preview.style.left = `${left}%`;
		preview.style.width = `${width}%`;
	}

	/**
	 * Hide segment preview
	 */
	_hideSegmentPreview() {
		const preview = this.segmentsEl.querySelector('.timeline__segment-preview');
		if (preview) preview.remove();
	}

	/**
	 * Render all segments
	 */
	renderSegments() {
		if (!this.segmentsEl) return;
		this.segmentsEl.innerHTML = '';

		if (!this.segmentManager || !this.duration) return;

		const frag = document.createDocumentFragment();

		this.segmentManager.segments.forEach((segment, index) => {
			const left = (segment.start / this.duration) * 100;
			const width = ((segment.end - segment.start) / this.duration) * 100;

			const el = document.createElement('div');
			el.className = `timeline__segment timeline__segment--${segment.type}`;
			if (index === this.selectedSegmentIndex) {
				el.classList.add('timeline__segment--selected');
			}
			el.dataset.index = index;
			el.style.left = `${left}%`;
			el.style.width = `${width}%`;

			// Add resize handles
			const handleStart = document.createElement('div');
			handleStart.className = 'timeline__segment-handle timeline__segment-handle--start';
			el.appendChild(handleStart);

			const handleEnd = document.createElement('div');
			handleEnd.className = 'timeline__segment-handle timeline__segment-handle--end';
			el.appendChild(handleEnd);

			frag.appendChild(el);
		});

		this.segmentsEl.appendChild(frag);
	}

	/**
	 * Delete the currently selected segment
	 */
	deleteSelectedSegment() {
		if (this.selectedSegmentIndex >= 0 && this.segmentManager) {
			this.segmentManager.removeSegment(this.selectedSegmentIndex);
			this.selectedSegmentIndex = -1;
		}
	}

	/**
	 * Toggle the type of the selected segment
	 */
	toggleSelectedSegmentType() {
		if (this.selectedSegmentIndex >= 0 && this.segmentManager) {
			const segment = this.segmentManager.segments[this.selectedSegmentIndex];
			const newType = segment.type === 'movement' ? 'rest' : 'movement';
			this.segmentManager.updateSegment(this.selectedSegmentIndex, { type: newType });
		}
	}
}
