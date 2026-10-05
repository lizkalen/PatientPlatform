/**
 * Progress Sidebar Component
 * ==========================
 *
 * Displays exercise progress for patients during a sequence.
 * Shows:
 * - List of all sequence items with status (pending/current/completed)
 * - Current repetition count
 * - Play/Stop controls
 * - EMG view access button
 *
 * Only visible in patient mode.
 */

import { MODE } from './PatientModeController';
import { movementName } from '../config/movements';

export default class ProgressSidebar {
	constructor(app) {
		this.app = app;
		this.sequencePlayer = app.sequencePlayer;
		this.modeController = app.modeController;

		// Track item states
		this.itemStates = [];
		this.currentItemIndex = -1;
		this.currentRepetition = 0;
		this.totalRepetitions = 0;
		this.isPlaying = false;

		this._createElements();
		this._subscribeToSequencePlayer();
		this._subscribeToModeChanges();

		// Initialize display
		this._updateDisplay();
	}

	_createElements() {
		// Main container
		this.el = document.createElement('div');
		this.el.className = 'progress-sidebar';

		// Header
		const header = document.createElement('div');
		header.className = 'progress-sidebar__header';
		header.innerHTML = '<h2>Exercise Progress</h2>';
		this.el.appendChild(header);

		// Item list container
		this.listEl = document.createElement('div');
		this.listEl.className = 'progress-sidebar__list';
		this.el.appendChild(this.listEl);

		// Footer with controls
		const footer = document.createElement('div');
		footer.className = 'progress-sidebar__footer';

		// Play/Stop button
		this.playButton = document.createElement('button');
		this.playButton.className = 'progress-sidebar__play-button';
		this.playButton.textContent = 'Start Exercises';
		this.playButton.addEventListener('click', () => this._onPlayClick());
		footer.appendChild(this.playButton);

		// EMG button
		this.emgButton = document.createElement('button');
		this.emgButton.className = 'progress-sidebar__emg-button';
		this.emgButton.innerHTML = `
			<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
				<path d="M22 12h-4l-3 9L9 3l-3 9H2"/>
			</svg>
			View EMG Data
		`;
		this.emgButton.addEventListener('click', () => this._onEMGClick());
		footer.appendChild(this.emgButton);

		this.el.appendChild(footer);

		document.body.appendChild(this.el);
	}

	_subscribeToSequencePlayer() {
		if (!this.sequencePlayer) return;

		// Subscribe to state changes
		this.sequencePlayer.onChange((state) => {
			this._onSequenceStateChange(state);
		});

		// Subscribe to sequence completion
		this.sequencePlayer.onSequenceComplete(() => {
			this._onSequenceComplete();
		});
	}

	_subscribeToModeChanges() {
		if (!this.modeController) return;

		this.modeController.onModeChange((mode) => {
			// Sidebar visibility is handled by CSS based on body class
			// But we can do additional updates here if needed
			if (mode === MODE.PATIENT) {
				this._refreshItemList();
			}
		});
	}

	_onSequenceStateChange(state) {
		this.isPlaying = state.isPlaying;
		this.currentItemIndex = state.currentItemIndex;
		this.currentRepetition = state.currentRepetition;
		this.totalRepetitions = state.totalRepetitions;

		this._updateDisplay();
	}

	_onSequenceComplete() {
		this.isPlaying = false;
		this._updateDisplay();
	}

	_onPlayClick() {
		if (!this.sequencePlayer) return;

		if (this.isPlaying) {
			this.sequencePlayer.stop();
		} else {
			this.sequencePlayer.play();
		}
	}

	_onEMGClick() {
		if (this.app.emgChannelView) {
			this.app.emgChannelView.show();
		}
	}

	_updateDisplay() {
		this._updateItemList();
		this._updatePlayButton();
	}

	_refreshItemList() {
		// Reset and rebuild the item list from current config
		this._updateItemList();
	}

	_updateItemList() {
		this.listEl.innerHTML = '';

		const config = this.sequencePlayer?.config;
		if (!config || !config.items || config.items.length === 0) {
			this.listEl.innerHTML = '<div style="padding: 20px; opacity: 0.5; text-align: center;">No exercises configured</div>';
			return;
		}

		config.items.forEach((item, index) => {
			const itemEl = this._createItemElement(item, index);
			this.listEl.appendChild(itemEl);
		});
	}

	_createItemElement(item, index) {
		const el = document.createElement('div');
		el.className = 'progress-sidebar__item';

		// Determine status
		let status = 'pending';
		let repText = `0/${item.repetitions}`;

		if (index < this.currentItemIndex) {
			status = 'completed';
			repText = `${item.repetitions}/${item.repetitions}`;
		} else if (index === this.currentItemIndex && this.isPlaying) {
			status = 'current';
			repText = `${this.currentRepetition + 1}/${this.totalRepetitions}`;
		} else if (index === this.currentItemIndex && !this.isPlaying && this.currentRepetition > 0) {
			// Paused mid-item
			status = 'current';
			repText = `${this.currentRepetition}/${item.repetitions}`;
		}

		el.classList.add(`progress-sidebar__item--${status}`);

		// Icon
		const icon = document.createElement('div');
		icon.className = `progress-sidebar__item-icon progress-sidebar__item-icon--${status}`;
		if (status === 'completed') {
			icon.innerHTML = '&#10003;'; // Checkmark
		} else if (status === 'current') {
			icon.innerHTML = '&#9654;'; // Play arrow
		}
		el.appendChild(icon);

		// Info
		const info = document.createElement('div');
		info.className = 'progress-sidebar__item-info';

		const name = document.createElement('div');
		name.className = 'progress-sidebar__item-name';
		name.textContent = this._getDisplayName(item);
		info.appendChild(name);

		const reps = document.createElement('div');
		reps.className = 'progress-sidebar__item-reps';
		reps.textContent = `Rep ${repText}`;
		info.appendChild(reps);

		el.appendChild(info);

		return el;
	}

	_getDisplayName(item) {
		return movementName(item, 'Unknown Exercise');
	}

	_updatePlayButton() {
		if (this.isPlaying) {
			this.playButton.textContent = 'Stop';
			this.playButton.classList.add('progress-sidebar__play-button--stop');
		} else {
			this.playButton.textContent = 'Start Exercises';
			this.playButton.classList.remove('progress-sidebar__play-button--stop');
		}
	}

	show() {
		this.el.style.display = 'flex';
	}

	hide() {
		this.el.style.display = 'none';
	}

	destroy() {
		this.el.remove();
	}
}
