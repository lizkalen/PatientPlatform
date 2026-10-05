/**
 * Phase Overlay Component
 * =======================
 *
 * Displays visual cues for the current sequence phase:
 * - IDLE: Hidden
 * - PREP: Yellow "Get Ready" with countdown
 * - MOVE: Green "Move" indicator
 * - REST: Red "Hold" with countdown
 * - TRANSITION: Gray "Next..." indicator
 *
 * Also shows a traffic light style indicator.
 */

import { PHASE } from './SequencePlayer';
import { movementName } from '../config/movements';

export default class PhaseOverlay {
	constructor(sequencePlayer) {
		this.sequencePlayer = sequencePlayer;
		this.currentPhase = PHASE.IDLE;
		this.timeRemaining = 0;

		// Phase configurations
		this.phaseConfig = {
			[PHASE.IDLE]: {
				text: '',
				color: 'transparent',
				bgColor: 'transparent',
				light: 'none',
				showCountdown: false,
			},
			[PHASE.TUTORIAL]: {
				text: 'WATCH & LEARN',
				color: '#ffffff',
				bgColor: '#6366f1',
				light: 'none',
				showCountdown: false,
			},
			[PHASE.PREP]: {
				text: 'GET READY',
				color: '#000000',
				bgColor: '#ffd700',
				light: 'yellow',
				showCountdown: true,
			},
			[PHASE.MOVE]: {
				text: 'HOLD',
				color: '#ffffff',
				bgColor: '#00cc44',
				light: 'green',
				showCountdown: false,
			},
			[PHASE.REST]: {
				text: 'REST',
				color: '#ffffff',
				bgColor: '#cc0000',
				light: 'red',
				showCountdown: true,
			},
			[PHASE.TRANSITION]: {
				text: 'NEXT...',
				color: '#ffffff',
				bgColor: '#666666',
				light: 'none',
				showCountdown: true,
			},
		};

		this._createElements();
		this._subscribeToPlayer();
	}

	_createElements() {
		// Main overlay container (centered at top of screen)
		this.overlay = document.createElement('div');
		this.overlay.className = 'phase-overlay';
		this.overlay.style.cssText = `
			position: fixed;
			top: 20px;
			left: 50%;
			transform: translateX(-50%);
			z-index: 500;
			display: flex;
			flex-direction: column;
			align-items: center;
			gap: 15px;
			pointer-events: none;
			transition: opacity 0.3s ease;
		`;

		// Traffic light container
		this.trafficLight = document.createElement('div');
		this.trafficLight.className = 'traffic-light';
		this.trafficLight.style.cssText = `
			display: flex;
			flex-direction: column;
			gap: 8px;
			padding: 15px 12px;
			background: #1a1a1a;
			border-radius: 30px;
			box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
		`;

		// Create traffic light bulbs
		this.lights = {
			red: this._createLight('#cc0000', '#330000'),
			yellow: this._createLight('#ffd700', '#332b00'),
			green: this._createLight('#00cc44', '#003311'),
		};

		this.trafficLight.appendChild(this.lights.red);
		this.trafficLight.appendChild(this.lights.yellow);
		this.trafficLight.appendChild(this.lights.green);

		// Phase text container
		this.textContainer = document.createElement('div');
		this.textContainer.style.cssText = `
			display: flex;
			flex-direction: column;
			align-items: center;
			gap: 5px;
		`;

		// Phase text badge
		this.phaseBadge = document.createElement('div');
		this.phaseBadge.style.cssText = `
			padding: 12px 30px;
			border-radius: 8px;
			font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
			font-size: 28px;
			font-weight: bold;
			letter-spacing: 2px;
			text-transform: uppercase;
			box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
			transition: all 0.3s ease;
		`;

		// Countdown text
		this.countdownText = document.createElement('div');
		this.countdownText.style.cssText = `
			font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
			font-size: 48px;
			font-weight: bold;
			color: white;
			text-shadow: 0 2px 10px rgba(0, 0, 0, 0.5);
		`;

		// Progress info (rep count, item)
		this.progressText = document.createElement('div');
		this.progressText.style.cssText = `
			font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
			font-size: 14px;
			color: rgba(255, 255, 255, 0.8);
			background: rgba(0, 0, 0, 0.5);
			padding: 6px 16px;
			border-radius: 20px;
		`;

		this.textContainer.appendChild(this.phaseBadge);
		this.textContainer.appendChild(this.countdownText);
		this.textContainer.appendChild(this.progressText);

		this.overlay.appendChild(this.trafficLight);
		this.overlay.appendChild(this.textContainer);

		document.body.appendChild(this.overlay);

		// Initially hidden
		this._updateDisplay(PHASE.IDLE, 0);
	}

	_createLight(activeColor, inactiveColor) {
		const light = document.createElement('div');
		light.style.cssText = `
			width: 35px;
			height: 35px;
			border-radius: 50%;
			background: ${inactiveColor};
			box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.3);
			transition: all 0.3s ease;
		`;
		light.dataset.activeColor = activeColor;
		light.dataset.inactiveColor = inactiveColor;
		return light;
	}

	_subscribeToPlayer() {
		this.sequencePlayer.onPhaseChange((phase, timeRemaining) => {
			this._updateDisplay(phase, timeRemaining);
		});

		this.sequencePlayer.onChange((state) => {
			this._updateProgress(state);
		});
	}

	_updateDisplay(phase, timeRemaining) {
		this.currentPhase = phase;
		this.timeRemaining = timeRemaining;

		const config = this.phaseConfig[phase];

		// Update visibility - hide during IDLE and TUTORIAL (tutorial overlay handles that phase)
		if (phase === PHASE.IDLE || phase === PHASE.TUTORIAL) {
			this.overlay.style.opacity = '0';
			return;
		} else {
			this.overlay.style.opacity = '1';
		}

		// Update phase badge
		this.phaseBadge.textContent = config.text;
		this.phaseBadge.style.backgroundColor = config.bgColor;
		this.phaseBadge.style.color = config.color;

		// Update countdown
		if (config.showCountdown && timeRemaining > 0) {
			this.countdownText.textContent = Math.ceil(timeRemaining);
			this.countdownText.style.display = 'block';
		} else {
			this.countdownText.style.display = 'none';
		}

		// Update traffic lights
		this._updateTrafficLight(config.light);
	}

	_updateTrafficLight(activeLight) {
		for (const [color, light] of Object.entries(this.lights)) {
			if (color === activeLight) {
				light.style.background = light.dataset.activeColor;
				light.style.boxShadow = `0 0 20px ${light.dataset.activeColor}, inset 0 2px 4px rgba(255, 255, 255, 0.3)`;
			} else {
				light.style.background = light.dataset.inactiveColor;
				light.style.boxShadow = 'inset 0 2px 4px rgba(0, 0, 0, 0.3)';
			}
		}
	}

	_updateProgress(state) {
		if (!state.isPlaying) {
			this.progressText.style.display = 'none';
			return;
		}

		const item = state.currentItem;
		const modelName = item?.model ? movementName(item.model, '') : '';

		this.progressText.textContent = `${modelName} - Item ${state.currentItemIndex + 1}/${state.totalItems} | Rep ${state.currentRepetition + 1}/${state.totalRepetitions}`;
		this.progressText.style.display = 'block';
	}

	show() {
		this.overlay.style.display = 'flex';
	}

	hide() {
		this.overlay.style.display = 'none';
	}
}
