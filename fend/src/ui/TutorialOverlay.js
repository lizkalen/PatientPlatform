/**
 * Tutorial Overlay Component
 * ==========================
 *
 * Two-stage tutorial system:
 * 1. INSTRUCTIONS stage: Explains what will happen + safety notice + Start button
 * 2. PRACTICE stage: Animation plays, patient practices, clicks "I'm Ready"
 *
 * The 3D hand animation plays in the background so patients can:
 * - See the full movement clearly
 * - Rotate the camera view (drag to rotate)
 * - Practice along with the looping animation
 */

import { MODE } from './PatientModeController';
import { movementName } from '../config/movements';

// Tutorial stages
const STAGE = {
	INSTRUCTIONS: 'instructions',
	PRACTICE: 'practice'
};

export default class TutorialOverlay {
	constructor(app) {
		this.app = app;
		this.sequencePlayer = app.sequencePlayer;
		this.modeController = app.modeController;

		this._currentItem = null;
		this._confirmCallback = null;
		this._isVisible = false;
		this._stage = STAGE.INSTRUCTIONS;

		this._createElements();
		this._subscribeToSequencePlayer();
		this._subscribeToModeChanges();
	}

	_createElements() {
		// Main overlay container
		this.el = document.createElement('div');
		this.el.className = 'tutorial-overlay';

		// Content container
		const content = document.createElement('div');
		content.className = 'tutorial-overlay__content';

		// =====================
		// INSTRUCTIONS STAGE
		// =====================
		this.instructionsContainer = document.createElement('div');
		this.instructionsContainer.className = 'tutorial-overlay__instructions';

		// Badge for instructions
		this.instructionsBadge = document.createElement('div');
		this.instructionsBadge.className = 'tutorial-overlay__badge';
		this.instructionsBadge.textContent = 'NEXT EXERCISE';
		this.instructionsContainer.appendChild(this.instructionsBadge);

		// Title (exercise name)
		this.title = document.createElement('h2');
		this.title.className = 'tutorial-overlay__title';
		this.title.textContent = 'Exercise Name';
		this.instructionsContainer.appendChild(this.title);

		// Subtitle (repetition count)
		this.subtitle = document.createElement('p');
		this.subtitle.className = 'tutorial-overlay__subtitle';
		this.subtitle.textContent = '5 repetitions';
		this.instructionsContainer.appendChild(this.subtitle);

		// Instructions explanation
		this.explanationText = document.createElement('p');
		this.explanationText.className = 'tutorial-overlay__explanation';
		this.explanationText.textContent = 'A movement will be shown on screen. Watch carefully and attempt to replicate it. Press "I\'m Ready" when you feel confident.';
		this.instructionsContainer.appendChild(this.explanationText);

		// Safety notice
		this.safetyNotice = document.createElement('div');
		this.safetyNotice.className = 'tutorial-overlay__safety';
		this.safetyNotice.innerHTML = `
			<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
				<path d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"/>
			</svg>
			<span>Please notify your physician if you feel any discomfort or fatigue</span>
		`;
		this.instructionsContainer.appendChild(this.safetyNotice);

		// Start button
		this.startButton = document.createElement('button');
		this.startButton.className = 'tutorial-overlay__start-btn';
		this.startButton.textContent = 'Start Tutorial';
		this.startButton.addEventListener('click', () => this._onStartClick());
		this.instructionsContainer.appendChild(this.startButton);

		content.appendChild(this.instructionsContainer);

		// =====================
		// PRACTICE STAGE
		// =====================
		this.practiceContainer = document.createElement('div');
		this.practiceContainer.className = 'tutorial-overlay__practice';
		this.practiceContainer.style.display = 'none';

		// Badge for practice
		this.practiceBadge = document.createElement('div');
		this.practiceBadge.className = 'tutorial-overlay__badge tutorial-overlay__badge--practice';
		this.practiceBadge.textContent = 'WATCH & PRACTICE';
		this.practiceContainer.appendChild(this.practiceBadge);

		// Practice instruction text
		this.practiceInstruction = document.createElement('p');
		this.practiceInstruction.className = 'tutorial-overlay__instruction';
		this.practiceInstruction.textContent = 'Practice the movement above - drag to rotate the view';
		this.practiceContainer.appendChild(this.practiceInstruction);

		// Ready button
		this.readyButton = document.createElement('button');
		this.readyButton.className = 'tutorial-overlay__ready-btn';
		this.readyButton.textContent = "I'm Ready";
		this.readyButton.addEventListener('click', () => this._onReadyClick());
		this.practiceContainer.appendChild(this.readyButton);

		content.appendChild(this.practiceContainer);

		this.el.appendChild(content);
		document.body.appendChild(this.el);
	}

	_subscribeToSequencePlayer() {
		if (!this.sequencePlayer) return;

		// Listen for tutorial requests
		this.sequencePlayer.onTutorialRequest((item, confirmFn) => {
			this._showTutorial(item, confirmFn);
		});
	}

	_subscribeToModeChanges() {
		if (!this.modeController) return;

		// Enable tutorial when entering patient mode
		this.modeController.onModeChange((mode) => {
			if (mode === MODE.PATIENT) {
				this.sequencePlayer?.setTutorialEnabled(true);
			} else {
				this.sequencePlayer?.setTutorialEnabled(false);
				// Hide if switching away from patient mode while tutorial is showing
				if (this._isVisible) {
					this.hide();
				}
			}
		});
	}

	_showTutorial(item, confirmFn) {
		this._currentItem = item;
		this._confirmCallback = confirmFn;

		// Update content
		this.title.textContent = this._getDisplayName(item);
		this.subtitle.textContent = `${item.repetitions} repetition${item.repetitions !== 1 ? 's' : ''}`;

		// Start in instructions stage
		this._setStage(STAGE.INSTRUCTIONS);

		// Show overlay
		this.show();

		// Don't start animation yet - wait for Start button
	}

	_onStartClick() {
		// Transition to practice stage
		this._setStage(STAGE.PRACTICE);

		// Now start the animation loop for practice
		if (this.app.webgl && this.app.webgl.action) {
			this.app.webgl.action.reset();
			this.app.webgl.action.setLoop(2, Infinity); // THREE.LoopRepeat = 2
			this.app.webgl.play();
		}
	}

	_setStage(stage) {
		this._stage = stage;

		// Hide all containers first
		this.instructionsContainer.style.display = 'none';
		this.practiceContainer.style.display = 'none';

		switch (stage) {
			case STAGE.INSTRUCTIONS:
				this.instructionsContainer.style.display = '';
				this.el.classList.remove('tutorial-overlay--practice');
				this.el.classList.add('tutorial-overlay--instructions');
				break;

			case STAGE.PRACTICE:
				this.practiceContainer.style.display = '';
				this.el.classList.remove('tutorial-overlay--instructions');
				this.el.classList.add('tutorial-overlay--practice');
				break;
		}
	}

	_onReadyClick() {
		// Directly finish tutorial - PREP phase will handle the countdown
		this._finishTutorial();
	}

	_finishTutorial() {
		// Pause and reset animation to beginning, ready for the actual exercise
		if (this.app.webgl && this.app.webgl.action) {
			const action = this.app.webgl.action;
			// Pause first
			action.paused = true;
			this.app.webgl.isPlaying = false;
			// Reset to time 0 (reset() also sets time to 0 and re-enables the action)
			action.reset();
			// Keep it paused after reset (reset() unpauses)
			action.paused = true;
		}

		// Hide overlay
		this.hide();

		// Reset stage classes
		this.el.classList.remove('tutorial-overlay--instructions', 'tutorial-overlay--practice');

		// Call the confirm callback (will trigger PREP phase with traffic light)
		if (this._confirmCallback) {
			this._confirmCallback();
			this._confirmCallback = null;
		}

		this._currentItem = null;
	}

	_getDisplayName(item) {
		return movementName(item, 'Exercise');
	}

	show() {
		this._isVisible = true;
		this.el.classList.add('tutorial-overlay--visible');
	}

	hide() {
		this._isVisible = false;
		this.el.classList.remove('tutorial-overlay--visible');
	}

	destroy() {
		this.el.remove();
	}
}
