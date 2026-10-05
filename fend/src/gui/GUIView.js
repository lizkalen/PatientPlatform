import * as THREE from 'three';
import { Pane } from 'tweakpane';
import Stats from 'stats.js';
import WebGLView from '../webgl/WebGLView.js';

export default class GUIView {

	constructor(app) {
		this.app = app;

		// Model selection state
		this.selectedModel = 0;
		this.selectedAnimation = 0;

		// Segment state
		this.isAddingSegment = false;
		this.addingSegmentType = 'movement';
		this.movementSpeed = 1.0;
		this.restSpeed = 3.0;

		// Decomposition state
		// Machine-specific path to the decomposition model (.pkl). Set
		// VITE_DECOMP_MODEL_PATH in fend/.env.local rather than hardcoding it (see
		// .env.example); the panel field below stays editable at runtime.
		this.decompModelPath = import.meta.env.VITE_DECOMP_MODEL_PATH || '';
		this.decompMu1 = 0;
		this.decompMu2 = 1;
		this.decompThreshold = 10.0;
		this.decompLabel1ModelIdx = 0;
		this.decompLabel2ModelIdx = 1;
		this.decompBtPort = '';
		this.decompStatus = { status: 'Inactive' };
		this.mu1Binding = null;
		this.mu2Binding = null;

		this.initPane();
		this.initStats();
		this.subscribeToModeChanges();

		this.enable();
	}

	subscribeToModeChanges() {
		// Subscribe to patient mode controller to auto-hide in patient mode
		if (this.app.modeController) {
			this.app.modeController.onModeChange((mode) => {
				if (mode === 'patient') {
					this.disable();
				} else {
					this.enable();
				}
			});
		}
	}

	initPane() {
		let folder;

		document.title = `PatientGUI ${APP_VERSION}`;

		this.pane = new Pane({ title: document.title });
		this.pane.containerElem_.classList.add('full');

		// Model selection folder
		this.modelFolder = this.pane.addFolder({ title: 'MODEL' });
		this.initModelSelector();

		// Sequence player folder
		this.sequenceFolder = this.pane.addFolder({ title: 'SEQUENCE PLAYER' });
		this.initSequenceControls();

		// Segment controls folder
		this.segmentFolder = this.pane.addFolder({ title: 'SEGMENTS' });
		this.initSegmentControls();

		// EMG controls folder
		this.emgFolder = this.pane.addFolder({ title: 'EMG' });
		this.initEMGControls();

		// Decomposition folder
		this.decompFolder = this.pane.addFolder({ title: 'DECOMPOSITION' });
		this.initDecompositionControls();
	}

	initSequenceControls() {
		// Status display
		this.sequenceStatus = { info: 'Stopped' };
		this.sequenceStatusBinding = this.sequenceFolder.addBinding(
			this.sequenceStatus,
			'info',
			{ label: 'Status', readonly: true }
		);

		// Play/Pause button
		this.sequencePlayBtn = this.sequenceFolder.addButton({
			title: 'Play Sequence',
		}).on('click', () => {
			const player = this.app.sequencePlayer;
			if (!player) return;

			if (player.isPlaying) {
				player.pause();
			} else {
				player.play();
			}
		});

		// Stop button
		this.sequenceFolder.addButton({
			title: 'Stop',
		}).on('click', () => {
			this.app.sequencePlayer?.stop();
		});

		// Navigation buttons
		this.sequenceFolder.addButton({
			title: 'Previous',
		}).on('click', () => {
			this.app.sequencePlayer?.skipToPrevious();
		});

		this.sequenceFolder.addButton({
			title: 'Next',
		}).on('click', () => {
			this.app.sequencePlayer?.skipToNext();
		});

		// Configure sequence button
		this.sequenceFolder.addBlade({ view: 'separator' });

		this.sequenceFolder.addButton({
			title: 'Configure Sequence...',
		}).on('click', () => {
			if (this.app.sequenceConfigPanel) {
				this.app.sequenceConfigPanel.show();
			}
		});

		// Register for state changes
		this.app.sequencePlayer?.onChange((state) => {
			this._updateSequenceUI(state);
		});

		this.app.sequencePlayer?.onSequenceComplete(() => {
			this.sequencePlayBtn.title = 'Play Sequence';
		});
	}

	_updateSequenceUI(state) {
		if (state.isPlaying) {
			this.sequenceStatus.info = `${state.currentItemIndex + 1}/${state.totalItems} - Rep ${state.currentRepetition + 1}/${state.totalRepetitions}`;
			this.sequencePlayBtn.title = 'Pause';
		} else {
			this.sequenceStatus.info = state.totalItems > 0 ? `Paused (${state.currentItemIndex + 1}/${state.totalItems})` : 'No sequence';
			this.sequencePlayBtn.title = 'Play Sequence';
		}
		this.pane.refresh();
	}

	initSegmentControls() {
		// Add segment buttons
		this.addMovementBtn = this.segmentFolder.addButton({
			title: 'Add Movement Segment',
		}).on('click', () => {
			this.toggleAddingSegment('movement');
		});

		this.addRestBtn = this.segmentFolder.addButton({
			title: 'Add Rest Segment',
		}).on('click', () => {
			this.toggleAddingSegment('rest');
		});

		// Movement speed slider
		this.segmentFolder.addBinding(this, 'movementSpeed', {
			label: 'Movement Speed',
			min: 0.1,
			max: 5.0,
			step: 0.1,
		}).on('change', (e) => {
			if (this.app.segmentManager) {
				this.app.segmentManager.setSpeed('movement', e.value);
			}
		});

		// Rest speed slider
		this.segmentFolder.addBinding(this, 'restSpeed', {
			label: 'Rest Speed',
			min: 0.1,
			max: 5.0,
			step: 0.1,
		}).on('change', (e) => {
			if (this.app.segmentManager) {
				this.app.segmentManager.setSpeed('rest', e.value);
			}
		});

		// Toggle selected segment type
		this.segmentFolder.addButton({
			title: 'Toggle Selected Type',
		}).on('click', () => {
			if (this.app.timeline) {
				this.app.timeline.toggleSelectedSegmentType();
			}
		});

		// Delete selected segment
		this.segmentFolder.addButton({
			title: 'Delete Selected',
		}).on('click', () => {
			if (this.app.timeline) {
				this.app.timeline.deleteSelectedSegment();
			}
		});

		// Clear all segments
		this.segmentFolder.addButton({
			title: 'Clear All Segments',
		}).on('click', () => {
			if (this.app.segmentManager) {
				this.app.segmentManager.clearSegments();
			}
		});

		// Export/Import separator
		this.segmentFolder.addBlade({ view: 'separator' });

		// Export button
		this.segmentFolder.addButton({
			title: 'Export JSON',
		}).on('click', () => {
			if (this.app.segmentManager) {
				this.app.segmentManager.downloadJSON();
			}
		});

		// Import button (using hidden file input)
		this.segmentFolder.addButton({
			title: 'Import JSON',
		}).on('click', () => {
			this._showImportDialog();
		});
	}

	toggleAddingSegment(type) {
		if (this.isAddingSegment && this.addingSegmentType === type) {
			// Turn off adding mode
			this.isAddingSegment = false;
			this.addMovementBtn.title = 'Add Movement Segment';
			this.addRestBtn.title = 'Add Rest Segment';
		} else {
			// Turn on adding mode for specified type
			this.isAddingSegment = true;
			this.addingSegmentType = type;

			if (type === 'movement') {
				this.addMovementBtn.title = '[ Adding Movement... ]';
				this.addRestBtn.title = 'Add Rest Segment';
			} else {
				this.addMovementBtn.title = 'Add Movement Segment';
				this.addRestBtn.title = '[ Adding Rest... ]';
			}
		}

		// Update timeline
		if (this.app.timeline) {
			this.app.timeline.setAddingSegment(this.isAddingSegment, this.addingSegmentType);
		}
	}

	_showImportDialog() {
		const input = document.createElement('input');
		input.type = 'file';
		input.accept = '.json';
		input.onchange = (e) => {
			const file = e.target.files[0];
			if (!file) return;

			const reader = new FileReader();
			reader.onload = (ev) => {
				const json = ev.target.result;
				if (this.app.segmentManager) {
					const success = this.app.segmentManager.importJSON(json);
					if (success) {
						this.updateSegmentSpeedsFromManager();
						console.log('Segments imported successfully');
					} else {
						console.error('Failed to import segments');
					}
				}
			};
			reader.readAsText(file);
		};
		input.click();
	}

	updateSegmentSpeedsFromManager() {
		if (this.app.segmentManager) {
			this.movementSpeed = this.app.segmentManager.speeds.movement;
			this.restSpeed = this.app.segmentManager.speeds.rest;
			this.pane.refresh();
		}
	}

	initEMGControls() {
		// EMG connection status
		this.emgStatus = { status: 'Disconnected' };
		this.emgStatusBinding = this.emgFolder.addBinding(
			this.emgStatus,
			'status',
			{ label: 'Server', readonly: true }
		);

		// Recording status
		this.emgRecording = { recording: 'Not Recording' };
		this.emgRecordingBinding = this.emgFolder.addBinding(
			this.emgRecording,
			'recording',
			{ label: 'Recording', readonly: true }
		);

		// Open EMG Channels button
		this.emgFolder.addButton({
			title: 'View EMG Channels',
		}).on('click', () => {
			if (this.app.emgChannelView) {
				this.app.emgChannelView.show();
			}
		});

		// Manual recording controls
		this.emgFolder.addBlade({ view: 'separator' });

		this.emgFolder.addButton({
			title: 'Start Recording',
		}).on('click', () => {
			if (this.app.emgClient) {
				this.app.emgClient.startRecording();
			}
		});

		this.emgFolder.addButton({
			title: 'Stop Recording',
		}).on('click', () => {
			if (this.app.emgClient) {
				this.app.emgClient.stopRecording();
			}
		});

		// Reconnect button
		this.emgFolder.addBlade({ view: 'separator' });

		this.emgFolder.addButton({
			title: 'Reconnect',
		}).on('click', () => {
			if (this.app.emgClient) {
				this.app.emgClient.disconnect();
				setTimeout(() => {
					this.app.emgClient.autoReconnect = true;
					this.app.emgClient.connect();
				}, 500);
			}
		});

		// OTBioLab+ configuration (Quattrocento): point the server at a different
		// .otb+stp config at runtime, instead of the one baked into the launch
		// script. The server reloads it and reconnects with the new channel count.
		this.emgFolder.addBlade({ view: 'separator' });

		this.otbConfigPath = this.otbConfigPath || '';
		this._otbConfigBinding = this.emgFolder.addBinding(this, 'otbConfigPath', {
			label: 'OTB config',
		});

		this.emgFolder.addButton({ title: 'Browse OTB Config...' }).on('click', () => {
			const input = document.createElement('input');
			input.type = 'file';
			input.style.display = 'none';
			input.onchange = (e) => {
				const file = e.target.files[0];
				if (file) {
					// file.path is the absolute path (Electron/local); browsers only
					// expose the NAME, so the picked value may be just the filename.
					this.otbConfigPath = file.path || file.name;
					if (!file.path) {
						this.otbConfigInfo = 'Browser gave filename only — paste the FULL server path';
					}
					this.pane.refresh();
				}
				input.remove();
			};
			document.body.appendChild(input);
			input.click();
		});

		this.emgFolder.addButton({ title: 'Load OTB Config' }).on('click', () => {
			// Read straight from the input element: Tweakpane text bindings may not
			// flush the typed value back to the object until the field loses focus.
			const inputEl = this._otbConfigBinding.element.querySelector('input');
			if (inputEl) this.otbConfigPath = inputEl.value.trim();
			if (this.app.emgClient && this.otbConfigPath) {
				this.otbConfigInfo = 'loading…';
				this.pane.refresh();
				this.app.emgClient.setOtbConfig(this.otbConfigPath);
			}
		});

		// Feedback on the loaded device configuration (updated from config_status /
		// connect): total channels, EMG count, aux count, sample rate.
		this.otbConfigInfo = this.otbConfigInfo || 'no config loaded';
		this.emgFolder.addBinding(this, 'otbConfigInfo', { label: 'Device', readonly: true });

		// Trigger channel: which STREAMED channel carries the stim trigger. Movements
		// are labelled and stim is blanked from it, so it must be an exposed aux
		// channel (launch WITHOUT --emg-only so aux channels are streamed). Auto-set
		// to the first aux channel when a config loads; adjust here if needed.
		this.triggerChannel = this.triggerChannel ?? 192;
		this._trigBinding = this.emgFolder.addBinding(this, 'triggerChannel', {
			label: 'Trigger channel', step: 1, min: 0,
		});
		this._trigBinding.on('change', () => this._applyTriggerChannel());

		// Live trigger waveform to verify the selected channel carries the stim sync.
		this.emgFolder.addButton({ title: 'Show / hide trigger' }).on('click', () => {
			const m = this.app.triggerMonitor;
			if (!m) return;
			if (m.isOpen()) m.stop(); else m.start(this.triggerChannel);
		});

		// Subscribe to EMG client events
		this._subscribeToEMGClient();
	}

	_applyTriggerChannel() {
		this.app.movementSession?.setTrigCh?.(this.triggerChannel);
	}

	_subscribeToEMGClient() {
		const client = this.app.emgClient;
		if (!client) return;

		client.onConnect((info) => {
			this.emgStatus.status = `Connected (${info.nChannels}ch)`;
			this.otbConfigInfo = `${info.nChannels} ch @ ${info.sampleRate ?? '?'} Hz`;
			this.pane.refresh();
		});

		client.onDisconnect(() => {
			this.emgStatus.status = 'Disconnected';
			this.pane.refresh();
		});

		client.onRecordingStatus((status) => {
			this.emgRecording.recording = status.recording ? 'RECORDING' : 'Not Recording';
			this.pane.refresh();
		});

		client.onRecordingSaved((info) => {
			console.log(`EMG recording saved: ${info.filepath} (${info.durationSeconds.toFixed(1)}s)`);
		});

		client.onError((msg) => {
			console.error('EMG Error:', msg);
			// Don't leave the config field stuck on "loading…" when a load fails.
			if (this.otbConfigInfo === 'loading…') {
				this.otbConfigInfo = `Error: ${msg}`;
				this.pane.refresh();
			}
		});

		client.onConfigStatus((msg) => {
			this.emgStatus.status = msg.message || 'Config applied';
			if (msg.nch != null) {
				const exposed = msg.n_exposed ?? msg.nch;
				const auxExposed = exposed - (msg.n_emg ?? 0);   // 0 if --emg-only dropped aux
				this.otbConfigInfo = `${msg.device_name || 'device'}: ${exposed}/${msg.nch} exposed, ` +
					`${msg.n_emg ?? '?'} EMG + ${auxExposed} aux @ ${msg.fsamp ?? '?'}Hz` +
					(auxExposed <= 0 ? ' — NO AUX! relaunch server without --emg-only' : '');
				// Auto-suggest the trigger = first aux channel (just after the EMG);
				// the operator can override in the Trigger channel field.
				if (msg.n_emg != null && auxExposed > 0) {
					this.triggerChannel = msg.n_emg;
					this._applyTriggerChannel();
				}
			}
			this.pane.refresh();
		});
	}

	initDecompositionControls() {
		const folder = this.decompFolder;
		const client = this.app.emgClient;

		// Status display
		this.decompStatusBinding = folder.addBinding(this.decompStatus, 'status', {
			label: 'Model',
			readonly: true,
		});

		// Model path text input
		folder.addBinding(this, 'decompModelPath', { label: 'Path (.pkl)' });

		// Browse button for file selection
		folder.addButton({ title: 'Browse...' }).on('click', () => {
			const input = document.createElement('input');
			input.type = 'file';
			input.accept = '.pkl';
			input.style.display = 'none';
			input.onchange = (e) => {
				const file = e.target.files[0];
				if (file) {
					// Use webkitRelativePath or file name; for Electron/local use, the full path
					// is available via file.path (non-standard but supported in Electron)
					this.decompModelPath = file.path || file.name;
					this.pane.refresh();
				}
				input.remove();
			};
			document.body.appendChild(input);
			input.click();
		});

		// Load / Unload buttons
		folder.addButton({ title: 'Load Model' }).on('click', () => {
			const ctrl = this.app.decompositionController;
			if (ctrl) ctrl.loadModel(this.decompModelPath);
		});

		folder.addButton({ title: 'Unload Model' }).on('click', () => {
			const ctrl = this.app.decompositionController;
			if (ctrl) ctrl.unloadModel();
		});

		folder.addBlade({ view: 'separator' });

		// MU dropdowns — placeholders, rebuilt dynamically after model loads
		// (mu1Binding and mu2Binding are initially hidden until a model loads)
		this._rebuildMUDropdowns(0);

		// Threshold slider
		folder.addBinding(this, 'decompThreshold', {
			label: 'Threshold (Hz)',
			min: 1,
			max: 50,
			step: 0.5,
		}).on('change', (e) => {
			const ctrl = this.app.decompositionController;
			if (ctrl) ctrl.threshold = e.value;
		});

		// Optional COM port for exo hand control — text field + Set button
		// A dedicated "Set" button is used because Tweakpane text bindings may not
		// flush the typed value back to the JS object until the field loses focus.
		this._btPortBinding = folder.addBinding(this, 'decompBtPort', { label: 'Exo COM port' });
		folder.addButton({ title: 'Set Exo COM port' }).on('click', () => {
			// Read directly from the specific input element for this binding
			const inputEl = this._btPortBinding.element.querySelector('input');
			if (inputEl) this.decompBtPort = inputEl.value.trim();
			const ctrl = this.app.decompositionController;
			if (ctrl) ctrl.btPort = this.decompBtPort;
			console.log('[GUIView] Exo COM port set to:', JSON.stringify(this.decompBtPort));
		});

		// Start / Stop Classification
		folder.addButton({ title: 'Start Classification' }).on('click', () => {
			const ctrl = this.app.decompositionController;
			if (!ctrl) return;
			ctrl.mu1Idx = this.decompMu1;
			ctrl.mu2Idx = this.decompMu2;
			ctrl.threshold = this.decompThreshold;
			// btPort was already set via "Set Exo COM port" button
			console.log(`[GUIView] Start Classification clicked: mu1=${this.decompMu1} mu2=${this.decompMu2} ` +
				`threshold=${this.decompThreshold} btPort=${JSON.stringify(ctrl.btPort)}`);
			ctrl.startClassification();
		});

		folder.addButton({ title: 'Stop Classification' }).on('click', () => {
			const ctrl = this.app.decompositionController;
			if (ctrl) ctrl.stopClassification();
		});

		folder.addBlade({ view: 'separator' });

		// Model dropdowns for label 1 and label 2
		const modelOptions = {};
		WebGLView.AVAILABLE_MODELS.forEach((m, i) => { modelOptions[m.name] = i; });

		folder.addBinding(this, 'decompLabel1ModelIdx', {
			label: 'MOVE 1 model',
			options: modelOptions,
		}).on('change', (e) => {
			const ctrl = this.app.decompositionController;
			if (ctrl) ctrl.setLabel1Model(WebGLView.AVAILABLE_MODELS[e.value].file);
		});

		folder.addBinding(this, 'decompLabel2ModelIdx', {
			label: 'MOVE 2 model',
			options: modelOptions,
		}).on('change', (e) => {
			const ctrl = this.app.decompositionController;
			if (ctrl) ctrl.setLabel2Model(WebGLView.AVAILABLE_MODELS[e.value].file);
		});

		// Apply initial model selections to the controller
		const ctrl = this.app.decompositionController;
		if (ctrl) {
			ctrl.setLabel1Model(WebGLView.AVAILABLE_MODELS[this.decompLabel1ModelIdx].file);
			ctrl.setLabel2Model(WebGLView.AVAILABLE_MODELS[this.decompLabel2ModelIdx].file);
		}

		folder.addBlade({ view: 'separator' });

		// Toggle sidebar panel
		folder.addButton({ title: 'View Panel' }).on('click', () => {
			if (this.app.decompositionView) {
				this.app.decompositionView.toggle();
				this.app.resize();
			}
		});

		// Subscribe to decomposition status to update label and rebuild MU dropdowns
		if (client) {
			client.onDecompositionStatus((status) => {
				if (status.active) {
					this.decompStatus.status = `Active (${status.nMUs} MUs)`;
					this._rebuildMUDropdowns(status.nMUs);
				} else {
					this.decompStatus.status = 'Inactive';
					this._rebuildMUDropdowns(0);
				}
				this.pane.refresh();
			});
		}
	}

	_rebuildMUDropdowns(nMUs) {
		// Dispose existing bindings
		if (this.mu1Binding) { this.mu1Binding.dispose(); this.mu1Binding = null; }
		if (this.mu2Binding) { this.mu2Binding.dispose(); this.mu2Binding = null; }

		if (nMUs < 2) return;

		const options = {};
		for (let i = 0; i < nMUs; i++) { options[`MU ${i}`] = i; }

		// Clamp current selections within new range
		this.decompMu1 = Math.min(this.decompMu1, nMUs - 1);
		this.decompMu2 = Math.min(this.decompMu2, nMUs - 1);

		// Insert after the separator that follows status/path/load — use the
		// separator blade index approach by inserting into the folder directly.
		// Tweakpane doesn't support positional inserts, so we add at the end of
		// what the folder has so far. Since _rebuildMUDropdowns is called before
		// threshold is added during init, the order is correct on first build.
		// On subsequent rebuilds (after model load) they appear at end of folder,
		// which is acceptable as Tweakpane doesn't support reordering.
		this.mu1Binding = this.decompFolder.addBinding(this, 'decompMu1', {
			label: 'MU 1 index',
			options,
		}).on('change', (e) => {
			const ctrl = this.app.decompositionController;
			if (ctrl) ctrl.mu1Idx = e.value;
		});

		this.mu2Binding = this.decompFolder.addBinding(this, 'decompMu2', {
			label: 'MU 2 index',
			options,
		}).on('change', (e) => {
			const ctrl = this.app.decompositionController;
			if (ctrl) ctrl.mu2Idx = e.value;
		});
	}

	initModelSelector() {
		// Build model options from available models
		const modelOptions = {};
		WebGLView.AVAILABLE_MODELS.forEach((model, index) => {
			modelOptions[model.name] = index;
		});

		// Model dropdown
		this.modelBinding = this.modelFolder.addBinding(
			this,
			'selectedModel',
			{
				label: 'Model',
				options: modelOptions,
			}
		).on('change', (e) => {
			const model = WebGLView.AVAILABLE_MODELS[e.value];
			this.app.webgl.currentModelIndex = e.value;
			this.app.webgl.loadModel(model.file);
		});

		// Animation dropdown (will be populated when model loads)
		this.animationBinding = null;
	}

	updateModelControls() {
		// Remove existing animation binding if present
		if (this.animationBinding) {
			this.animationBinding.dispose();
			this.animationBinding = null;
		}

		const webgl = this.app.webgl;
		const animations = webgl.availableAnimations || [];

		if (animations.length > 1) {
			// Build animation options
			const animOptions = {};
			animations.forEach((clip, index) => {
				const name = clip.name || `Animation ${index}`;
				animOptions[name] = index;
			});

			this.selectedAnimation = 0;
			this.animationBinding = this.modelFolder.addBinding(
				this,
				'selectedAnimation',
				{
					label: 'Animation',
					options: animOptions,
				}
			).on('change', (e) => {
				webgl.playAnimation(e.value);
			});
		}

		// Sync segment speeds from manager
		this.updateSegmentSpeedsFromManager();
	}

	initStats() {
		this.stats = new Stats();
		document.body.appendChild(this.stats.dom);
	}

	// ---------------------------------------------------------------------------------------------
	// PUBLIC
	// ---------------------------------------------------------------------------------------------

	enable() {
		this.pane.hidden = false;
		if (this.stats) this.stats.dom.style.display = '';

		if (!this.pane.containerElem_.classList.contains('full')) return;
		this.app.el.style.width = `calc(100vw - ${this.pane.containerElem_.offsetWidth}px)`;
		this.app.resize();
	}

	disable() {
		this.pane.hidden = true;
		if (this.stats) this.stats.dom.style.display = 'none';

		if (!this.pane.containerElem_.classList.contains('full')) return;
		this.app.el.style.width = ``;
		this.app.resize();
	}

	toggle() {
		if (!this.pane.hidden) this.disable();
		else this.enable();
	}
}