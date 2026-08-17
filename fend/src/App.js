import './styles/style.scss';

import WebGLView from './webgl/WebGLView';
import GUIView from './gui/GUIView';
import TimelineView from './ui/TimelineView';
import SegmentManager from './ui/SegmentManager';
import SequencePlayer from './ui/SequencePlayer';
import sequenceConfig from './config/sequenceConfig';
import { loadSavedConfig } from './config/configStore';
import EMGClient from './emg/EMGClient';
import EMGChannelView from './emg/EMGChannelView';
import DecompositionController from './emg/DecompositionController';
import DecompositionView from './emg/DecompositionView';
import MovementSessionController from './emg/MovementSessionController';
import MovementSessionView from './emg/MovementSessionView';
import MovementStimController from './emg/MovementStimController';
import TriggerMonitor from './emg/TriggerMonitor';
import PhaseOverlay from './ui/PhaseOverlay';
import SequenceConfigPanel from './ui/SequenceConfigPanel';
import PatientModeController from './ui/PatientModeController';
import ModeMenu from './ui/ModeMenu';
import ProgressSidebar from './ui/ProgressSidebar';
import TutorialOverlay from './ui/TutorialOverlay';
import StatusBanner from './ui/StatusBanner';

export default class App {

	constructor() {
		this.el = document.querySelector('#app');
	}

	init() {
		this.initSegmentManager();
		this.initSequencePlayer();
		this.initEMG();
		this.initWebGL();
		this.initDecomposition();
		// MUST precede initGUI/initTimeline: both subscribe to mode changes from their
		// constructors, and silently skip it if modeController doesn't exist yet.
		this.initModeController();
		this.initGUI();
		this.initTimeline();
		this.initPatientMode();
		this.addListeners();
		this.animate();
		this.resize();
	}

	initModeController() {
		// Patient mode controller (manages config/patient mode switching). Created
		// before every component that subscribes to it — GUIView and TimelineView
		// subscribe from their constructors, so this cannot move below initGUI().
		this.modeController = new PatientModeController(this);
	}

	initPatientMode() {
		// Initialize progress sidebar (shows exercise list in patient mode)
		this.progressSidebar = new ProgressSidebar(this);

		// Initialize tutorial overlay (shows before each exercise)
		this.tutorialOverlay = new TutorialOverlay(this);

		// Guided movement-discrimination session. Created here (after modeController)
		// so it can switch into patient mode during recording, reusing the existing
		// hand + PhaseOverlay + ProgressSidebar for cueing. Uses the LIVE config the
		// user edits via the sequence panel (player.config), not a static import.
		this.movementSession = new MovementSessionController(
			this.emgClient, this.sequencePlayer,
			// config = static fallback (sequence + onlineStimulation) used if the live
			// player config doesn't carry them; live player.config still takes priority.
			// nostimTrigger = the stimulator ch wired to the trigger (fired at amp 0 to
			// the patient) so the no-stim block still records a trigger to segment on.
			{
				config: sequenceConfig,
				stimController: this.movementStim,
				modeController: this.modeController,
				triggerMonitor: this.triggerMonitor,
				nostimTrigger: sequenceConfig.stimulation?.nostimTrigger || null,
			},
		);
		// The sensor pass drives the stimulator directly through the client, so its own
		// failsafes report to the banner too (the closed-loop ones are wired in initEMG).
		this.movementSession.onAlarm((message) => this.statusBanner.raiseAlarm(message));

		this.movementSessionView = new MovementSessionView(this.movementSession);

		// Single mode dropdown (Config / Patient / Movement Session) — replaces the
		// floating mode toggle and the session launcher so buttons don't stack up.
		this.modeMenu = new ModeMenu(this);

		// Align every subscriber with the initial (config) mode. syncMode() rather than
		// setMode('config') because the controller already starts in config, so setMode
		// would early-return and notify nobody.
		this.modeController.syncMode();
	}

	initSegmentManager() {
		this.segmentManager = new SegmentManager();
	}

	initSequencePlayer() {
		this.sequencePlayer = new SequencePlayer();
		// Restore the last applied config so the running sequence matches what the
		// config panel shows. Booting from the static default here left the player on
		// a config the operator never chose until they opened the panel and pressed
		// Apply — and patient mode would run that ghost config in the meantime.
		this.sequencePlayer.loadConfig(loadSavedConfig() || sequenceConfig);

		// Initialize phase overlay for visual cues
		this.phaseOverlay = new PhaseOverlay(this.sequencePlayer);

		// Initialize sequence config panel
		this.sequenceConfigPanel = new SequenceConfigPanel(this);
	}

	initEMG() {
		// Initialize EMG WebSocket client
		this.emgClient = new EMGClient({
			host: 'localhost',
			port: 8765,
			autoReconnect: true,
		});

		// Connection + stimulation banner. Mounted on <body>, so it is the one status
		// indicator visible in patient mode as well as config mode. Created before
		// connect() so it sees the very first connect/disconnect.
		this.statusBanner = new StatusBanner(this.emgClient);

		// Initialize EMG channel visualization
		this.emgChannelView = new EMGChannelView(this.emgClient);

		// Initialize decomposition sidebar (needs emgClient; controller created after webgl)
		this.decompositionView = new DecompositionView(this.emgClient);

		// Closed-loop stim (recognized movement -> its FES pattern) for the online phase.
		// Its failsafes (stop not sent, dead-man watchdog) surface on the banner.
		this.movementStim = new MovementStimController(this.emgClient);
		this.movementStim.onAlarm((message) => this.statusBanner.raiseAlarm(message));

		// Live trigger waveform (for verifying the stim-trigger channel during setup)
		this.triggerMonitor = new TriggerMonitor(this.emgClient);

		// Connect to EMG server
		this.emgClient.connect();

		// Hook sequence player to EMG recording
		this.sequencePlayer.onEMGClient = this.emgClient;
	}

	initDecomposition() {
		// Controller needs both emgClient and webgl, so created after both are ready
		this.decompositionController = new DecompositionController(
			this.emgClient,
			this.webgl,
		);
	}

	initWebGL() {
		this.webgl = new WebGLView(this);
		this.webgl.setSegmentManager(this.segmentManager);
		this.sequencePlayer.setWebGLView(this.webgl);
		this.el.appendChild(this.webgl.renderer.domElement);
	}

	initGUI() {
		this.gui = new GUIView(this);
	}

	initTimeline() {
		this.timeline = new TimelineView(this, { heightPx: 72 });
		this.timeline.setSegmentManager(this.segmentManager);
		this.el.appendChild(this.timeline.el);
	}

	addListeners() {
		this.handlerAnimate = this.animate.bind(this);

		window.addEventListener('resize', this.resize.bind(this));
		window.addEventListener('keyup', this.keyup.bind(this));
	}

	animate() {
		this.update();
		this.draw();

		this.raf = requestAnimationFrame(this.handlerAnimate);
	}

	// ---------------------------------------------------------------------------------------------
	// PUBLIC
	// ---------------------------------------------------------------------------------------------

	update() {
		if (this.gui?.stats) this.gui.stats.begin();
		if (this.webgl) this.webgl.update();
		if (this.decompositionController) this.decompositionController.update();
		if (this.timeline && this.webgl) this.timeline.update(this.webgl.animationTime);
	}

	draw() {
		if (this.webgl) this.webgl.draw();
		if (this.gui?.stats) this.gui.stats.end();
	}

	// ---------------------------------------------------------------------------------------------
	// EVENT HANDLERS
	// ---------------------------------------------------------------------------------------------

	resize() {
		const isPatientMode = this.modeController?.isPatientMode();
		const sidebarWidth = isPatientMode ? 280 : 0;
		const timelineH = isPatientMode ? 0 : (this.timeline?.heightPx || 0);
		const decompW = this.decompositionView?.isVisible ? 320 : 0;

		const totalW = window.innerWidth;
		const totalH = window.innerHeight;

		const vw = Math.max(1, totalW - sidebarWidth - decompW);
		const vh = Math.max(1, totalH - timelineH);

		if (this.webgl) this.webgl.resize(vw, vh);
	}

	keyup(e) {
		// g or p
		if (e.keyCode == 71 || e.keyCode == 80) { if (this.gui) this.gui.toggle(); }
		// h
		if (e.keyCode == 72) { if (this.webgl.controls) this.webgl.controls.reset(); }
	}
}
