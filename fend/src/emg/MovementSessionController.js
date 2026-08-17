/**
 * Movement Session Controller
 * ===========================
 *
 * Orchestrates the guided movement-discrimination procedure as one state machine:
 *
 *   welcome -> record_train -> training -> trained
 *           -> record_calib -> calibrating -> sensor_ready -> sensor_sim
 *           -> sensor_done -> online -> done
 *
 * After calibration the trained two-headed model is loaded. Before free-running
 * closed-loop control we run a SENSOR-TRIGGERED pass: the movement config is
 * replayed with the player's auto-stim disabled, and a timer fires each movement's
 * stim pattern like an external sensor would — no stim for the first second of the
 * movement, stim on until 2s after the movement ends. Meanwhile the loaded model
 * predicts in the background: the no-stim head should own the pre-stim onset and the
 * return-to-rest, the stim head the stimulated span (head routing is automatic from
 * the trigger channel). A GUIDED online closed-loop run follows the pass: a fresh
 * calibration runs right before it, then the trained movements are cued round-robin
 * (a short rest between cues) until their training reps are spent, with the recognized
 * movement driving its own stim.
 *
 * It reuses SequencePlayer to run the two cued recordings under stimulation (the
 * multi-rep training sequence and the 1-rep-each calibration sequence), and drives
 * the server training flow (mv_record_start/stop, mv_train, mv_calibrate),
 * reacting to mv_train_status / movement_status. A view subscribes to onChange and
 * renders the narration + loading indicators for each state.
 *
 * The movement classes and the per-bout class sequence are DERIVED from the
 * sequence config: each distinct (model, animation) among the stimulated items is
 * a class; each rep is one trigger bout (server labels bouts from the trigger).
 */
import { PHASE } from '../ui/SequencePlayer';

export const SESSION = {
	WELCOME: 'welcome',
	RECORD_TRAIN: 'record_train',
	TRAINING: 'training',
	TRAINED: 'trained',
	RECORD_CLEAN: 'record_clean',       // no-stim pass (clean specialist)
	TRAINING_CLEAN: 'training_clean',
	TRAINED_CLEAN: 'trained_clean',
	RECORD_CALIB: 'record_calib',
	CALIBRATING: 'calibrating',
	SENSOR_READY: 'sensor_ready',       // calibrated + model loaded; offer the sensor pass
	SENSOR_SIM: 'sensor_sim',           // timed-stim pass with background prediction
	SENSOR_DONE: 'sensor_done',         // sensor pass complete; offer live control
	ONLINE: 'online',
	DONE: 'done',
	// TERMINAL. Entered by emergencyStop() and left ONLY by reset() — an explicit
	// operator action. No automatic transition may leave it, and no cueing, sequence
	// or stim may be armed while it is set (fend plan A1).
	HALTED: 'halted',
	ERROR: 'error',
};

export default class MovementSessionController {
	constructor(emgClient, sequencePlayer, options = {}) {
		this.client = emgClient;
		this.player = sequencePlayer;
		this.baseConfig = options.config || null;   // fallback only
		this.params = options.params || {};        // { clf, norm, pca, winMs, stepMs, smooth, trigCh }
		this.savePath = options.savePath || null;
		this.stim = options.stimController || null; // closed-loop stim for the online phase
		this.mode = options.modeController || null; // to enter patient mode during recording
		this.trigMon = options.triggerMonitor || null;
		// Trigger channel spec for the NO-STIM block: the stimulator's ch3 is wired to
		// the trigger, not the patient. In that block we zero the patient channels'
		// amplitude but still fire THIS channel so the trigger records for segmentation.
		// { id, amplitude, pulse_width, frequency, is_biphasic } — must be set to run it.
		this.nostimTrigger = options.nostimTrigger || null;
		// Sensor-triggered pass timing (simulates an external sensor firing the stim):
		//   preStimDelayMs  — no stim for this long after the movement begins, then stim on
		//   postStimHoldMs  — keep stim on this long after the movement ends, then stim off
		this.preStimDelayMs = options.preStimDelayMs ?? 1000;
		this.postStimHoldMs = options.postStimHoldMs ?? 2000;
		// Guided online run pacing (seconds): a gap between consecutive cued movements
		// plus a get-ready countdown before each, so reps don't run fully back-to-back.
		this.onlineRestSec = options.onlineRestSec ?? 2;
		this.onlinePrepSec = options.onlinePrepSec ?? 1;
		// Redundant hard stop for a sensor-pass train. The intended stop is a single
		// setTimeout armed on the leave-MOVE transition; if that transition never
		// arrives (player stalled, config swapped, socket dropped) nothing else turns
		// the train off. The deadline is derived per bout from the cued clip's own
		// duration (see _armHardStop) — this value is only the FALLBACK for when that
		// duration cannot be read. It must not act as a ceiling: movement segments of
		// 12-14 s are normal (plan E2), so a fixed 15 s cap would false-positive into a
		// terminal halt on a legitimately long bout.
		this.maxTimedStimMs = options.maxTimedStimMs ?? 15000;
		// Margin added to a derived deadline, and the cadence for retrying a stop that
		// failed to send.
		this.hardStopMarginMs = options.hardStopMarginMs ?? 2000;
		this.stopRetryMs = options.stopRetryMs ?? 2000;
		this._sessionConfig = null;                 // the live config captured at session start
		this.montage = { perGrid: 64, nGrids: 3 };  // EMG grid layout (drives good-mask + features)
		this.deviceInfo = '';                       // device/channel feedback for the UI

		// -- sensor-triggered pass runtime state --
		this._prevPhase = PHASE.IDLE;               // last player phase (to detect move enter/leave)
		this._sensorLabel = null;                   // movement currently cued in the sensor pass
		this._sensorStim = null;                    // 'prestim' | 'stim' | 'rest' — for the UI
		this._sensorStimOn = false;                 // is a timed stim train currently firing
		this._stimStartTimer = null;                // fires stim preStimDelayMs after move start
		this._stimStopTimer = null;                 // stops stim postStimHoldMs after move end
		this._sensorFinalizeTimer = null;           // finalizes the pass after the last stim hold
		this._hardStopTimer = null;                 // watchdog: unconditional stop for a timed train
		this._pendingStimStop = false;              // a timed-stim stop that never reached the server

		// Emergency-stop latch. Terminal: only reset() clears it. Every path that could
		// arm stimulation again bails on it, so a STOP cannot be undone by the sequence
		// simply advancing to the next MOVE phase (fend plan A1).
		this._halted = false;
		this._haltedFrom = null;                    // state at halt time (reset() closes its capture)
		this._alarmListeners = [];                  // operator-visible alarms -> status banner

		this.state = SESSION.WELCOME;
		this.message = '';
		this.error = null;
		this.summary = null;                        // last train/calibrate summary
		this.trainLog = [];                         // running training progress log (for the UI)
		this._mvPaused = false;                     // capture paused during a tutorial phase
		this._calibThenOnline = false;              // the pending calibration should launch the guided online run

		this._derived = null;                       // { classOrder, classSequence, classKeys, items }
		this._calibSequence = null;
		this._onSeqComplete = null;                 // one-shot handler for the running sequence
		this._listeners = [];
		this._applyMontage();          // fill params.grids + default trigger channel
		this._subscribe();
	}

	// -- montage / trigger config -------------------------------------------------
	/** EMG grid layout: `nGrids` grids of `perGrid` channels. Drives the good-channel
	 * mask + grid features, and sets the default trigger = first aux (= total EMG). */
	setMontage(perGrid, nGrids) {
		this.montage = { perGrid: Math.max(1, perGrid | 0), nGrids: Math.max(1, nGrids | 0) };
		this._applyMontage();
	}
	_applyMontage() {
		const { perGrid, nGrids } = this.montage;
		const grids = [];
		for (let i = 0; i < nGrids; i++) grids.push([i * perGrid, (i + 1) * perGrid]);
		this.params.grids = grids;
		this.params.nEmg = perGrid * nGrids;
		this.params.trigCh = perGrid * nGrids;     // trigger = first aux channel after EMG
		this._emit();
	}
	/** Override the trigger channel index (persists until montage/config changes). */
	setTrigCh(v) { this.params.trigCh = Math.max(0, v | 0); this._emit(); }

	/** Toggle the live trigger waveform for the current trigger channel. */
	toggleTrigger() {
		if (!this.trigMon) return;
		if (this.trigMon.isOpen()) this.trigMon.stop();
		else this.trigMon.start(this.params.trigCh);
	}

	// -- public API ---------------------------------------------------------------
	setConfig(config) { this.baseConfig = config; }
	onChange(cb) { this._listeners.push(cb); }

	/** Subscribe to operator-visible stim alarms. Called with a message string. */
	onAlarm(cb) { this._alarmListeners.push(cb); }

	_raiseAlarm(message) {
		console.warn('[MovementSession] ALARM:', message);
		this.trainLog.push({ t: Date.now(), level: 'error', line: message });
		for (const cb of this._alarmListeners) {
			try { cb(message); } catch (e) { console.error('MovementSession alarm listener error', e); }
		}
	}

	/** The live sequence config the user edited via the panel (loaded into the
	 * player), falling back to the static default only if none is present. */
	_config() { return this.player?.config || this.baseConfig; }

	/** The planned movements for display: [{ label, reps }] in play order, plus the
	 * class order (movements + 'rest'). Derived from the LIVE config. */
	getPlan() {
		const cfg = this._config();
		if (!cfg) return { items: [], classOrder: [] };
		const items = (cfg.items || []).filter((it) => it.stimulate !== false);
		const labels = this._labels(items);
		const d = this._deriveClasses(cfg);
		return {
			items: items.map((it, i) => ({
				label: it.restClass ? 'rest (stim)' : labels[i],
				reps: it.repetitions || 1,
			})),
			classOrder: d.classOrder,
		};
	}

	getState() {
		return {
			state: this.state,
			message: this.message,
			error: this.error,
			summary: this.summary,
			seq: this.player.getState(),
			classOrder: this._derived?.classOrder || null,
			deviceInfo: this.deviceInfo,
			montage: { ...this.montage },
			trigCh: this.params.trigCh,
			trigStimId: (this.nostimTrigger || this._config()?.stimulation?.nostimTrigger)?.id ?? null,
			nEmg: this.params.nEmg,
			trainLog: this.trainLog.slice(-40),
			// Sensor-triggered pass: cued movement + timed-stim state + configured delays.
			sensorStim: this._sensorStim,
			sensorLabel: this._sensorLabel,
			halted: this._halted,
			preStimDelayMs: this.preStimDelayMs,
			postStimHoldMs: this.postStimHoldMs,
		};
	}

	/** Operator: adjust the sensor-pass timing (ms). Clamped to sane, non-negative values. */
	setPreStimDelayMs(v) { this.preStimDelayMs = Math.max(0, v | 0); this._emit(); }
	setPostStimHoldMs(v) { this.postStimHoldMs = Math.max(0, v | 0); this._emit(); }

	/** Which STIMULATOR channel drives the trigger during the no-stim block — fired at
	 * amplitude while the patient channels are zeroed. Operator-definable in the GUI
	 * (not hardcoded); other pulse params inherit the config's nostimTrigger template. */
	setTrigStimId(id) {
		id = Math.max(1, id | 0);
		const base = this.nostimTrigger || this._config()?.stimulation?.nostimTrigger
			|| { amplitude: 8.0, pulse_width: 200, frequency: 30.0, is_biphasic: true };
		this.nostimTrigger = { ...base, id };
		this._emit();
	}

	/** Turn a mv_train_status message into a human line for the training log panel, so
	 * the operator can see bouts detected / class counts / trigger threshold live. */
	_logTrain(msg) {
		const cc = msg.class_counts, order = msg.class_order || this._derived?.classOrder;
		const counts = (cc && order) ? order.map((c, i) => `${c} ${cc[i] ?? 0}`).join(' · ') : '';
		const trg = msg.trigger;
		const PHASE = { stim: 'Stim', nostim: 'No-stim', calib: 'Calib' };
		let line, level = 'info';
		switch (msg.state) {
			case 'recording': line = 'recording…'; break;
			case 'recorded': line = `recorded (${((msg.n_samples || 0) / (this.params.fs || 2048)).toFixed(1)}s)`; break;
			case 'training': line = 'training stim model…'; break;
			case 'training_clean': line = 'training clean model…'; break;
			case 'calibrating': line = 'calibrating + fitting…'; break;
			case 'trained':
			case 'trained_clean': {
				const exp = this._derived?.classSequence?.length;
				const boutWarn = (msg.n_spans_detected != null && exp != null && msg.n_spans_detected !== exp);
				level = (msg.n_spans_detected === 0 || boutWarn) ? 'warn' : 'ok';
				line = `${PHASE[msg.phase] || ''} trained ✓ — ${msg.n_spans_detected}/${exp ?? '?'} bouts`
					+ (counts ? `, ${counts}` : '')
					+ (trg ? `  ·  trig mid ${trg.mid}, ${Math.round(trg.frac_below * 100)}% below` : '');
				break;
			}
			case 'calibrated':
				level = 'ok';
				line = `Calibrated ✓ — ${msg.n_calib_bouts} bouts, stim resub ${(msg.resub_acc ?? 0).toFixed(2)}`
					+ (msg.two_stage ? `, clean resub ${(msg.clean_resub ?? 0).toFixed(2)} (two-stage)` : ' (single-stage)');
				break;
			case 'online_recording': line = 'online run recording…'; break;
			case 'online_saved': line = 'online run saved'; break;
			case 'error': level = 'error'; line = `ERROR: ${msg.message || 'training failed'}`; break;
			default: return;
		}
		this.trainLog.push({ t: Date.now(), level, line });
	}

	/** Operator: begin the session -> record the training sequence under stim.
	 * Enters patient mode so the existing hand + PhaseOverlay + ProgressSidebar
	 * do the cueing (we don't re-render any of that). */
	beginTraining() {
		const cfg = this._config();
		if (!cfg || !(cfg.items || []).length) {
			return this._fail('No movement sequence configured.');
		}
		this._sessionConfig = cfg;
		this._derived = this._deriveClasses(cfg);
		if (this._derived.classSequence.length === 0) {
			return this._fail('No stimulated movements in the sequence (enable stimulation + stimulate:true).');
		}
		this.trainLog = [];                         // fresh log for this session
		this.mode?.setMode('patient');
		this.client.mvRecordStart(this._recMeta('stim_train'));
		this._set(SESSION.RECORD_TRAIN, 'Recording your movements — perform each one when cued.');
		this._runSequence(cfg, () => this._onTrainRecorded());
	}

	/** Metadata saved with each raw recording: drives the informative filename and the
	 * per-movement segment files ({label}_{movement}.pkl). */
	_recMeta(label, classSequence) {
		const cfg = this._sessionConfig || this.baseConfig || {};
		return {
			label,
			session: cfg.name || cfg.metadata?.sessionId || 'session',
			class_order: this._derived?.classOrder || [],
			class_sequence: classSequence ?? this._derived?.classSequence ?? [],
			trig_ch: this.params.trigCh,
			grids: this.params.grids,
		};
	}

	/** Operator: after the stim training, record the same movements with NO stim to
	 * the patient (ch3 still fires the trigger for segmentation) to train the clean
	 * specialist. Patient-channel amplitude is zeroed; the trigger channel fires. */
	beginCleanRecording() {
		if (!this._sessionConfig || !this._derived) {
			return this._fail('Record the stim training block first.');
		}
		if (!(this.nostimTrigger || this._sessionConfig?.stimulation?.nostimTrigger)) {
			return this._fail('No-stim block needs a trigger stim channel (set "Trigger stim ch" '
				+ 'in the session settings — the stimulator ch wired to the trigger) so bouts can be segmented.');
		}
		this.mode?.setMode('patient');
		this.client.mvRecordStart(this._recMeta('nostim_train', this._derived.classSequenceNoRest));
		this._set(SESSION.RECORD_CLEAN,
			'No-stim pass — perform each movement when cued. You will feel no stimulation.');
		this._runSequence(this._nostimConfig(this._sessionConfig), () => this._onCleanRecorded());
	}

	/** Operator: after training, record one rep of each movement for calibration. */
	beginCalibration() {
		this._startCalibration('Calibration — perform each movement once when cued.');
	}

	/** Record one rep of each movement + calibrate, narrated by `message`. Shared by the
	 * post-training calibration and the fresh re-calibration run right before live control. */
	_startCalibration(message) {
		const calib = this._calibConfig(this._sessionConfig, this._derived);
		this._calibSequence = calib.sequence;
		this.mode?.setMode('patient');
		this.client.mvRecordStart(this._recMeta('calib', calib.sequence));
		this._set(SESSION.RECORD_CALIB, message);
		this._runSequence(calib.config, () => this._onCalibRecorded());
	}

	/** Operator: run the SENSOR-TRIGGERED pass. Replays the movement config with the
	 * player's auto-stim disabled; a timer fires each cued movement's stim pattern like
	 * an external (e.g. pressure) sensor would — no stim for the first `preStimDelayMs`
	 * of the movement, then stim on until `postStimHoldMs` after the movement ends. The
	 * already-loaded two-headed model predicts in the background the whole time. */
	beginSensorRun() {
		if (!this._sessionConfig || !this._derived) {
			return this._fail('Calibrate first — the model must be trained and loaded.');
		}
		this._clearSensorTimers();
		this._sensorStimOn = false;
		this._sensorStim = null;
		this._sensorLabel = null;
		this.mode?.setMode('patient');
		// Capture the run (raw + decisions) while the loaded model streams predictions.
		this.client.mvOnlineStart({
			label: 'sensor_sim',
			session: (this._sessionConfig || this.baseConfig || {}).name || 'session',
			class_order: this._derived?.classOrder || [],
		});
		this._set(SESSION.SENSOR_SIM,
			'Sensor-triggered run — move when cued. Stimulation follows automatically.');
		this._runSequence(this._noAutoStimConfig(this._sessionConfig), () => this._onSensorRunComplete());
	}

	_onSensorRunComplete() {
		const finalize = () => {
			this._sensorFinalizeTimer = null;
			this._clearSensorTimers();
			this._stopTimedStim();
			this._sensorStim = null;
			this.client.mvOnlineStop();          // persist the sensor run (raw + decisions)
			this._set(SESSION.SENSOR_DONE,
				'Sensor-triggered pass complete ✓. Start live control when ready.');
		};
		// The player idles the instant the final movement ends, so honor that bout's stim
		// hold — and keep capturing a moment longer so the no-stim head is seen recognizing
		// the return to rest — before finalizing.
		if (this._sensorStimOn || this._stimStopTimer) {
			this._clearSensorTimers();
			this._sensorStim = 'stim';
			this._stimStopTimer = setTimeout(() => {
				this._stimStopTimer = null;
				this._stopTimedStim();
				this._sensorStim = 'rest';
				this._emit();
				this._sensorFinalizeTimer = setTimeout(finalize, 1000);   // brief rest-capture tail
			}, this.postStimHoldMs);
			this._emit();
		} else {
			finalize();
		}
	}

	/** Operator: enter online closed-loop control. A fresh calibration runs FIRST (one rep
	 * of each movement, re-fitting the model right before live use); when it completes the
	 * GUIDED run starts automatically. Reached after the sensor pass, or by skipping it. */
	beginOnline() {
		if (!this._sessionConfig || !this._derived) return this._fail('Calibrate first.');
		this._clearSensorTimers();
		this._stopTimedStim();
		this._sensorStim = null;
		this._calibThenOnline = true;   // routes the calibration's completion into the guided run
		this._startCalibration('Quick re-calibration before live control — perform each movement once when cued.');
	}

	/** Start the GUIDED online run: each trained movement is cued once per round, round-robin
	 * in class order, until every movement has used up the repetitions it was given during
	 * training (e.g. 3 movements × 5 reps → the three cycle 5 times). A short rest + get-ready
	 * separates each cue; the recognized movement drives its own stim. Invoked once the
	 * pre-online calibration finishes (see the movement_status handler). */
	_startGuidedOnline() {
		if (!this._derived) return this._fail('Calibrate first.');
		this._enableClosedLoopStim();
		this.client.mvOnlineStart({
			label: 'online',
			session: (this._sessionConfig || this.baseConfig || {}).name || 'session',
			class_order: this._derived?.classOrder || [],
		});
		const guided = this._guidedOnlineConfig();
		if (guided) {
			this.mode?.setMode('patient');
			this.player.setTutorialEnabled(false);   // no tutorial; pacing comes from rest + prep
			this._set(SESSION.ONLINE, 'Recognition active — follow the cued movements.');
			this._runSequence(guided, () => this._onOnlineComplete());
		} else {
			this._set(SESSION.ONLINE, 'Recognition active — move naturally.');
		}
	}

	/** Build the GUIDED online config: each trained movement cued once per round,
	 * round-robin in class order, until each has consumed the reps it was given during
	 * training. Reps are summed per movement class from the session items; rest bouts are
	 * not cued. Tutorial is off and looping disabled; a rest (`onlineRestSec`) + get-ready
	 * (`onlinePrepSec`) separates each cue so reps don't run back-to-back. Auto-stim is off
	 * (the model drives stim via the closed loop). Returns null when there are no movement
	 * classes to cue (caller falls back to a free run). */
	_guidedOnlineConfig() {
		const cfg = this._sessionConfig || this.baseConfig;
		const d = this._derived;
		if (!cfg || !d || d.classKeys.length === 0) return null;
		const repsByKey = {};
		for (const it of d.items) {
			if (it.restClass === true) continue;
			const k = `${it.model}#${it.animation}`;
			repsByKey[k] = (repsByKey[k] || 0) + (it.repetitions || 1);
		}
		const maxReps = Math.max(0, ...d.classKeys.map((k) => repsByKey[k] || 0));
		const items = [];
		for (let r = 0; r < maxReps; r++) {
			for (const k of d.classKeys) {
				if (r < (repsByKey[k] || 0)) items.push({ ...d.firstItem[k], repetitions: 1 });
			}
		}
		if (items.length === 0) return null;
		const base = this._noAutoStimConfig(cfg);   // stimulation.enabled=false: the model drives stim
		return {
			...base,
			name: `${cfg.name || 'Sequence'} — guided online`,
			settings: {
				...(cfg.settings || {}),
				loopSequence: false,
				prepTime: this.onlinePrepSec,       // get-ready countdown before each cued movement
				restBetweenReps: 0,                 // (1 rep per item; the gap is pauseBetweenItems)
				pauseBetweenItems: this.onlineRestSec,
			},
			items,
		};
	}

	/** The guided online run cued every movement through its reps -> close the capture. */
	_onOnlineComplete() {
		this.stim?.disable();
		this.client.mvOnlineStop();          // persist the online run (raw + decisions)
		this.mode?.setMode('config');
		this._set(SESSION.DONE, 'Guided online run complete ✓ — every movement cued through its reps.');
	}

	/** Operator: skip the (rest of the) sensor pass and go straight to live control. */
	skipToOnline() {
		if (this.state !== SESSION.SENSOR_READY && this.state !== SESSION.SENSOR_SIM) return;
		const wasRunning = this.state === SESSION.SENSOR_SIM;
		this._clearSensorTimers();
		this._stopTimedStim();
		this._onSeqComplete = null;                 // don't let a late completion re-fire
		try { this.player.stop(); } catch (e) { /* noop */ }
		if (wasRunning) this.client.mvOnlineStop();  // close the sensor capture cleanly
		this.beginOnline();
	}

	/** Operator: end the session (stop streaming + stim, save the online run, unload). */
	endSession() {
		this._clearSensorTimers();
		this._stopTimedStim();
		this.stim?.disable();
		try { this.player.stop(); } catch (e) { /* noop */ }
		this.client.mvOnlineStop();          // persist the online run (raw + decisions)
		this.client.unloadMovementModel();
		this.mode?.setMode('config');
		this._set(SESSION.DONE, 'Session complete.');
	}

	/** Restart from the top (after done/error/halt). This is the ONLY thing that clears
	 * the emergency-stop latch — an explicit operator action, never a state transition. */
	reset() {
		// A halt froze the state machine at HALTED, but the capture that was open when
		// STOP was pressed is still open on the server. Close it against the state we
		// halted FROM, not against HALTED.
		const from = this._halted ? this._haltedFrom : this.state;
		this._clearSensorTimers();
		this._stopTimedStim({ force: this._halted });
		this.stim?.disable();
		try { this.player.stop(); } catch (e) { /* noop */ }
		// Release recording control AFTER stopping, so the stop above stays silent and
		// our mvRecordStop/mvOnlineStop remains the only capture boundary.
		this.player.setExternalRecording(false);
		// Close whichever capture is open, or the server keeps recording after we abort.
		// Recording blocks capture via mv_record_*; the online + sensor runs via
		// mv_online_*. mv_record_stop also persists whatever was captured before the
		// abort rather than discarding it.
		// No mvRecordResume() first: the server's pause flag only gates appending, and
		// stop works while paused — resuming would re-admit the tutorial-phase samples
		// we deliberately excluded, in the window before the stop lands.
		if (from === SESSION.RECORD_TRAIN || from === SESSION.RECORD_CLEAN
			|| from === SESSION.RECORD_CALIB) {
			this._mvPaused = false;
			this.client.mvRecordStop();
		}
		// Save any live run — both free online and the sensor pass capture via mv_online.
		if (from === SESSION.ONLINE || from === SESSION.SENSOR_SIM) this.client.mvOnlineStop();
		this._sensorStim = null;
		this._calibThenOnline = false;
		this._halted = false;
		this._haltedFrom = null;
		this.mode?.setMode('config');
		this._derived = null;
		this._calibSequence = null;
		this.summary = null;
		this.error = null;
		this._set(SESSION.WELCOME, '');
	}

	/**
	 * Operator emergency stop — TERMINAL. Halts stimulation, stops the player, and
	 * latches `_halted` so nothing re-arms stim until the operator calls reset().
	 *
	 * Both stops are UNCONDITIONAL: a stale `_sensorStimOn`/`stimming === false` (a
	 * reconnect reinitialises them, and another driver may own the live train) must
	 * never suppress the command (audit S5). Cheap duplicate stops are the correct
	 * trade against a train that keeps running.
	 */
	emergencyStop() {
		this._halted = true;
		if (this._haltedFrom == null) this._haltedFrom = this.state;
		this._clearSensorTimers();
		this._stopTimedStim({ force: true });   // the sensor pass drives the stimulator directly
		this.stim?.emergencyStop();             // the closed loop drives it through the controller
		// Stop the cueing too: without this the sequence keeps running and the next MOVE
		// phase re-arms the timed stim 1-2 s later.
		this._onSeqComplete = null;             // a late completion must not restart anything
		try { this.player.stop(); } catch (e) { /* noop */ }
		this._sensorStim = 'rest';
		this._set(SESSION.HALTED,
			'EMERGENCY STOP — stimulation halted and cueing stopped. Reset to start a new session.');
	}

	// -- sequence running ---------------------------------------------------------
	_runSequence(config, onComplete) {
		// Terminal halt: no block may be cued (and therefore no stim armed) until the
		// operator resets. Nothing in the halted view offers this, so reaching it means
		// a stale control — refuse loudly rather than silently re-arming.
		if (this._halted) {
			this._raiseAlarm('Refused to start a block: the session is HALTED — reset it first.');
			this._emit();
			return;
		}
		this._onSeqComplete = onComplete;
		this._mvPaused = false;         // capture resumes fresh for each block
		// We bracket every block with our own mv_record_*/mv_online_* capture, so the
		// player must not also fire start_recording/stop_recording. Released when the
		// block completes (_subscribe) or the session is torn down (reset).
		this.player.setExternalRecording(true);
		this.player.loadConfig(config);
		this.player.play();
	}

	_onTrainRecorded() {
		this.client.mvRecordStop();
		this.client.mvTrain({
			classSequence: this._derived.classSequence,
			classOrder: this._derived.classOrder,
			params: this.params,
		});
		// Don't block on training — the server trains in the BACKGROUND while the
		// operator records the no-stim block. Advance straight to the next step.
		this._set(SESSION.TRAINED,
			'Stim recording done ✓ — training in the background. Start the no-stim pass now.');
	}

	_onCleanRecorded() {
		this.client.mvRecordStop();
		this.client.mvTrainClean({
			classSequence: this._derived.classSequenceNoRest,
			classOrder: this._derived.classOrder,
		});
		// Clean model trains in the background while the operator records calibration.
		this._set(SESSION.TRAINED_CLEAN,
			'No-stim recording done ✓ — training in the background. Start calibration now.');
	}

	/** No-stim variant of a config: patient channels at amplitude 0, plus the trigger
	 * channel firing so ch3 still records the trigger for bout segmentation. */
	_nostimConfig(config) {
		const trig = this.nostimTrigger || config.stimulation?.nostimTrigger;
		const withTrig = (chs = []) => {
			const zeroed = chs.map((c) => ({ ...c, amplitude: 0 }));
			if (trig && !zeroed.some((c) => c.id === trig.id)) zeroed.push({ ...trig });
			return zeroed;
		};
		const stimulation = config.stimulation
			? { ...config.stimulation, enabled: true, channels: withTrig(config.stimulation.channels) }
			: null;
		// Stim-rest bouts belong to the STIM block only — drop them from the no-stim pass.
		const items = (config.items || [])
			.filter((it) => it.restClass !== true)
			.map((it) => (it.stim
				? { ...it, stim: { ...it.stim, channels: withTrig(it.stim.channels) } }
				: it));
		return { ...config, name: `${config.name || 'Sequence'} — no-stim`, stimulation, items };
	}

	_onCalibRecorded() {
		this.client.mvRecordStop();
		this._set(SESSION.CALIBRATING, 'Finalizing calibration…');
		this.client.mvCalibrate({ classSequence: this._calibSequence, savePath: this.savePath });
	}

	// -- sensor-triggered pass: timed stim ----------------------------------------
	/** A no-auto-stim variant of a config: the player must NOT fire stim on the move
	 * phase (we drive stim ourselves on a timer). Everything else is preserved. */
	_noAutoStimConfig(config) {
		const stimulation = config.stimulation
			? { ...config.stimulation, enabled: false }
			: config.stimulation;
		return { ...config, name: `${config.name || 'Sequence'} — sensor run`, stimulation };
	}

	/** The movement label currently cued by the player (maps its item -> class label). */
	_currentCuedLabel() {
		const it = this.player.getState()?.currentItem;
		const d = this._derived;
		if (!it || !d) return null;
		const idx = d.classKeys.indexOf(`${it.model}#${it.animation}`);
		return idx >= 0 ? d.classOrder[idx] : null;
	}

	/** Append the trigger stim channel so the trigger fires while we stimulate — this is
	 * what makes the engine route those windows to the STIM head (blanking present). */
	_withTriggerChannel(channels = []) {
		const trig = this.nostimTrigger || this._config()?.stimulation?.nostimTrigger;
		const out = [...channels];
		if (trig && !out.some((c) => c.id === trig.id)) out.push({ ...trig });
		return out;
	}

	/** Phase transitions during the sensor pass -> schedule the timed stim. Stim starts
	 * `preStimDelayMs` after the movement begins and stops `postStimHoldMs` after it ends. */
	_onSensorPhase(prev, phase) {
		if (this._halted) return;   // emergency stop is terminal: never re-arm from a phase change
		const enteringMove = phase === PHASE.MOVE && prev !== PHASE.MOVE;
		const leavingMove = prev === PHASE.MOVE && phase !== PHASE.MOVE;
		if (enteringMove) {
			// New bout: clean slate, then no-stim window, then arm the stim onset.
			this._clearSensorTimers();
			this._stopTimedStim();
			this._sensorLabel = this._currentCuedLabel();
			this._sensorStim = 'prestim';
			this._stimStartTimer = setTimeout(() => this._startTimedStim(this._sensorLabel), this.preStimDelayMs);
			this._emit();
		} else if (leavingMove) {
			// Movement ended -> keep stimulating for the hold, then release to rest.
			this._stimStopTimer = setTimeout(() => {
				this._stimStopTimer = null;
				this._stopTimedStim();
				this._sensorStim = 'rest';
				this._emit();
			}, this.postStimHoldMs);
		}
	}

	/** Channels to fire for a cued movement in the sensor pass: its per-movement pattern
	 * if defined, else the config's GLOBAL recording stim channels (same fallback order
	 * the player uses during training). Returns null if neither is configured. */
	_sensorStimChannels(label) {
		const { patterns } = this._resolvePatterns();
		const pat = label && patterns[label];
		if (pat?.channels?.length) return pat.channels;
		const cfg = this._sessionConfig || this.baseConfig || {};
		const global = cfg.stimulation?.channels;
		return global?.length ? global : null;
	}

	/** Fire the cued movement's stim pattern (+ trigger channel) like a sensor would. */
	_startTimedStim(label) {
		this._stimStartTimer = null;
		if (this._halted) return;   // emergency stop is terminal: never arm stim again
		const { stimCfg } = this._resolvePatterns();
		const channels = this._sensorStimChannels(label);
		if (!channels?.length) {
			this.trainLog.push({ t: Date.now(), level: 'warn',
				line: `Sensor: no per-movement or global stim channels for "${label || '?'}" — skipping stim for this bout.` });
			this._sensorStim = 'stim';   // reflect intent even when nothing is wired
			this._emit();
			return;
		}
		const sent = this.client.stimulateStart({
			channels: this._withTriggerChannel(channels),
			stimulatorType: stimCfg.stimulatorType,
			port: stimCfg.port,
			controllerUrl: stimCfg.controllerUrl,
		});
		if (!sent) return;   // never left the socket: nothing is running, so claim nothing
		this._sensorStimOn = true;
		this._sensorStim = 'stim';
		this._armHardStop();
		this._emit();
	}

	/**
	 * Redundant, unconditional stop for the train just started — the backstop for a
	 * leave-MOVE transition that never arrives. Cleared by every normal stop path.
	 *
	 * Firing is a FAULT, not a normal boundary: the phase machinery that should have
	 * ended this train is not working, so the very next MOVE phase would arm another
	 * one. It therefore halts the whole session (same terminal latch as the operator's
	 * emergency stop) rather than just cutting this train — re-arming needs reset().
	 *
	 * Because the consequence is terminal, the deadline is derived from THIS bout's
	 * planned duration rather than a fixed cap: we schedule the real stop ourselves, so
	 * we know when it is due. `maxTimedStimMs` is only the fallback for a bout whose
	 * clip duration cannot be read.
	 */
	_armHardStop() {
		this._clearHardStopTimer();
		const deadline = this._hardStopDeadlineMs();
		this._hardStopTimer = setTimeout(() => {
			this._hardStopTimer = null;
			if (!this._sensorStimOn) return;
			this._raiseAlarm(`STIM WATCHDOG — timed train ran past its ${deadline} ms deadline. `
				+ 'Stop forced and the session HALTED; reset to continue.');
			this.emergencyStop();
		}, deadline);
	}

	/**
	 * When the train started now is due to end, plus a margin. The bout runs from the
	 * pre-stim delay to the end of the movement, then `postStimHoldMs`; the movement's
	 * length is the cued clip's duration at the configured playback speed, extended by
	 * the slowest segment multiplier in play (segments can deliberately slow a clip
	 * down, and under-estimating here would halt a healthy session).
	 */
	_hardStopDeadlineMs() {
		const margin = this.hardStopMarginMs + this.postStimHoldMs;
		const wv = this.player?.webglView;
		const clipSec = wv?.animationDuration;
		if (!(clipSec > 0)) return this.maxTimedStimMs + margin;   // duration unknown
		const speed = wv.playbackSpeed > 0 ? wv.playbackSpeed : 1;
		const mults = Object.values(wv.segmentManager?.speeds || {}).filter((n) => n > 0);
		const slowest = Math.min(1, ...(mults.length ? mults : [1]));
		const moveMs = (clipSec / speed / slowest) * 1000;
		// The pre-stim delay has already elapsed when this is armed, so only the rest of
		// the movement is still to come.
		return Math.max(0, moveMs - this.preStimDelayMs) + margin;
	}

	/**
	 * Stop the timed stim train (idempotent).
	 *
	 * `_sensorStimOn` is cleared ONLY on a confirmed send: a dead socket leaves the
	 * train running server-side, so the flag stays set, the stop is latched for the
	 * next connect and the operator is alarmed (fend plan A2).
	 *
	 * @param {object} [opts]
	 * @param {boolean} [opts.force] - send regardless of local belief (emergency stop)
	 * @returns {boolean} whether the command left the socket
	 */
	_stopTimedStim({ force = false } = {}) {
		if (!force && !this._sensorStimOn && !this._pendingStimStop) {
			this._clearHardStopTimer();
			return true;
		}
		const { stimCfg } = this._resolvePatterns();
		const sent = this.client.stimulateStop({ controllerUrl: stimCfg.controllerUrl });
		if (sent) {
			this._sensorStimOn = false;
			this._pendingStimStop = false;
			this._clearHardStopTimer();
		} else {
			// The train is still presumed running: do NOT tear down the timer that is the
			// only thing bounding it. Re-purpose it as a slow retry until the stop lands
			// or the socket returns. One alarm per failure episode; retries are silent.
			const first = !this._pendingStimStop;
			this._pendingStimStop = true;
			this._armStopRetry();
			if (first) {
				this._raiseAlarm('STOP NOT SENT — no connection to the server; stimulation may still be running');
			}
		}
		return sent;
	}

	/** Keep chasing a stop that never left the socket (the train is presumed live). */
	_armStopRetry() {
		this._clearHardStopTimer();
		this._hardStopTimer = setTimeout(() => {
			this._hardStopTimer = null;
			if (this._pendingStimStop) this._stopTimedStim({ force: true });
		}, this.stopRetryMs);
	}

	_clearHardStopTimer() {
		if (this._hardStopTimer) { clearTimeout(this._hardStopTimer); this._hardStopTimer = null; }
	}

	_clearSensorTimers() {
		if (this._stimStartTimer) { clearTimeout(this._stimStartTimer); this._stimStartTimer = null; }
		if (this._stimStopTimer) { clearTimeout(this._stimStopTimer); this._stimStopTimer = null; }
		if (this._sensorFinalizeTimer) { clearTimeout(this._sensorFinalizeTimer); this._sensorFinalizeTimer = null; }
		this._clearHardStopTimer();
	}

	// -- subscriptions ------------------------------------------------------------
	_subscribe() {
		this.player.onSequenceComplete(() => {
			const cb = this._onSeqComplete;
			this._onSeqComplete = null;
			// The block is over — hand recording control back to the player so standalone
			// playback records normally again. The player's own stopRecording guard has
			// already run by this point, so this cannot resurrect a duplicate recording.
			this.player.setExternalRecording(false);
			if (cb) cb();
		});
		// Re-emit during recording so the view can show live cue + progress.
		const isRecording = () => this.state === SESSION.RECORD_TRAIN
			|| this.state === SESSION.RECORD_CLEAN || this.state === SESSION.RECORD_CALIB;
		const bump = () => { if (isRecording()) this._emit(); };
		// The TUTORIAL still plays for the patient, but we pause capture during it so
		// it is neither recorded nor used for training (resume when it ends).
		this.player.onPhaseChange((phase) => {
			bump();
			// Sensor pass: drive the timed stim off the cued movement's phase transitions.
			if (this.state === SESSION.SENSOR_SIM) this._onSensorPhase(this._prevPhase, phase);
			this._prevPhase = phase;
			if (!isRecording()) return;
			if (phase === 'tutorial' && !this._mvPaused) {
				this._mvPaused = true; this.client.mvRecordPause();
			} else if (phase !== 'tutorial' && this._mvPaused) {
				this._mvPaused = false; this.client.mvRecordResume();
			}
		});
		this.player.onChange(bump);

		// A timed-stim stop that never left the socket is retried as soon as there is a
		// connection again — the train it was meant to end is still running.
		this.client.onConnect?.(() => {
			if (this._pendingStimStop) this._stopTimedStim({ force: true });
		});
		this.client.onDisconnect?.(() => this._onDisconnect());

		this.client.onMvTrainStatus((msg) => this._onTrainStatus(msg));
		this.client.onConfigStatus?.((msg) => this._onConfigStatus(msg));
		this.client.onError?.((msg) => {
			if (typeof msg === 'string' && /config/i.test(msg)) { this.deviceInfo = msg; this._emit(); }
		});
		this.client.onMovementStatus((s) => {
			// Calibration finished -> the model auto-loads. If this was the fresh calibration
			// requested right before live control, launch the guided online run now; otherwise
			// it's the post-training calibration -> offer the sensor pass. See beginOnline.
			if (s.active && this.state === SESSION.CALIBRATING) {
				if (this._calibThenOnline) {
					this._calibThenOnline = false;
					this._startGuidedOnline();
				} else {
					this._set(SESSION.SENSOR_READY,
						'Calibration done ✓ — model loaded. Next: the sensor-triggered run.');
				}
			}
		});
	}

	/** States in which a dropped connection has actually cost the operator something:
	 * a block being cued, a live run, or server-side work whose result we will now miss. */
	static INTERRUPTIBLE = new Set([
		SESSION.RECORD_TRAIN, SESSION.RECORD_CLEAN, SESSION.RECORD_CALIB,
		SESSION.TRAINING, SESSION.TRAINING_CLEAN, SESSION.CALIBRATING,
		SESSION.SENSOR_SIM, SESSION.ONLINE,
	]);

	/**
	 * The socket dropped. This is RECOVERABLE and deliberately does NOT use the HALTED
	 * latch: HALTED means "stimulation was stopped in an emergency and nothing may
	 * re-arm until an operator says so", and diluting it with routine network trouble
	 * would make the one screen that matters ambiguous. A disconnect goes to the
	 * ordinary error surface, which already offers Restart -> reset().
	 *
	 * The player raises the operator-visible RECORDING INTERRUPTED alarm (it is the
	 * component that knows a run was in progress, and it covers standalone playback
	 * too), so this path does not duplicate it on the banner.
	 */
	_onDisconnect() {
		if (!MovementSessionController.INTERRUPTIBLE.has(this.state)) return;
		const was = this.state;
		this._clearSensorTimers();
		// If a timed train was firing, this cannot send — it latches the retry for the
		// reconnect and raises its own (different, higher-priority) alarm.
		this._stopTimedStim();
		this._sensorStim = null;
		this._mvPaused = false;
		this.trainLog.push({ t: Date.now(), level: 'warn', line: `disconnected during "${was}"` });
		this._fail('Connection lost during the block — cueing stopped. The server saves whatever '
			+ 'was captured; reconnect, then Restart to run this block again.');
	}

	_onConfigStatus(msg) {
		if (msg.nch == null) { this.deviceInfo = msg.message || ''; this._emit(); return; }
		const exposed = msg.n_exposed ?? msg.nch;
		const aux = exposed - (msg.n_emg ?? 0);
		this.deviceInfo = `${exposed}/${msg.nch} ch · ${msg.n_emg ?? '?'} EMG + ${aux} aux @ ${msg.fsamp ?? '?'}Hz`
			+ (aux <= 0 ? ' — NO AUX (relaunch without --emg-only)' : '');
		if (msg.n_emg && aux > 0) {   // auto-fit montage + trigger from the loaded config
			// FLOOR, not round. Grids are whole 64-electrode arrays, so n_emg should be an
			// exact multiple of perGrid; if it isn't, the config is already wrong. Rounding
			// UP would extend the last grid past the EMG block and into the aux channels —
			// pulling the trigger itself into the feature set, which is the one channel
			// that must stay out of it (it routes the stim head). Flooring can only leave
			// trailing EMG channels unused, which is wasteful but harmless.
			const perGrid = this.montage.perGrid;
			if (msg.n_emg % perGrid !== 0) {
				console.warn(`MovementSession: ${msg.n_emg} EMG channels is not a multiple of the `
					+ `${perGrid}-electrode grid size — using ${Math.floor(msg.n_emg / perGrid)} `
					+ `grid(s) and ignoring the remaining ${msg.n_emg % perGrid} channel(s). `
					+ `Check the device config.`);
			}
			this.montage.nGrids = Math.max(1, Math.floor(msg.n_emg / perGrid));
			this._applyMontage();
			// _applyMontage derives the trigger from perGrid*nGrids, which only equals
			// n_emg when n_emg is an exact multiple of the grid size. The device's own
			// n_emg is the authoritative first-aux index, and it is what GUIView shows
			// the operator — take it, so the channel displayed is the one mv_train gets.
			// (Both components subscribe to config_status; this one runs second and used
			// to silently overwrite GUIView's value.)
			this.params.trigCh = msg.n_emg;
		}
		this._emit();
	}

	_onTrainStatus(msg) {
		this._logTrain(msg);
		if (msg.state === 'error') return this._fail(msg.message || 'Training error.');
		// Training runs in the background; these completions must NOT regress the UI if
		// the operator has already advanced to the next recording. Only refresh the
		// label when we're still on the matching screen.
		if (msg.state === 'trained') {
			this.summary = msg;
			if (this.state === SESSION.TRAINED) {
				const nMov = (this._derived?.classOrder?.length || 1) - 1;
				this._set(SESSION.TRAINED,
					`Stim model trained ✓ — ${nMov} movements, ${msg.n_windows} windows. Start the no-stim pass.`);
			}
		}
		if (msg.state === 'trained_clean') {
			this.summary = msg;
			if (this.state === SESSION.TRAINED_CLEAN) {
				this._set(SESSION.TRAINED_CLEAN,
					`Clean model trained ✓ — ${msg.n_windows} windows. Start calibration.`);
			}
		}
		// 'calibrated' is followed by movement_status(active) -> ONLINE (above).
	}

	/** Resolve the per-movement stim patterns + backend from the LIVE session config.
	 * Per-movement patterns, keyed by movement label, come from three sources in order
	 * of increasing precedence:
	 *   0) the global stimulation.channels — the recording default, used for any movement
	 *      that defines no pattern of its own (rest is never given a default pattern).
	 *   1) onlineStimulation.patterns: { <label>: { channels:[...] } }
	 *   2) a per-movement `stim: { channels:[...] }` on the item (GUI editor) — the same
	 *      channels used during recording. (2) overrides (1) overrides (0).
	 * Shared by the online closed loop AND the sensor-triggered pass. */
	_resolvePatterns() {
		const cfg = this._sessionConfig || this.baseConfig || {};
		const os = cfg.onlineStimulation || {};
		const d = this._derived;
		const patterns = {};
		// (0) Seed every movement class with the global recording channels as a fallback.
		const globalCh = cfg.stimulation?.channels;
		if (d && globalCh?.length) {
			for (const label of d.classOrder) {
				if (label !== 'rest') patterns[label] = { channels: globalCh };
			}
		}
		Object.assign(patterns, os.patterns || {});   // (1) named online patterns override
		if (d) {
			d.classKeys.forEach((k, i) => {
				const item = d.firstItem[k];
				if (item?.stim?.channels?.length) patterns[d.classOrder[i]] = item.stim;   // (2) per-item wins
			});
		}
		// Stim backend: onlineStimulation's, else reuse the recording stimulation's.
		const stimCfg = {
			stimulatorType: os.stimulatorType ?? cfg.stimulation?.stimulatorType,
			controllerUrl: os.controllerUrl ?? cfg.stimulation?.controllerUrl,
			port: os.port ?? cfg.stimulation?.port,
		};
		return { patterns, stimCfg, classOrder: d?.classOrder || [], enabled: !!os.enabled,
			latchMs: os.latchMs };
	}

	/** Configure + enable closed-loop stim from the config's onlineStimulation. */
	_enableClosedLoopStim() {
		if (!this.stim) return;
		const { patterns, stimCfg, classOrder, enabled, latchMs } = this._resolvePatterns();
		if (!enabled && Object.keys(patterns).length === 0) return;   // nothing to fire
		this.stim.configure({ classOrder, patterns, stimCfg, latchMs });
		this.stim.enable();
	}

	// -- class / sequence derivation from the config ------------------------------
	_deriveClasses(config) {
		const items = (config.items || []).filter((it) => it.stimulate !== false);
		const keyOf = (it) => `${it.model}#${it.animation}`;
		// A `restClass` item is a STIMULATED bout (fires stim + trigger, cued like any
		// movement) that is LABELED rest — it feeds the stim head genuine "stim on + at
		// rest" examples. It defines no movement class and is skipped by calibration.
		const isRest = (it) => it.restClass === true;

		const keys = [];
		const firstItem = {};
		let restItem = null;                 // first stim-rest item (for the calib rest bout)
		for (const it of items) {
			if (isRest(it)) { if (!restItem) restItem = it; continue; }
			const k = keyOf(it);
			if (!(k in firstItem)) { keys.push(k); firstItem[k] = it; }
		}
		const classOrder = this._labels(keys.map((k) => firstItem[k])).concat(['rest']);
		const restIdx = keys.length;                 // 'rest' is appended last
		const idxOf = {};
		keys.forEach((k, i) => { idxOf[k] = i; });

		const classSequence = [];           // stim block: movements + stim-rest bouts
		const classSequenceNoRest = [];     // no-stim block: movements only (stim-rest is stim-only)
		for (const it of items) {
			const rest = isRest(it);
			const idx = rest ? restIdx : idxOf[keyOf(it)];
			for (let r = 0; r < (it.repetitions || 1); r++) {
				classSequence.push(idx);
				if (!rest) classSequenceNoRest.push(idx);
			}
		}
		return { classOrder, classSequence, classSequenceNoRest, classKeys: keys, firstItem, restItem, items };
	}

	/** Readable, de-duplicated class labels from items (model base, +anim if clashing). */
	_labels(items) {
		const base = items.map((it) => (it.label || it.model || 'mov').replace(/\.glb$/i, ''));
		const counts = {};
		for (const b of base) counts[b] = (counts[b] || 0) + 1;
		return base.map((b, i) => (counts[b] > 1 ? `${b}-${items[i].animation}` : b));
	}

	/** One-rep-each calibration config + its per-bout class sequence [0,1,2,…]. */
	_calibConfig(config, derived) {
		const items = derived.classKeys.map((k) => ({ ...derived.firstItem[k], repetitions: 1 }));
		const sequence = derived.classKeys.map((_, i) => i);
		// One STIMULATED-REST bout in calibration too, so the stim head's CORAL target
		// (the calibration domain) represents rest with stimulated rest — the regime it
		// trains and runs on — not just unstimulated between-bout gaps.
		if (derived.restItem) {
			items.push({ ...derived.restItem, repetitions: 1 });
			sequence.push(derived.classOrder.length - 1);   // rest = last class index
		}
		const cfg = {
			...config,
			name: `${config.name || 'Sequence'} — calibration`,
			settings: { ...(config.settings || {}), loopSequence: false },
			items,
		};
		return { config: cfg, sequence };
	}

	// -- state emit ---------------------------------------------------------------
	/** The single chokepoint for state transitions — and therefore where HALTED is made
	 * terminal. While the latch is set nothing may transition out of it; only reset(),
	 * which clears the latch first, can. This is what keeps the STOP confirmation on
	 * screen (fend plan B5) instead of being overwritten by the next status message. */
	_set(state, message = '') {
		if (this._halted && state !== SESSION.HALTED) {
			this.trainLog.push({ t: Date.now(), level: 'warn',
				line: `Ignored transition to "${state}" — session is HALTED (reset to continue).` });
			this._emit();
			return;
		}
		this.state = state;
		this.message = message;
		if (state !== SESSION.ERROR) this.error = null;
		this._emit();
	}

	_fail(message) {
		this.error = message;
		if (this._halted) {   // a halt outranks a background failure; keep the STOP on screen
			this.trainLog.push({ t: Date.now(), level: 'error', line: `ERROR while halted: ${message}` });
			this._emit();
			return;
		}
		this.state = SESSION.ERROR;
		this.message = message;
		this._emit();
	}

	_emit() {
		const snap = this.getState();
		for (const cb of this._listeners) {
			try { cb(snap); } catch (e) { console.error('MovementSession listener error', e); }
		}
	}
}
