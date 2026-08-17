/**
 * Disconnect handling — the session layer must learn the connection dropped.
 * ==========================================================================
 *
 * A drop mid-recording used to be invisible on this side of the boundary: the subject
 * kept performing every cued rep into a void, the RECORDING indicator kept pulsing,
 * and the block was lost with no signal to anyone (plan B1). The backend now finalizes
 * the partial recording when the last client disconnects, so the data survives — this
 * layer's job is to tell the operator the truth about it.
 *
 * Verifies:
 *   - a drop mid-sequence halts cueing WITHOUT sending anything (the socket is gone and
 *     the server finalizes on its side), preserves position, and raises the
 *     RECORDING INTERRUPTED alarm exactly once;
 *   - a drop while idle raises nothing;
 *   - MovementSessionController lands in the recoverable ERROR state — NOT the terminal
 *     HALTED latch, which is reserved for stimulation emergencies — and reset() recovers;
 *   - a drop outside a block is ignored;
 *   - a stim emergency already in force outranks a later disconnect (HALTED sticks).
 *
 * Drives the real SequencePlayer and MovementSessionController against a fake EMG
 * client (see harness.mjs).
 *
 *   node fend/tests/run-all.mjs          (or: npm test, from fend/)
 *   node fend/tests/disconnect-recovery.test.mjs
 */
import { createSuite, fakeEmgClient, loadSrc, isEntryPoint, runStandalone } from './harness.mjs';

export default async function run() {
	const { default: SequencePlayer, PHASE } = await loadSrc('ui/SequencePlayer.js');
	const { default: MovementSessionController, SESSION } = await loadSrc('emg/MovementSessionController.js');
	const { default: EMGClient } = await loadSrc('emg/EMGClient.js');
	const t = createSuite('disconnect-recovery');

	const tick = () => new Promise((r) => setImmediate(r));
	const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

	/** The minimum WebGLView surface `play()` touches, with a model load we control. */
	const fakeWebGL = () => {
		class FakeWebGLView {
			static AVAILABLE_MODELS = [{ file: 'a.glb' }, { file: 'b.glb' }];
			constructor() {
				this.currentModelIndex = 0;
				this.playbackSpeed = 1;
				this.playAnimationCalls = 0;
				this.pendingLoad = null;
				this.action = { setLoop() {}, reset() {}, clampWhenFinished: false, paused: false, time: 0 };
				this.mixer = { addEventListener() {}, removeEventListener() {}, update() {} };
			}
			playAnimation() { this.playAnimationCalls++; }
			onceModelLoaded(cb) { this.pendingLoad = cb; }
			loadModel() {}
			resolveLoad() { const cb = this.pendingLoad; this.pendingLoad = null; cb?.(); }
			play() {}
			pause() {}
		}
		return new FakeWebGLView();
	};

	/** A socket stand-in, so EMGClient.connect() wires its OWN onopen/onclose. */
	const installFakeWebSocket = () => {
		class FakeWebSocket {
			static OPEN = 1;
			static last = null;
			constructor(url) { this.url = url; this.readyState = 1; FakeWebSocket.last = this; }
			send() {}
			close() { this.readyState = 3; this.onclose?.(); }
		}
		const real = globalThis.WebSocket;
		globalThis.WebSocket = FakeWebSocket;
		t.defer(() => { globalThis.WebSocket = real; });
		return FakeWebSocket;
	};

	/** A fake client that also fans out disconnects, like EMGClient._notifyDisconnect. */
	const clientWithDisconnect = () => {
		const c = fakeEmgClient();
		c._disc = [];
		c.onDisconnect = (cb) => c._disc.push(cb);
		c.disconnect = () => { c.isConnected = false; c.sendOk = false; for (const cb of c._disc) cb(); };
		return c;
	};

	// ==== the player stops cueing ==============================================
	t.section('a drop mid-sequence stops the cueing');
	{
		const c = clientWithDisconnect();
		const p = new SequencePlayer();
		p.onEMGClient = c;
		const alarms = []; p.onAlarm((m) => alarms.push(m));
		c.onDisconnect(() => p.haltForDisconnect());
		p.config = {
			stimulation: { enabled: false },
			items: [{ model: 'm', animation: 0, repetitions: 3 }, { model: 'm2', animation: 0, repetitions: 1 }],
			settings: {},
		};
		p.isPlaying = true;
		p.currentItemIndex = 1;
		p.currentRepetition = 2;
		p.currentPhase = PHASE.MOVE;

		const sentBefore = c.sent.length;
		c.disconnect();
		t.check('cueing halted', p.isPlaying === false);
		t.check('phase returned to IDLE', p.currentPhase === PHASE.IDLE);
		t.check('nothing was sent over the dead socket', c.sent.length === sentBefore);
		t.check('position preserved (a disconnect is recoverable)',
			p.currentItemIndex === 1 && p.currentRepetition === 2);
		t.check('interruption alarm raised exactly once',
			alarms.filter((a) => /RECORDING INTERRUPTED/.test(a)).length === 1);
		t.check('the alarm is honest about the data and says what to do next',
			/kept on the server/.test(alarms[0]) && /Reset before re-running/.test(alarms[0])
			&& !/saved|finalized/i.test(alarms[0]));

		c.disconnect();   // a second drop while already halted
		t.check('no duplicate alarm when nothing was running',
			alarms.filter((a) => /RECORDING INTERRUPTED/.test(a)).length === 1);
	}
	{
		const c = clientWithDisconnect();
		const p = new SequencePlayer();
		p.onEMGClient = c;
		const alarms = []; p.onAlarm((m) => alarms.push(m));
		c.onDisconnect(() => p.haltForDisconnect());
		p.config = { items: [{ model: 'm', animation: 0, repetitions: 1 }], settings: {} };
		t.check('a drop while idle interrupts nothing', p.haltForDisconnect() === false);
		c.disconnect();
		t.check('a drop while idle raises no alarm', alarms.length === 0);
	}

	// ==== the session controller fails the block, recoverably ==================
	t.section('the session fails the block: ERROR, not the terminal HALTED latch');
	const session = (state) => {
		const client = clientWithDisconnect();
		const player = {
			config: null, webglView: null,
			onSequenceComplete() {}, onPhaseChange() {}, onChange() {},
			getState: () => ({}),
			stop() {}, loadConfig() {}, play() {}, setExternalRecording() {},
		};
		const c = new MovementSessionController(client, player, {});
		t.defer(() => c._clearSensorTimers());
		c.state = state;
		return { client, c };
	};
	{
		const { client, c } = session(SESSION.RECORD_TRAIN);
		client.disconnect();
		t.check('block failed visibly', c.state === SESSION.ERROR);
		t.check('NOT the terminal stim latch', c.state !== SESSION.HALTED && c._halted === false);
		t.check('the error is honest about the data and says what to do next',
			/kept\s+on the server/.test(c.error) && /Restart/.test(c.error)
			&& !/saved|finalized/i.test(c.error));
		t.check('the drop is in the training log',
			c.trainLog.some((l) => /disconnected during "record_train"/.test(l.line)));
		c.reset();
		t.check('reset() recovers to WELCOME', c.state === SESSION.WELCOME && c.error === null);
	}
	{
		const { client, c } = session(SESSION.ONLINE);
		client.disconnect();
		t.check('a live run is also failed', c.state === SESSION.ERROR);
	}
	{
		const { client, c } = session(SESSION.WELCOME);
		client.disconnect();
		t.check('a drop outside a block is ignored', c.state === SESSION.WELCOME);
	}
	{
		const { client, c } = session(SESSION.SENSOR_SIM);
		c.emergencyStop();                       // stim emergency first
		t.check('emergency stop halted the session', c.state === SESSION.HALTED);
		client.disconnect();                     // then the socket drops
		t.check('a stim emergency outranks a later disconnect', c.state === SESSION.HALTED);
	}

	// ==== the orphaned capture must be closed on recovery ======================
	t.section('reset() after a disconnect-ERROR closes the capture it left open');
	{
		const { client, c } = session(SESSION.RECORD_TRAIN);
		client.disconnect();
		t.check('failure remembered what it interrupted', c._failedFrom === SESSION.RECORD_TRAIN);
		client.isConnected = true; client.sendOk = true;      // operator reconnects
		c.reset();
		t.check('reset() closes the orphaned mv capture', client.didSend('mv_record_stop'));
		t.check('ready to run the next block', c.state === SESSION.WELCOME);
		t.check('the failure marker is cleared', c._failedFrom === null);
	}
	{
		const { client, c } = session(SESSION.SENSOR_SIM);
		client.disconnect();
		client.isConnected = true; client.sendOk = true;
		c.reset();
		t.check('a live run closes its online capture', client.didSend('mv_online_stop'));
	}
	{
		const { client, c } = session(SESSION.WELCOME);
		c.reset();
		t.check('an untouched session closes nothing',
			!client.didSend('mv_record_stop') && !client.didSend('mv_online_stop'));
	}

	// ==== the player, driven for real ==========================================
	t.section('a drop during an ACTIVE stim train: the stim warning is the last word');
	{
		const c = clientWithDisconnect();
		const p = new SequencePlayer();
		p.onEMGClient = c;
		const alarms = []; p.onAlarm((m) => alarms.push(m));
		c.onDisconnect(() => p.haltForDisconnect());
		p.setWebGLView(fakeWebGL());
		p.loadConfig({
			stimulation: { enabled: true, channels: [{ id: 1, amplitude: 8 }] },
			items: [{ model: 'a.glb', animation: 0, repetitions: 2, stimulate: true }],
			settings: { prepTime: 0 },
		});
		p.play();
		await tick();
		t.check('a train is running', p._stimActive === true && c.starts() === 1);

		c.disconnect();
		t.check('cueing halted mid-train', p.isPlaying === false);
		t.check('the interruption was reported once',
			alarms.filter((a) => /RECORDING INTERRUPTED/.test(a)).length === 1);
		t.check('the stim warning is the LAST alarm, so the banner shows it',
			/STOP NOT SENT/.test(alarms[alarms.length - 1]));
		t.check('the possibly-live train is still believed',
			p._stimActive === true && p._pendingStop === true);
	}

	t.section('one real drop, both App-wired handlers');
	{
		// Exactly the wiring App.js installs, driven through the real EMGClient's own
		// onclose: the player halts the cueing, the session fails the block.
		const FakeWebSocket = installFakeWebSocket();
		const client = new EMGClient({ autoReconnect: false });
		const player = new SequencePlayer();
		player.onEMGClient = client;
		player.setWebGLView(fakeWebGL());
		const alarms = [];                       // the banner sees BOTH alarm channels
		player.onAlarm((m) => alarms.push(m));
		client.onDisconnect(() => player.haltForDisconnect());
		const sess = new MovementSessionController(client, player, {});
		t.defer(() => sess._clearSensorTimers());
		sess.onAlarm((m) => alarms.push(m));
		sess.state = SESSION.RECORD_TRAIN;

		client.connect();
		FakeWebSocket.last.onopen();
		player.loadConfig({
			stimulation: { enabled: false },
			items: [{ model: 'a.glb', animation: 0, repetitions: 2 }],
			settings: { prepTime: 0 },
		});
		player.play();
		await tick();
		t.check('a block is being cued', player.isPlaying === true);

		FakeWebSocket.last.close();              // one real drop
		t.check('the player halted the cueing', player.isPlaying === false);
		t.check('the session failed the block', sess.state === SESSION.ERROR);
		t.check('exactly one interruption alarm across both channels',
			alarms.filter((a) => /RECORDING INTERRUPTED/.test(a)).length === 1);
	}

	t.section('a mid-await drop cannot be revived when the await resolves');
	{
		const c = clientWithDisconnect();
		const p = new SequencePlayer();
		p.onEMGClient = c;
		c.onDisconnect(() => p.haltForDisconnect());
		const wv = fakeWebGL();
		wv.currentModelIndex = 1;                 // forces a model load -> a real await
		p.setWebGLView(wv);
		const phases = [];
		p.onPhaseChange((phase) => phases.push(phase));
		// A real prep countdown, so the resume path below would arm a NEW phase timer if
		// the guard were missing — that is the defect this test is about.
		p.loadConfig({
			stimulation: { enabled: false },
			items: [{ model: 'a.glb', animation: 0, repetitions: 1 }],
			settings: { prepTime: 0.05 },
		});
		p.play();
		await sleep(200);   // PREP elapses -> _playCurrentItem -> suspended on the model load
		t.check('suspended awaiting the model load', wv.pendingLoad !== null && p.isPlaying === true);

		c.disconnect();
		t.check('halted while suspended', p.isPlaying === false && p.currentPhase === PHASE.IDLE);
		const phasesAtHalt = phases.length;
		const animationsAtHalt = wv.playAnimationCalls;

		wv.resolveLoad();                          // the await finally resolves
		await tick();
		t.check('no phase change on resume', phases.length === phasesAtHalt);
		t.check('the overlay stays hidden (still IDLE)', p.currentPhase === PHASE.IDLE);
		t.check('the animation was not restarted', wv.playAnimationCalls === animationsAtHalt);
		t.check('no countdown timer was armed', p._phaseTimer === null);
		t.check('still not playing', p.isPlaying === false);
	}

	return t.finish();
}

if (isEntryPoint(import.meta.url)) await runStandalone(run);
