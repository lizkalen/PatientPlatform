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
	const t = createSuite('disconnect-recovery');

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
		t.check('the alarm names what happened to the data',
			/connection lost; the server saves the partial block/.test(alarms[0]));

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
		t.check('the error says the data was kept', /server saves whatever was captured/.test(c.error));
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

	return t.finish();
}

if (isEntryPoint(import.meta.url)) await runStandalone(run);
