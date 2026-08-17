/**
 * Stop retry, reconnect ordering, and watchdog lifecycle.
 * =======================================================
 *
 * THIS LAYER IS THE OPERATOR'S STOP PATH FOR ELECTRICAL STIMULATION DELIVERED TO A
 * HUMAN SUBJECT. Every case here was a real defect: a stop believed but never sent,
 * a second Stop press silently swallowed, a reconnect snapshot overwriting the result
 * of the reconnect's own retry, a presumed-active train with no timer left bounding it.
 *
 * Verifies:
 *   - SequencePlayer's auto-stim driver (the third stim driver) follows the same rule
 *     as the controllers: claim a train only on a confirmed send, retain the belief on
 *     a failed stop so a second Stop press still sends, alarm once, flush on connect;
 *   - leaving MOVE while disconnected still ATTEMPTS the stop;
 *   - EMGClient rehydrates the `connected` payload BEFORE running onConnect callbacks,
 *     so a reconnect-triggered retry is the last word (end-to-end against the REAL
 *     EMGClient, driven through a stubbed `ws`);
 *   - a failed stop does not tear down the watchdog protecting the presumed-live train;
 *     it is retried on a slow cadence until it lands;
 *   - an adopted/presumed-active train gets a watchdog too, and starves into a stop;
 *   - the watchdog cause survives into the final banner text.
 *
 * Drives the real SequencePlayer, MovementStimController and EMGClient against a fake
 * EMG client / fake socket (see harness.mjs).
 *
 *   node fend/tests/run-all.mjs          (or: npm test, from fend/)
 *   node fend/tests/stop-retry-reconnect.test.mjs
 */
import { createSuite, fakeEmgClient, loadSrc, isEntryPoint, runStandalone } from './harness.mjs';

export default async function run() {
	const { default: MovementStimController } = await loadSrc('emg/MovementStimController.js');
	const { default: SequencePlayer, PHASE } = await loadSrc('ui/SequencePlayer.js');
	const { default: EMGClient } = await loadSrc('emg/EMGClient.js');

	const t = createSuite('stop-retry-reconnect');
	const make = (client) => {
		const s = new MovementStimController(client);
		t.defer(() => s._clearWatchdog());
		return s;
	};
	const openSocket = () => ({ readyState: 1, send() {} });   // WebSocket.OPEN

	// ==== SequencePlayer's own driver ==========================================
	t.section('SequencePlayer auto-stim: a failed stop is not a success');
	{
		const c = fakeEmgClient();
		const p = new SequencePlayer();
		p.onEMGClient = c;
		const alarms = []; p.onAlarm((m) => alarms.push(m));
		c.onConnect(() => p.flushPendingStimStop());
		p.config = {
			stimulation: { enabled: true, channels: [{ id: 1 }] },
			items: [{ stimulate: true, model: 'm', animation: 0, repetitions: 1 }],
			settings: {},
		};

		p._handleStimulationTransition(PHASE.PREP, PHASE.MOVE);
		t.check('start claimed on a good send', p._stimActive === true);

		c.sendOk = false;
		p._handleStimulationTransition(PHASE.MOVE, PHASE.REST);
		t.check('failed stop RETAINS the belief', p._stimActive === true);
		t.check('failed stop latches pending', p._pendingStop === true);
		t.check('failed stop alarms once', alarms.length === 1 && /STOP NOT SENT/.test(alarms[0]));

		const before = c.stops();
		p.stop();                                    // operator presses Stop again
		t.check('second Stop press still SENDS', c.stops() === before + 1);
		t.check('still not believed stopped', p._stimActive === true);
		t.check('no duplicate alarm for the same episode', alarms.length === 1);

		c.sendOk = true;
		c.connect();
		t.check('reconnect flushes the pending stop', p._stimActive === false && p._pendingStop === false);
	}
	{
		const c = fakeEmgClient();
		const p = new SequencePlayer();
		p.onEMGClient = c;
		p.config = {
			stimulation: { enabled: true, channels: [{ id: 1 }] },
			items: [{ stimulate: true }],
			settings: {},
		};
		c.sendOk = false;
		p._handleStimulationTransition(PHASE.PREP, PHASE.MOVE);
		t.check('failed START does not claim a train', p._stimActive === false);
		c.sendOk = true;
		c.isConnected = false;
		p._stimActive = true;                        // train live, socket then dropped
		p._handleStimulationTransition(PHASE.MOVE, PHASE.REST);
		t.check('leave-MOVE while disconnected still attempts the stop', c.stops() === 1);
	}

	// ==== reconnect ordering, end to end against the real EMGClient ============
	t.section('EMGClient: rehydrate before onConnect');
	{
		const client = new EMGClient({ autoReconnect: false });
		const stim = make(client);
		client.ws = openSocket();                    // sends succeed
		stim.configure({ patterns: { a: { channels: [{ id: 1 }] } }, stimCfg: {} });
		stim.enable();
		stim._start();
		client.ws = null;                            // socket gone -> _send returns false
		stim.stopAll();
		t.check('stop failed -> belief retained + pending',
			stim.stimming === true && stim._pendingStop === true);
		// The server returns; its connect snapshot still says a train is active, because
		// it was taken before our retry landed.
		client.ws = openSocket();
		client._handleConnected({
			type: 'connected', n_channels: 2, sample_rate: 2048,
			recording: true, stimulation_active: true,
		});
		t.check('rehydration consumed recording', client.isRecording === true);
		t.check('reconnect retry is the LAST word (stimming false)', stim.stimming === false);
		t.check('pending stop cleared', stim._pendingStop === false);
	}

	// ==== watchdog lifecycle across a failed stop ==============================
	t.section('a failed stop must not tear down the watchdog');
	{
		const c = fakeEmgClient();
		const s = make(c);
		let now = 1000; s._now = () => now;
		s.configure({ patterns: { a: { channels: [{ id: 1 }] } }, stimCfg: {} });
		s.enable();
		c.decision({ pMove: 0.9 });
		c.sendOk = false;
		now += 5000;
		c.decision({ pMove: 0.0 });                  // rest -> stop, which fails
		t.check('watchdog STILL armed after a failed stop', s._wdTimer !== null);
		const stopsBefore = c.stops();
		now += 2100; s._checkWatchdog();
		t.check('failed stop is retried on cadence', c.stops() === stopsBefore + 1);
		c.sendOk = true;
		now += 2100; s._checkWatchdog();
		t.check('retry lands -> belief cleared', s.stimming === false && s._pendingStop === false);
		t.check('watchdog cleared once the stop landed', s._wdTimer === null);
	}

	t.section('an adopted train is watched like one we started');
	{
		const c = fakeEmgClient();
		const s = make(c);
		let now = 1000; s._now = () => now;
		c.stimStatus({ active: true, status: 'ok', rehydrated: true });
		t.check('rehydrated adoption arms the watchdog', s.stimming === true && s._wdTimer !== null);
		t.check('decision clock refreshed on adoption', s._lastDecisionAt === 1000);
		now += 2100; s._checkWatchdog();
		t.check('adopted train starves into a forced stop', s.stimming === false && c.stops() === 1);
	}
	{
		const c = fakeEmgClient();
		const s = make(c);
		let now = 1000; s._now = () => now;
		c.stimStatus({ active: true, status: 'error', reason: 'requested' });
		t.check('failed-stop adoption arms the watchdog', s.stimming === true && s._wdTimer !== null);
	}

	// ==== the alarm the operator actually reads ================================
	t.section('the watchdog cause survives into the banner text');
	{
		const c = fakeEmgClient();
		const s = make(c);
		const alarms = []; s.onAlarm((m) => alarms.push(m));
		let now = 1000; s._now = () => now;
		s.configure({ patterns: { a: { channels: [{ id: 1 }] } }, stimCfg: {} });
		s.enable();
		c.decision({ pMove: 0.9 });
		c.sendOk = false;                            // socket dies
		now += 2100; s._checkWatchdog();             // starvation fires, stop cannot send
		const last = alarms[alarms.length - 1];
		t.check('final alarm names the watchdog cause',
			/STIM WATCHDOG \(no movement decision/.test(last) && /STOP NOT SENT/.test(last));
		t.check('fault latch set despite the failed send', s.enabled === false);
	}

	return t.finish();
}

if (isEntryPoint(import.meta.url)) await runStandalone(run);
