/**
 * MovementStimController — the closed-loop stimulation gate.
 * ==========================================================
 *
 * THIS LAYER IS THE OPERATOR'S STOP PATH FOR ELECTRICAL STIMULATION DELIVERED TO A
 * HUMAN SUBJECT. A stop that is believed but never sent, or a train nothing is
 * timing, is a patient-safety defect — not a UI glitch. These tests exist so those
 * paths cannot regress silently.
 *
 * Verifies (fend plan A2/A3, audit S5):
 *   - a stop is believed ONLY when the socket accepted it; a failed stop keeps the
 *     belief, latches a retry, alarms, and is flushed on the next connect;
 *   - the server's `stimulation_status` is authoritative — a failed stop reported as
 *     `{active: true, status: 'error'}` is adopted, a plain start ack is not;
 *   - a train already running on (re)connect is adopted so the emergency stop is not
 *     suppressed by a stale `stimming === false`;
 *   - the emergency stop always sends, regardless of local belief;
 *   - the dead-man watchdog (2 s decision starvation) and the 30 s absolute cap both
 *     force a stop, LATCH the controller off, and cannot be re-armed by a decision;
 *   - after any stop the gate requires a genuine rest crossing before firing again;
 *   - no watchdog timer is left running after a normal stop.
 *
 * Drives the real controller against a fake EMG client (see harness.mjs); time is
 * controlled through the controller's own overridable `_now()`.
 *
 *   node fend/tests/run-all.mjs          (or: npm test, from fend/)
 *   node fend/tests/stim-controller.test.mjs
 */
import { createSuite, fakeEmgClient, loadSrc, isEntryPoint, runStandalone } from './harness.mjs';

export default async function run() {
	const { default: MovementStimController } = await loadSrc('emg/MovementStimController.js');
	const t = createSuite('stim-controller');
	/** Every controller's watchdog is torn down at the end — a leak would hang the runner. */
	const make = (client) => {
		const s = new MovementStimController(client);
		t.defer(() => s._clearWatchdog());
		return s;
	};
	const pattern = { patterns: { a: { channels: [{ id: 1 }] } }, stimCfg: {} };

	// -- A2: a stop on a dead socket is not a success ----------------------------
	t.section('A2: a failed stop is never recorded as success');
	{
		const c = fakeEmgClient();
		const s = make(c);
		const alarms = []; s.onAlarm((m) => alarms.push(m));
		s.configure(pattern);
		s.enable();
		c.decision({ pMove: 0.9 });
		t.check('start fired', s.stimming === true && c.sent[0][0] === 'start');
		c.sendOk = false;
		s._now = () => Date.now() + 10000;      // past the onset latch
		c.decision({ pMove: 0.0 });
		t.check('failed stop keeps stimming', s.stimming === true);
		t.check('failed stop latches pending', s._pendingStop === true);
		t.check('failed stop alarms', alarms.length === 1 && /STOP NOT SENT/.test(alarms[0]));
		c.sendOk = true;
		c.connect();
		t.check('reconnect flushes pending stop', s._pendingStop === false && s.stimming === false);
		t.check('two stops attempted total', c.sent.filter((x) => x[0] === 'stop').length === 2);
	}

	// -- A2: the server reports the stop FAILED at the device --------------------
	t.section('A2: server-reported failed stop');
	{
		const c = fakeEmgClient();
		const s = make(c);
		const alarms = []; s.onAlarm((m) => alarms.push(m));
		s.configure(pattern);
		s.enable();
		c.decision({ pMove: 0.9 });
		s._now = () => Date.now() + 10000;
		c.decision({ pMove: 0.0 });
		t.check('local stop believed sent', s.stimming === false);
		c.stimStatus({ active: true, status: 'error', reason: 'requested' });
		t.check('server says still active -> adopt', s.stimming === true);
		t.check('server-failed stop alarms', alarms.some((a) => /STILL ACTIVE/.test(a)));
		c.stimStatus({ active: false, status: 'ok', reason: 'requested' });
		t.check('confirmed off clears belief', s.stimming === false && s._pendingStop === false);
	}

	// -- S5: reconnect rehydration ------------------------------------------------
	t.section('S5: adopt a train that was already running');
	{
		const c = fakeEmgClient();
		const s = make(c);
		c.stimStatus({ active: true, status: 'ok', rehydrated: true });
		t.check('adopts a live train on reconnect', s.stimming === true);
		s.emergencyStop();
		t.check('emergency stop after reconnect SENDS', c.sent.some((x) => x[0] === 'stop'));
	}
	{
		const c = fakeEmgClient();
		const s = make(c);
		s.emergencyStop();   // never stimmed; force must still send
		t.check('emergency stop is unconditional', c.sent.length === 1 && c.sent[0][0] === 'stop');
		t.check('plain start ack is NOT adopted as ours', (() => {
			const c2 = fakeEmgClient(); const s2 = make(c2);
			c2.stimStatus({ active: true, status: 'ok', reason: 'requested' });
			return s2.stimming === false;
		})());
	}

	// -- A3: dead-man watchdog + absolute cap -------------------------------------
	t.section('A3: dead-man watchdog');
	{
		const c = fakeEmgClient();
		const s = make(c);
		const alarms = []; s.onAlarm((m) => alarms.push(m));
		let now = 1000; s._now = () => now;
		s.configure(pattern);
		s.enable();
		c.decision({ pMove: 0.9 });
		now += 1500; s._checkWatchdog();
		t.check('watchdog silent before 2000 ms', s.stimming === true);
		now += 600; s._checkWatchdog();
		t.check('watchdog fires at 2000 ms of silence', s.stimming === false);
		t.check('watchdog alarms', alarms.some((a) => /WATCHDOG/.test(a)));
		t.check('watchdog timer cleared', s._wdTimer === null);
	}
	{
		const c = fakeEmgClient();
		const s = make(c);
		const alarms = []; s.onAlarm((m) => alarms.push(m));
		let now = 1000; s._now = () => now;
		s.configure(pattern);
		s.enable();
		c.decision({ pMove: 0.9 });
		for (let i = 0; i < 40; i++) { now += 1000; c.decision({ pMove: 0.9 }); s._checkWatchdog(); }
		t.check('absolute 30 s cap fires despite live decisions', s.stimming === false
			&& alarms.some((a) => /cap/.test(a)));
		t.check('cap latches the gate off', s.enabled === false);
		c.decision({ pMove: 0.99 });
		t.check('no auto re-arm from a decision', s.stimming === false);
		s.enable();
		c.decision({ pMove: 0.99 });
		t.check('operator enable() re-arms', s.stimming === true);
	}

	t.section('A3: a watchdog fire is a fault, not a normal stop');
	{
		const c = fakeEmgClient();
		const s = make(c);
		let now = 1000; s._now = () => now;
		s.configure(pattern);
		s.enable();
		c.decision({ pMove: 0.9 });
		now += 2100; s._checkWatchdog();
		t.check('starvation latches the gate off', s.enabled === false && s.stimming === false);
		c.decision({ pMove: 0.99 });
		t.check('starvation: no auto re-arm', s.stimming === false);
	}

	// -- rest-before-restart hysteresis -------------------------------------------
	t.section('a stop is not undone by the signal that was already high');
	{
		const c = fakeEmgClient();
		const s = make(c);
		let now = 1000; s._now = () => now;
		s.configure(pattern);
		s.enable();
		c.decision({ pMove: 0.9 });
		now += 5000;
		c.decision({ pMove: 0.1 });                       // rest -> normal stop
		t.check('normal rest stop', s.stimming === false);
		c.decision({ pMove: 0.9 });                       // next bout starts normally
		t.check('normal bout boundary costs no bout', s.stimming === true);
		now += 5000;
		s.stopAll({ force: true });                       // an out-of-band stop
		c.decision({ pMove: 0.9 });
		t.check('out-of-band stop is not undone by a high decision', s.stimming === false);
		c.decision({ pMove: 0.1 });                       // genuine rest seen
		c.decision({ pMove: 0.9 });
		t.check('restarts after a genuine rest crossing', s.stimming === true);
	}

	// -- timer hygiene -------------------------------------------------------------
	t.section('no timer outlives a normal stop');
	{
		const c = fakeEmgClient();
		const s = make(c);
		s.configure(pattern);
		s.enable();
		c.decision({ pMove: 0.9 });
		const armed = s._wdTimer !== null;
		s.disable();
		t.check('watchdog armed on start, cleared on disable', armed && s._wdTimer === null);
	}

	return t.finish();
}

if (isEntryPoint(import.meta.url)) await runStandalone(run);
