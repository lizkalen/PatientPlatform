/**
 * MovementSessionController — the sensor pass's redundant hard stop.
 * ==================================================================
 *
 * THIS LAYER IS THE OPERATOR'S STOP PATH FOR ELECTRICAL STIMULATION DELIVERED TO A
 * HUMAN SUBJECT. The sensor-triggered pass drives the stimulator DIRECTLY through the
 * client: its only intended stop is one `setTimeout` armed on a phase transition, so
 * the hard stop is the last line of defence if that transition never arrives.
 *
 * Because firing it halts the whole session (terminal, operator-reset only), a false
 * positive is expensive — so the deadline is derived per bout instead of using a fixed
 * cap that a legitimately long movement would trip. Movement segments of 12-14 s are
 * normal (plan E2), and segment speed multipliers can stretch a clip further.
 *
 * Verifies:
 *   - the deadline tracks the cued clip's real duration, playback speed and slowest
 *     segment multiplier, and always clears the bout's true end;
 *   - the case a fixed 15 s cap got wrong (a 12 s clip at 0.5x runs 25 s);
 *   - both fallbacks when the clip duration cannot be read;
 *   - a timed-stim stop that fails to send retains the belief, re-purposes the timer
 *     as a retry rather than tearing it down, and is flushed on the next connect.
 *
 * Drives the real controller against a fake EMG client and a minimal player/WebGL
 * stand-in (a plain object with the two fields the deadline reads).
 *
 *   node fend/tests/run-all.mjs          (or: npm test, from fend/)
 *   node fend/tests/sensor-hard-stop.test.mjs
 */
import { createSuite, fakeEmgClient, loadSrc, isEntryPoint, runStandalone } from './harness.mjs';

export default async function run() {
	const { default: MovementSessionController } = await loadSrc('emg/MovementSessionController.js');
	const t = createSuite('sensor-hard-stop');

	/** Real controller + fake client + the minimum player surface it touches. */
	const harness = (webglView) => {
		const client = fakeEmgClient();
		const player = {
			config: null,
			webglView,
			onSequenceComplete() {}, onPhaseChange() {}, onChange() {},
			getState: () => ({}),
			stop() {}, loadConfig() {}, play() {}, setExternalRecording() {},
		};
		const c = new MovementSessionController(client, player, {});
		t.defer(() => c._clearSensorTimers());
		return { client, player, c };
	};

	t.section('the hard-stop deadline is derived from the cued bout');
	{
		// 13 s clip at 1x, no segment slowdown: a legitimately long bout (plan E2).
		const { c } = harness({ animationDuration: 13, playbackSpeed: 1, segmentManager: { speeds: {} } });
		c.preStimDelayMs = 1000; c.postStimHoldMs = 2000;
		const d = c._hardStopDeadlineMs();
		const trueEnd = (13000 - 1000) + 2000;              // stim span + post-move hold
		t.check('13 s clip does not undercut the bout', d === trueEnd + 2000 && d > trueEnd);
	}
	{
		// The case the old fixed cap got WRONG: a 12 s clip at a 0.5x segment speed runs
		// 24 s, so the true train ends at 25 s while the old deadline (15000 + hold) was
		// 17 s — a terminal halt on a healthy bout.
		const { c } = harness({
			animationDuration: 12, playbackSpeed: 1,
			segmentManager: { speeds: { movement: 0.5 } },
		});
		c.preStimDelayMs = 1000; c.postStimHoldMs = 2000;
		const trueEnd = (24000 - 1000) + 2000;
		t.check('old fixed cap WOULD have false-positived', (15000 + 2000) < trueEnd);
		t.check('derived deadline clears the true bout end', c._hardStopDeadlineMs() > trueEnd);
	}
	{
		// A segment slowdown must extend, not shorten, the deadline.
		const { c } = harness({
			animationDuration: 10, playbackSpeed: 1,
			segmentManager: { speeds: { movement: 0.5, rest: 1.0 } },
		});
		c.preStimDelayMs = 1000; c.postStimHoldMs = 2000;
		t.check('slowest segment multiplier extends the deadline',
			c._hardStopDeadlineMs() === 20000 - 1000 + 4000);
	}
	{
		const { c } = harness({ animationDuration: 6, playbackSpeed: 2, segmentManager: { speeds: {} } });
		c.preStimDelayMs = 0; c.postStimHoldMs = 2000;
		t.check('playback speed shortens the deadline', c._hardStopDeadlineMs() === 3000 + 4000);
	}
	{
		const { c } = harness(undefined);                       // no webgl view at all
		c.preStimDelayMs = 1000; c.postStimHoldMs = 2000;
		t.check('falls back to maxTimedStimMs when unknown', c._hardStopDeadlineMs() === 15000 + 2000 + 2000);
	}
	{
		const { c } = harness({ animationDuration: 0, playbackSpeed: 1 });   // unloaded clip
		t.check('fallback on a zero-duration clip',
			c._hardStopDeadlineMs() === 15000 + c.postStimHoldMs + 2000);
	}

	t.section('a failed timed-stim stop keeps its backstop');
	{
		const { client, c } = harness({
			animationDuration: 5, playbackSpeed: 1, segmentManager: { speeds: {} },
		});
		c._sensorStimOn = true;
		c._armHardStop();
		t.check('hard stop armed', c._hardStopTimer !== null);
		client.sendOk = false;
		c._stopTimedStim();
		t.check('failed stop retains the belief', c._sensorStimOn === true && c._pendingStimStop === true);
		t.check('timer re-armed as a retry, not torn down', c._hardStopTimer !== null);
		client.sendOk = true;
		client.connect();
		t.check('reconnect flushes the pending stop',
			c._sensorStimOn === false && c._pendingStimStop === false);
		t.check('timer cleared once the stop landed', c._hardStopTimer === null);
		t.check('two stop attempts made', client.stops() === 2);
	}

	return t.finish();
}

if (isEntryPoint(import.meta.url)) await runStandalone(run);
