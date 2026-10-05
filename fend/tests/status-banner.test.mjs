/**
 * StatusBanner — the one indicator visible in both operator and patient mode.
 * ===========================================================================
 *
 * The banner is where every failure this module can detect becomes something an
 * operator can see without opening DevTools. Its ordering is a safety property: a
 * stimulation alarm must never be pushed off screen by a lesser message, because it
 * is the only on-screen evidence that current may still be flowing.
 *
 * Verifies:
 *   - `recording_warning` (a fault during a recording that is STILL RUNNING) travels
 *     the real EMGClient dispatch and reaches the banner, carrying the server's own
 *     text — it used to land in the unknown-type void and be dropped (audit B3);
 *   - it does NOT touch `isRecording`: the server keeps it distinct from
 *     `recording_status` so a warning is never read as a state transition;
 *   - a stim alarm raised afterwards takes the visible slot, and the warning returns
 *     when that alarm is dismissed;
 *   - unknown message types are still dropped without throwing.
 *
 * Drives the real EMGClient and the real StatusBanner; the DOM is a minimal stand-in
 * (the banner only ever creates elements, sets text/class, and appends once).
 *
 *   node fend/tests/run-all.mjs          (or: npm test, from fend/)
 *   node fend/tests/status-banner.test.mjs
 */
import { createSuite, loadSrc, isEntryPoint, runStandalone } from './harness.mjs';

export default async function run() {
	const t = createSuite('status-banner');

	// Minimal DOM: enough for createElement/append/appendChild and the attributes the
	// banner writes. Installed before StatusBanner is imported-and-constructed.
	const installDom = () => {
		const el = () => ({
			className: '', textContent: '', hidden: false, style: {}, children: [],
			setAttribute(k, v) { this[k] = v; },
			append(...kids) { this.children.push(...kids); },
			appendChild(kid) { this.children.push(kid); return kid; },
		});
		const realDoc = globalThis.document;
		globalThis.document = { body: el(), createElement: () => el() };
		t.defer(() => { globalThis.document = realDoc; });
	};
	installDom();

	const { default: StatusBanner } = await loadSrc('ui/StatusBanner.js');
	const { default: EMGClient } = await loadSrc('emg/EMGClient.js');

	const setup = () => {
		const client = new EMGClient({ autoReconnect: false });
		const banner = new StatusBanner(client);
		// Exactly the wiring App.js installs for this channel.
		client.onRecordingWarning((w) => banner.recordingWarning(w.message));
		// Pretend we are connected, so the DISCONNECTED item does not mask the warning.
		client.isConnected = true;
		banner._connected = true;
		banner._everConnected = true;
		banner._render();
		const deliver = (msg) => client._handleMessage(JSON.stringify(msg));
		return { client, banner, deliver };
	};
	const WARNING = 'spool write failed: 4096 samples dropped (gap at t=12.3s)';

	t.section('recording_warning survives the dispatch and reaches the banner');
	{
		const { banner, deliver } = setup();
		t.check('nothing on the banner to begin with', banner.el.className === 'status-banner');

		deliver({ type: 'recording_warning', recording: true, message: WARNING });
		t.check('the warning is displayed', banner.el.className.includes('status-banner--warn'));
		t.check("it carries the server's own text", banner.textEl.textContent.includes(WARNING));
		t.check('it is labelled as a recording warning',
			/RECORDING WARNING/.test(banner.textEl.textContent));
		t.check('it is warning tier, not alarm tier',
			!banner.el.className.includes('status-banner--alarm'));
		t.check('the operator can dismiss it', banner.dismissEl.hidden === false);
	}

	t.section('a warning is not a state transition');
	{
		const { client, deliver } = setup();
		client.isRecording = true;
		deliver({ type: 'recording_warning', recording: true, message: WARNING });
		t.check('isRecording untouched by a warning', client.isRecording === true);

		client.isRecording = false;
		deliver({ type: 'recording_warning', recording: false, message: WARNING });
		t.check('a warning never turns recording back on', client.isRecording === false);
	}

	t.section('a stimulation alarm outranks it');
	{
		const { banner, deliver } = setup();
		deliver({ type: 'recording_warning', recording: true, message: WARNING });
		t.check('warning holds the slot while nothing else is wrong',
			/RECORDING WARNING/.test(banner.textEl.textContent));

		banner.raiseAlarm('STIMULATION STILL ACTIVE — the stop FAILED at the device');
		t.check('the stim alarm takes the visible slot',
			/STIMULATION STILL ACTIVE/.test(banner.textEl.textContent));
		t.check('and it renders as an alarm', banner.el.className.includes('status-banner--alarm'));
		t.check('the alarm cannot be dismissed while live', banner.dismissEl.hidden === true);

		// Server confirms stim is off -> the alarm becomes dismissible, then dismissed.
		deliver({ type: 'stimulation_status', active: false, status: 'ok', reason: 'requested' });
		t.check('cleared alarm still holds the slot until dismissed',
			/STIMULATION STILL ACTIVE/.test(banner.textEl.textContent));
		banner.dismissEl.onclick();
		t.check('the warning is still there underneath',
			/RECORDING WARNING/.test(banner.textEl.textContent));
		banner.dismissEl.onclick();
		t.check('dismissing the warning clears the banner', banner.el.className === 'status-banner');
	}

	t.section('a disconnect still outranks a recording warning');
	{
		const { banner, deliver } = setup();
		deliver({ type: 'recording_warning', recording: true, message: WARNING });
		banner._connected = false;
		banner._render();
		t.check('DISCONNECTED wins over the warning',
			/DISCONNECTED/.test(banner.textEl.textContent));
		banner._connected = true;
		banner._render();
		t.check('the warning returns once reconnected',
			/RECORDING WARNING/.test(banner.textEl.textContent));
	}

	t.section('unknown message types are still dropped quietly');
	{
		const { banner, deliver } = setup();
		let threw = false;
		try {
			deliver({ type: 'some_future_message', payload: 1 });
			deliver({ type: 'recording_warning', recording: true, message: WARNING });
			deliver({ type: 'another_unknown' });
		} catch { threw = true; }
		t.check('an unknown type does not throw', threw === false);
		t.check('and does not disturb what is on the banner',
			/RECORDING WARNING/.test(banner.textEl.textContent));
	}

	return t.finish();
}

if (isEntryPoint(import.meta.url)) await runStandalone(run);
