/**
 * Status Banner
 * =============
 *
 * The one indicator that is visible in BOTH config and patient mode. Patient mode —
 * the mode experiments actually run in — otherwise has no connection indicator at
 * all, and the channels that carry stimulation failures had no subscriber anywhere
 * in the frontend (audit S6/B2).
 *
 * Exactly three things reach this banner, in priority order:
 *   1. STIM ALARM — a stop that failed or never went out, the dead-man watchdog
 *      firing, or a train found already running on (re)connect. Red and STICKY: it
 *      stays until the server confirms stimulation is off AND the operator clicks it
 *      away. "No alarm on screen" is therefore a statement about the stimulator.
 *   2. Connection — connecting / disconnected while the socket is down.
 *   3. Device lost — the acquisition device dropped out (`device_status`).
 *
 * Alarms are raised here by `raiseAlarm()` (wired from the stim drivers in App.js)
 * and directly from `stimulation_status`.
 */
export default class StatusBanner {
	/** `stimulation_status.reason` values that mean the SERVER cut the train, not us. */
	static UNREQUESTED_STOPS = new Set(['watchdog', 'client_disconnect', 'device_lost', 'shutdown']);

	constructor(emgClient) {
		this.client = emgClient;
		this._alarm = null;          // { message, cleared } — cleared = safe to dismiss
		this._connected = !!emgClient?.isConnected;
		this._everConnected = this._connected;
		this._deviceLost = null;     // message while the acquisition device is gone

		this._build();

		this.client.onConnect(() => {
			this._connected = true;
			this._everConnected = true;
			this._render();
		});
		this.client.onDisconnect(() => { this._connected = false; this._render(); });
		this.client.onStimulationStatus((s) => this._onStimStatus(s));
		this.client.onDeviceStatus(({ connected, message }) => {
			this._deviceLost = connected ? null : (message || 'acquisition device disconnected');
			this._render();
		});

		this._render();
	}

	/** Raise a sticky stimulation alarm. Re-raising an already-cleared alarm re-arms it. */
	raiseAlarm(message) {
		this._alarm = { message: String(message || 'STIMULATION ALARM'), cleared: false };
		this._render();
	}

	/** Operator dismissal. Refused while the condition is still live. */
	dismissAlarm() {
		if (!this._alarm?.cleared) return;
		this._alarm = null;
		this._render();
	}

	/**
	 * `stimulation_status` — `active` is the SERVER's belief, not an echo of a command.
	 * A failed stop arrives as { active: true, status: 'error' } while the server
	 * retries; { active: false, status: 'ok' } is the only thing that clears an alarm.
	 */
	_onStimStatus(s) {
		if (!s || typeof s.active !== 'boolean') return;
		if (s.active) {
			if (s.status === 'error') {
				this.raiseAlarm('STIMULATION STILL ACTIVE — stop FAILED, server retrying. Check the patient.');
			} else if (s.rehydrated) {
				this.raiseAlarm('STIMULATION ACTIVE — a train was already running when this tab connected.');
			}
			return;
		}
		if (s.status === 'error') {
			// Nothing is running, but a command failed or was refused (e.g. out-of-range
			// parameters, reason "rejected"). Show it — immediately dismissible.
			this.raiseAlarm(`STIMULATION COMMAND FAILED — ${s.message || s.reason || 'refused by the server'}`);
			this._alarm.cleared = true;
		} else if (StatusBanner.UNREQUESTED_STOPS.has(s.reason)) {
			// The SERVER stopped the train on its own (its dead-man deadline, our socket
			// dropping, the device going away). Stim is off, but the operator must learn
			// that the run was cut — it is not a stop anyone at the keyboard asked for.
			this.raiseAlarm(`STIMULATION STOPPED BY THE SERVER (${s.reason})`
				+ (s.message ? ` — ${s.message}` : ''));
			this._alarm.cleared = true;
		} else if (this._alarm) {
			this._alarm.cleared = true;   // server confirms stim is off: dismissal unlocked
		}
		this._render();
	}

	// -- DOM ----------------------------------------------------------------------
	_build() {
		this.el = document.createElement('div');
		this.el.className = 'status-banner';
		this.el.setAttribute('role', 'status');
		this.el.setAttribute('aria-live', 'polite');

		this.dotEl = document.createElement('span');
		this.dotEl.className = 'status-banner__dot';

		this.textEl = document.createElement('span');
		this.textEl.className = 'status-banner__text';

		this.dismissEl = document.createElement('button');
		this.dismissEl.className = 'status-banner__dismiss';
		this.dismissEl.type = 'button';
		this.dismissEl.textContent = '✕';
		this.dismissEl.title = 'Dismiss (only once the condition has cleared)';
		this.dismissEl.onclick = () => this.dismissAlarm();

		this.el.append(this.dotEl, this.textEl, this.dismissEl);
		document.body.appendChild(this.el);
	}

	/** What to show right now, highest priority first. null = nothing to say. */
	_resolve() {
		if (this._alarm) {
			return {
				kind: 'alarm',
				text: this._alarm.cleared
					? `${this._alarm.message}  ·  condition cleared — click ✕ to dismiss`
					: this._alarm.message,
				dismissible: this._alarm.cleared,
			};
		}
		if (!this._connected) {
			return {
				kind: 'warn',
				text: this._everConnected
					? 'DISCONNECTED FROM THE SERVER — reconnecting…'
					: 'CONNECTING TO THE SERVER…',
				dismissible: false,
			};
		}
		if (this._deviceLost) {
			return { kind: 'warn', text: `DEVICE LOST — ${this._deviceLost}`, dismissible: false };
		}
		return null;
	}

	_render() {
		const s = this._resolve();
		this.el.className = s ? `status-banner status-banner--${s.kind}` : 'status-banner';
		if (!s) return;
		this.el.setAttribute('aria-live', s.kind === 'alarm' ? 'assertive' : 'polite');
		this.textEl.textContent = s.text;
		this.dismissEl.hidden = !s.dismissible;
	}
}
