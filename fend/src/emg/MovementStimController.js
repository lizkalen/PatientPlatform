/**
 * Movement Stim Controller
 * ========================
 *
 * Closed-loop stimulation for the online phase, run as a MOVE/REST GATE. Every
 * movement shares the same FES pattern, so we fire that pattern ONCE when the
 * recognised state crosses into "moving" and stop it when it returns to rest — we
 * do NOT stop/restart as the class flickers between movements (close/trp/ext),
 * which would needlessly interrupt an otherwise continuous train. Onset is owned
 * by the clean head, return-to-rest by the stim head (its stim-rest class); both
 * expose the same signal `p_move = 1 - P(rest)`, already EMA-smoothed upstream,
 * which is what we threshold. A global emergency stop halts everything instantly.
 *
 * The gate signal is head-agnostic (it works across pre-stim clean windows and
 * stim windows alike) and independent of the 4-class/binary DISPLAY toggle. The
 * shared pattern + backend (type/url/port) come from the session config's
 * `onlineStimulation` block. `onThreshold >= offThreshold` gives hysteresis around
 * the 0.5 boundary so a p_move that hovers there can't chatter the stimulator; an
 * onset latch (`latchMs`) additionally holds a freshly-started train on through a
 * momentary rest right after the clean->stim handoff, so it can't bounce off and jiggle.
 *
 * (For per-movement stim, this becomes a pattern-diff switch instead — start/stop
 * only when the actual channels change. Not needed while stim is shared.)
 *
 * SAFETY — three properties this class must hold (fend plan A2/A3):
 *   1. A stop is only believed once the socket accepted it. `_send` returning false
 *      means the train is still running: `stimming` stays true, the stop is latched
 *      and retried on the next connect, and the operator is alarmed.
 *   2. The server is the authority on whether stim is live. `stimulation_status`
 *      confirms stops and reports FAILED ones (`{active: true, status: 'error'}` —
 *      the server retries in the background); the connect payload rehydrates the
 *      standing state so a reconnected tab doesn't assume "off".
 *   3. A dead-man watchdog runs independently of the decision stream: no decision for
 *      `decisionTimeoutMs`, or any train older than `maxTrainMs`, forces a stop AND
 *      latches the controller off. A watchdog fire is a fault, so re-arming is an
 *      operator action (enable()), never a decision value.
 */
export default class MovementStimController {
	constructor(emgClient) {
		this.client = emgClient;
		this.enabled = false;
		this.classOrder = [];       // index -> label (only for the p_move fallback)
		this.patterns = {};         // label -> { channels: [...] } (all identical here)
		this.stimCfg = {};          // { stimulatorType, controllerUrl, port }
		this.stimming = false;      // is the shared train currently running?
		this._pendingStop = false;  // a stop that never reached the server; retried on connect
		this._awaitRest = false;    // after a stop, require a rest crossing before restarting
		// Move/rest gate thresholds on p_move (= 1 - P(rest), smoothed upstream).
		this.onThreshold = 0.5;     // at rest, p_move >= this -> START stim (movement)
		this.offThreshold = 0.4;    // stimming, p_move < this -> STOP stim (rest); hysteresis
		// Onset latch: once stim starts, hold it on for at least this long before an
		// automatic (rest) stop is allowed — so a momentary rest right after the clean->
		// stim handoff can't bounce it off and start jiggling. Safety stops (disable /
		// emergencyStop / model inactive) bypass the latch.
		this.latchMs = 600;
		this._stimStartedAt = 0;    // wall-clock (ms) when the current train started

		// -- dead-man watchdog (independent of the decision stream) --
		this.decisionTimeoutMs = 2000;   // no movement_decision for this long -> force stop
		this.maxTrainMs = 30000;         // absolute cap on any one train -> force stop
		this.stopRetryMs = 2000;         // cadence for retrying a stop that failed to send
		this._lastDecisionAt = 0;
		this._lastStopAttemptAt = 0;
		this._wdTimer = null;
		this._alarmListeners = [];       // operator-visible alarms (wired to the status banner)

		this.client.onMovementDecision((d) => this._onDecision(d));
		// If the model is unloaded / movement goes inactive, fail safe.
		this.client.onMovementStatus((s) => { if (!s.active) this.stopAll(); });
		// The server's own view of the stimulator: confirms stops, reports failed ones.
		this.client.onStimulationStatus((s) => this._onStimStatus(s));
		// A stop that never made it out is retried the moment there is a socket again.
		this.client.onConnect(() => { if (this._pendingStop) this.stopAll({ force: true }); });
	}

	/** Subscribe to operator-visible stim alarms. Called with a message string. */
	onAlarm(cb) { this._alarmListeners.push(cb); }

	_raiseAlarm(message) {
		console.warn('[MovementStim] ALARM:', message);
		for (const cb of this._alarmListeners) {
			try { cb(message); } catch (e) { console.error('MovementStim alarm listener error', e); }
		}
	}

	configure({ classOrder, patterns, stimCfg, onThreshold, offThreshold, latchMs }) {
		this.classOrder = classOrder || [];
		this.patterns = patterns || {};
		this.stimCfg = stimCfg || {};
		if (typeof onThreshold === 'number') this.onThreshold = onThreshold;
		if (typeof offThreshold === 'number') this.offThreshold = offThreshold;
		if (typeof latchMs === 'number') this.latchMs = latchMs;
	}

	/** Operator arming. This is the ONLY way back from a watchdog latch — see
	 * `_failsafeStop`; nothing in the decision stream can call it. */
	enable() { this.enabled = true; this._awaitRest = false; }
	disable() { this.enabled = false; this.stopAll(); }

	/** Operator emergency stop — halt all stim immediately and stop reacting.
	 * FORCED: a stale local `stimming === false` (e.g. reinitialised by a reconnect,
	 * or a train started by another driver) must never suppress the command. */
	emergencyStop() {
		this.enabled = false;
		console.warn('[MovementStim] EMERGENCY STOP');
		this.stopAll({ force: true });
	}

	/** Move/rest gate: START the shared pattern on rest->move, STOP on move->rest,
	 * and HOLD otherwise — so flicker between movements never interrupts the train. */
	_onDecision(d) {
		// Feed the watchdog before anything can return early: it measures the arrival of
		// decisions, not what we chose to do with them.
		this._lastDecisionAt = this._now();
		if (!this.enabled) return;
		const pMove = this._moveProb(d);
		if (!this.stimming && pMove >= this.onThreshold) {
			if (!this._awaitRest) this._start();
		} else if (this.stimming && pMove < this.offThreshold
			&& (this._now() - this._stimStartedAt) >= this.latchMs) {
			this.stopAll();   // rest confirmed AND past the onset latch
		}
		// else: hold — sustained movement, sustained rest, the hysteresis band, or still
		// within the post-onset latch (an early rest can't bounce the train off).

		// After ANY stop the gate must see p_move back below the off threshold before it
		// may fire again, so a stop can never be undone by the same elevated signal that
		// was already running. Evaluated last: a rest-confirmed stop above satisfies it
		// in the same decision, so a normal bout boundary costs nothing.
		if (pMove < this.offThreshold) this._awaitRest = false;
	}

	/** Wall clock in ms (overridable in tests). */
	_now() { return Date.now(); }

	/** Movement probability = 1 - P(rest). Prefer the engine's smoothed `p_move`; fall
	 * back to the probability vector (rest = the 'rest' class, else the last class). */
	_moveProb(d) {
		if (typeof d.pMove === 'number') return d.pMove;
		const order = d.classOrder || this.classOrder || [];
		const probs = d.probs || [];
		let ri = order.indexOf('rest');
		if (ri < 0) ri = probs.length - 1;
		return (ri >= 0 && probs[ri] != null) ? 1 - probs[ri] : 0;
	}

	/** The shared stim pattern (identical across movements) — the first configured. */
	_gatePattern() {
		return Object.values(this.patterns || {})[0] || null;
	}

	/** Fire the shared train (once). No-op if no pattern is configured. */
	_start() {
		const pat = this._gatePattern();
		if (!pat || !pat.channels) return;
		const sent = this.client.stimulateStart({
			channels: pat.channels,
			stimulatorType: this.stimCfg.stimulatorType,
			port: this.stimCfg.port,
			controllerUrl: this.stimCfg.controllerUrl,
		});
		if (!sent) return;   // never went out: nothing is running, so claim nothing
		this.stimming = true;
		this._stimStartedAt = this._now();
		this._lastDecisionAt = this._now();
		this._armWatchdog();
	}

	/**
	 * Stop the running train (idempotent; safe from any teardown path).
	 *
	 * `stimming` is cleared ONLY on a confirmed send. On a dead socket the train is
	 * still running server-side, so the flag stays true, the stop is latched for the
	 * next connect, and the operator gets an alarm — never a silent "stopped".
	 *
	 * @param {object} [opts]
	 * @param {boolean} [opts.force] - send regardless of local belief (emergency stop)
	 * @param {string} [opts.why] - watchdog cause, kept in the alarm text if the send fails
	 * @returns {boolean} whether the command left the socket
	 */
	stopAll({ force = false, why = null } = {}) {
		this._awaitRest = true;   // no restart until p_move is seen back below offThreshold
		if (!force && !this.stimming && !this._pendingStop) { this._clearWatchdog(); return true; }
		this._lastStopAttemptAt = this._now();
		const sent = this.client.stimulateStop({ controllerUrl: this.stimCfg.controllerUrl });
		if (sent) {
			// Sent, not yet confirmed: the server's `stimulation_status` is what actually
			// settles this (see _onStimStatus). Locally we stop driving immediately.
			this.stimming = false;
			this._pendingStop = false;
			this._clearWatchdog();
		} else {
			// The train is still presumed running, so the local backstop must NOT be torn
			// down here — it is the only thing left bounding it. Keep it armed; it retries
			// the stop on `stopRetryMs` until it lands or the socket returns.
			const first = !this._pendingStop;
			this._pendingStop = true;
			if (!this._wdTimer) this._armWatchdog();
			// One alarm per failure episode: the retries are silent, and the first
			// message (with the watchdog cause, if any) stays on the banner.
			if (first) {
				this._raiseAlarm((why ? `STIM WATCHDOG (${why}) — ` : '')
					+ 'STOP NOT SENT — no connection to the server; stimulation may still be running');
			}
		}
		return sent;
	}

	/**
	 * The server's view of the stimulator. Four cases:
	 *   { active: false, status: 'ok' }    -> confirmed off; clears belief + pending stop
	 *   { active: true,  status: 'error' } -> a stop FAILED, server retrying: alarm, and
	 *                                         the retry is now the server's, not ours
	 *   { active: true,  rehydrated }      -> a train was already running when we
	 *                                         (re)connected: adopt it (audit S5)
	 *   { active: true,  status: 'ok' }    -> ordinary start ack; belief unchanged, so a
	 *                                         train started by another driver is not
	 *                                         mistaken for ours
	 */
	_onStimStatus(s) {
		if (!s || typeof s.active !== 'boolean') return;
		if (!s.active) {
			this.stimming = false;
			this._pendingStop = false;
			this._clearWatchdog();
			return;
		}
		if (s.status === 'error') {
			this.stimming = true;
			this._pendingStop = false;
			this._adoptActiveTrain();
			this._raiseAlarm('STIMULATION STILL ACTIVE — the stop FAILED at the device; the server is retrying');
		} else if (s.rehydrated) {
			this.stimming = true;
			this._adoptActiveTrain();
		}
	}

	/**
	 * A train we now believe is running but did not start ourselves (a stop that failed
	 * at the device, or one already running when we connected). It gets the same local
	 * backstop as one we started — without this, a presumed-active train had no timer
	 * bounding it at all.
	 *
	 * The clocks start now because we do not know when the train really began. With no
	 * decisions flowing the 2 s starvation timeout will force a stop within ~2 s; that
	 * is the intended direction — a spurious stop is safe, a suppressed one is not.
	 */
	_adoptActiveTrain() {
		this._lastDecisionAt = this._now();
		this._stimStartedAt = this._now();
		this._armWatchdog();
	}

	// -- dead-man watchdog ---------------------------------------------------------
	/** Arm the watchdog for a train we started. Cleared by every stop path. */
	_armWatchdog() {
		this._clearWatchdog();
		this._wdTimer = setInterval(() => this._checkWatchdog(), 250);
	}

	_clearWatchdog() {
		if (this._wdTimer) { clearInterval(this._wdTimer); this._wdTimer = null; }
	}

	/** The two time-based failsafes, evaluated against the overridable `_now()` so the
	 * policy can be stepped in tests without real timers. */
	_checkWatchdog() {
		if (!this.stimming) { this._clearWatchdog(); return; }
		const now = this._now();
		if (this._pendingStop) {
			// A stop that failed to send. The alarm is already up and the fault latch (if
			// any) already set — just keep attempting on a slow cadence so the presumed-
			// active train is chased down without spamming the socket or the banner.
			if (now - this._lastStopAttemptAt >= this.stopRetryMs) this.stopAll({ force: true });
			return;
		}
		if (now - this._lastDecisionAt >= this.decisionTimeoutMs) {
			this._failsafeStop(`no movement decision for ${now - this._lastDecisionAt} ms`);
		} else if (now - this._stimStartedAt >= this.maxTrainMs) {
			this._failsafeStop(`train exceeded the ${this.maxTrainMs} ms cap`);
		}
	}

	/**
	 * A watchdog fire is a FAULT, not a normal stop. It latches the controller off
	 * (`enabled = false`, the same latch disable()/emergencyStop() use) so no decision
	 * value can re-arm the gate: re-arming takes an explicit operator action —
	 * enable(), i.e. a new online run after the session is reset. Same rule as the
	 * session's emergency stop (plan A1): a halt is cleared by an operator, never by a
	 * state transition.
	 */
	_failsafeStop(why) {
		this.enabled = false;
		this._raiseAlarm(`STIM WATCHDOG — ${why}. Stop forced and closed-loop stim LATCHED OFF; `
			+ 'it will not restart until the operator re-arms it.');
		// `why` rides along so that if the stop cannot be sent, the failure alarm that
		// replaces this one on the banner still names the cause (banner is last-write-wins).
		this.stopAll({ force: true, why });
	}
}
