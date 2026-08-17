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
 */
export default class MovementStimController {
	constructor(emgClient) {
		this.client = emgClient;
		this.enabled = false;
		this.classOrder = [];       // index -> label (only for the p_move fallback)
		this.patterns = {};         // label -> { channels: [...] } (all identical here)
		this.stimCfg = {};          // { stimulatorType, controllerUrl, port }
		this.stimming = false;      // is the shared train currently running?
		// Move/rest gate thresholds on p_move (= 1 - P(rest), smoothed upstream).
		this.onThreshold = 0.5;     // at rest, p_move >= this -> START stim (movement)
		this.offThreshold = 0.4;    // stimming, p_move < this -> STOP stim (rest); hysteresis
		// Onset latch: once stim starts, hold it on for at least this long before an
		// automatic (rest) stop is allowed — so a momentary rest right after the clean->
		// stim handoff can't bounce it off and start jiggling. Safety stops (disable /
		// emergencyStop / model inactive) bypass the latch.
		this.latchMs = 600;
		this._stimStartedAt = 0;    // wall-clock (ms) when the current train started

		this.client.onMovementDecision((d) => this._onDecision(d));
		// If the model is unloaded / movement goes inactive, fail safe.
		this.client.onMovementStatus((s) => { if (!s.active) this.stopAll(); });
	}

	configure({ classOrder, patterns, stimCfg, onThreshold, offThreshold, latchMs }) {
		this.classOrder = classOrder || [];
		this.patterns = patterns || {};
		this.stimCfg = stimCfg || {};
		if (typeof onThreshold === 'number') this.onThreshold = onThreshold;
		if (typeof offThreshold === 'number') this.offThreshold = offThreshold;
		if (typeof latchMs === 'number') this.latchMs = latchMs;
	}

	enable() { this.enabled = true; }
	disable() { this.enabled = false; this.stopAll(); }

	/** Operator emergency stop — halt all stim immediately and stop reacting. */
	emergencyStop() {
		this.enabled = false;
		this.stopAll();
		console.warn('[MovementStim] EMERGENCY STOP');
	}

	/** Move/rest gate: START the shared pattern on rest->move, STOP on move->rest,
	 * and HOLD otherwise — so flicker between movements never interrupts the train. */
	_onDecision(d) {
		if (!this.enabled) return;
		const pMove = this._moveProb(d);
		if (!this.stimming && pMove >= this.onThreshold) {
			this._start();
		} else if (this.stimming && pMove < this.offThreshold
			&& (this._now() - this._stimStartedAt) >= this.latchMs) {
			this.stopAll();   // rest confirmed AND past the onset latch
		}
		// else: hold — sustained movement, sustained rest, the hysteresis band, or still
		// within the post-onset latch (an early rest can't bounce the train off).
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
		this.client.stimulateStart({
			channels: pat.channels,
			stimulatorType: this.stimCfg.stimulatorType,
			port: this.stimCfg.port,
			controllerUrl: this.stimCfg.controllerUrl,
		});
		this.stimming = true;
		this._stimStartedAt = this._now();
	}

	/** Stop the running train (idempotent; safe from any teardown path). */
	stopAll() {
		if (this.stimming) {
			this.client.stimulateStop({ controllerUrl: this.stimCfg.controllerUrl });
		}
		this.stimming = false;
	}
}
