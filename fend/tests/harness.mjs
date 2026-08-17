/**
 * Test harness — node built-ins only, zero dependencies.
 *
 * Provides the three things every suite needs:
 *   - `loadSrc()`  imports the REAL modules from `fend/src/`, resolved relative to
 *                  this file rather than the cwd, so the suites behave identically
 *                  run from the repo root or from `fend/`.
 *   - `createSuite()` a `node:assert` wrapper that records a failure instead of
 *                  aborting, so one bad assertion still yields a full suite report.
 *   - `fakeEmgClient()` the shared test double: a fake WebSocket PEER (not a copy of
 *                  any production class) whose `stimulateStart/Stop` return the same
 *                  boolean `EMGClient._send` does, so "the socket was dead" is
 *                  expressible — that boolean is the entire subject of these tests.
 */
import { register } from 'node:module';
import assert from 'node:assert/strict';

// Must run before any suite dynamically imports from src/. See the hook's comment.
register('./resolve-extensionless.mjs', import.meta.url);

/** URL of a module under `fend/src`, resolved against THIS FILE, never the cwd. */
export function srcUrl(rel) {
	return new URL(`../src/${rel}`, import.meta.url).href;
}

/** Dynamically import a real application module (after the resolve hook is live). */
export function loadSrc(rel) {
	return import(srcUrl(rel));
}

export function createSuite(name) {
	return {
		name,
		passed: 0,
		failed: 0,
		failures: [],
		_cleanups: [],

		/** One assertion. Records and reports; never throws. */
		check(label, condition) {
			try {
				assert.ok(condition, label);
				this.passed++;
				console.log(`  PASS  ${label}`);
			} catch {
				this.failed++;
				this.failures.push(label);
				console.log(`  FAIL  ${label}`);
			}
		},

		/** Section heading, for readability in the output. */
		section(title) {
			console.log(`\n  -- ${title} --`);
		},

		/** Register teardown. Controllers arm real timers; a leaked one would hang the
		 * runner, which is exactly the defect class these tests exist to prevent. */
		defer(fn) {
			this._cleanups.push(fn);
		},

		finish() {
			for (const fn of this._cleanups.reverse()) {
				try { fn(); } catch (e) { console.error('  cleanup error', e); }
			}
			this._cleanups = [];
			return { name: this.name, passed: this.passed, failed: this.failed, failures: this.failures };
		},
	};
}

/**
 * A fake EMG server connection. `sendOk` models the socket: false is a dead socket,
 * i.e. `_send` returning false — the case where a stop never reaches the server.
 */
export function fakeEmgClient() {
	return {
		sendOk: true,
		isConnected: true,
		sent: [],
		_dec: [], _mvs: [], _stim: [], _conn: [],

		// -- subscription surface used by the controllers --
		onMovementDecision(cb) { this._dec.push(cb); },
		onMovementStatus(cb) { this._mvs.push(cb); },
		onStimulationStatus(cb) { this._stim.push(cb); },
		onConnect(cb) { this._conn.push(cb); },
		onMvTrainStatus() {},
		onConfigStatus() {},
		onError() {},

		// -- commands (return value == "did it leave the socket") --
		stimulateStart(msg) { this.sent.push(['start', msg]); return this.sendOk; },
		stimulateStop(msg) { this.sent.push(['stop', msg]); return this.sendOk; },
		mvRecordStop() {},
		mvOnlineStop() {},

		// -- server -> client events --
		decision(d) { for (const cb of this._dec) cb(d); },
		movementStatus(s) { for (const cb of this._mvs) cb(s); },
		stimStatus(s) { for (const cb of this._stim) cb(s); },
		connect() { for (const cb of this._conn) cb(); },

		// -- assertions helpers --
		stops() { return this.sent.filter((x) => x[0] === 'stop').length; },
		starts() { return this.sent.filter((x) => x[0] === 'start').length; },
	};
}

/** True when this module URL is the process entry point (standalone suite run). */
export function isEntryPoint(moduleUrl) {
	const arg = process.argv[1];
	if (!arg) return false;
	return moduleUrl === new URL(`file:///${arg.replace(/\\/g, '/').replace(/^\/+/, '')}`).href;
}

/** Standalone runner for a single suite file. */
export async function runStandalone(run) {
	const result = await run();
	console.log(`\n${result.failed ? 'FAILED' : 'OK'}  ${result.name}: `
		+ `${result.passed} passed, ${result.failed} failed`);
	process.exitCode = result.failed ? 1 : 0;
}
