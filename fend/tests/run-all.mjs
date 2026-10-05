/**
 * Frontend safety-test runner.
 * ============================
 *
 * Runs every suite under `fend/tests/` and exits nonzero if any assertion failed.
 *
 * These suites cover the operator's stop path for electrical stimulation delivered to
 * a human subject: the emergency stop, the dead-man watchdogs, and the rule that a
 * stop is only believed once the server has actually been told. They drive the REAL
 * controllers from `fend/src/` against a fake EMG client — no copies, no mocks of the
 * code under test — so they fail if that code regresses.
 *
 * Node built-ins only; nothing to install.
 *
 *   npm test                      (from fend/)
 *   node fend/tests/run-all.mjs   (from the repo root)
 *
 * Suite paths resolve against this file, so either working directory behaves the same.
 */
const SUITES = [
	'./stim-controller.test.mjs',
	'./stop-retry-reconnect.test.mjs',
	'./sensor-hard-stop.test.mjs',
	'./disconnect-recovery.test.mjs',
	'./status-banner.test.mjs',
];

const results = [];
for (const spec of SUITES) {
	const { default: run } = await import(new URL(spec, import.meta.url).href);
	const name = spec.replace('./', '').replace('.test.mjs', '');
	console.log(`\n=== ${name} ===`);
	try {
		results.push(await run());
	} catch (err) {
		console.error(`  ERROR  suite threw: ${err?.stack || err}`);
		results.push({ name, passed: 0, failed: 1, failures: [`suite threw: ${err?.message || err}`] });
	}
}

console.log('\n========================================');
let passed = 0;
let failed = 0;
for (const r of results) {
	passed += r.passed;
	failed += r.failed;
	const status = r.failed ? 'FAIL' : 'ok  ';
	console.log(`  ${status}  ${r.name.padEnd(24)} ${String(r.passed).padStart(3)} passed`
		+ (r.failed ? `, ${r.failed} FAILED` : ''));
	for (const f of r.failures) console.log(`          - ${f}`);
}
console.log('----------------------------------------');
console.log(`  ${failed ? 'FAILED' : 'ALL PASS'} — ${passed} assertions passed`
	+ (failed ? `, ${failed} failed` : '') + `, ${results.length} suites`);
console.log('========================================');

process.exitCode = failed ? 1 : 0;
