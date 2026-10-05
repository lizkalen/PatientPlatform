/**
 * Sequence Config Schema
 * ======================
 *
 * Declarative description of every scalar field in a sequence config: where it
 * lives, what it's called, how it's edited, and what it defaults to.
 *
 * This exists so a field is declared ONCE. Previously each one was restated in
 * five places — the defaults in `sequenceConfig.js`, a hand-built widget in
 * SequenceConfigPanel, a second copy of the defaults in `_loadConfigToUI`, a
 * third in `_readConfigFromUI`, and the reader in SequencePlayer — so adding a
 * setting meant six edits, and missing the `_readConfigFromUI` one silently
 * dropped the field on every Apply.
 *
 * Adding a field is now: add an entry here.
 *
 * NOT described here, because they aren't flat scalars and each has its own
 * editor or no UI at all:
 *   - `items[]`                    — the sequence itself
 *   - `onlineStimulation`          — closed-loop patterns, no UI
 *   - `stimulation.nostimTrigger`  — set from the movement-session panel
 * Those survive an edit because the panel spread-merges rather than replaces.
 */

/** Read a dotted path ('metadata.subjectId') out of an object. */
export function getAtPath(obj, path) {
	return path.split('.').reduce((o, k) => (o == null ? undefined : o[k]), obj);
}

/** Write a dotted path into an object, creating intermediate objects as needed. */
export function setAtPath(obj, path, value) {
	const keys = path.split('.');
	const last = keys.pop();
	let target = obj;
	for (const k of keys) {
		if (target[k] == null || typeof target[k] !== 'object') target[k] = {};
		target = target[k];
	}
	target[last] = value;
}

/**
 * @typedef {object} Field
 * @property {string}  path        - dotted path into the config object
 * @property {string}  label       - row label in the editor
 * @property {'text'|'number'|'checkbox'|'select'|'textarea'} type
 * @property {*}       default     - used when the config has no value
 * @property {number}  [min]       - number only; advisory (spinner bound), NOT clamped
 * @property {number}  [max]       - number only; advisory
 * @property {number}  [step]      - number only
 * @property {boolean} [integer]   - number only; parse with parseInt
 * @property {string}  [unit]      - suffix shown after the input ('seconds', 'x')
 * @property {string}  [placeholder]
 * @property {Array<[string,string]>} [options] - select only: [value, label]
 * @property {boolean} [falsyToDefault] - see note below
 * @property {Function} [parse]    - custom raw-string -> value
 *
 * NOTE on `min`/`max`: these are passed to the <input> as HTML attributes only.
 * Values are NOT clamped on read — that matches the behaviour these fields have
 * always had, and clamping would silently rewrite values an operator had already
 * saved. Keep it that way unless you deliberately decide otherwise.
 *
 * NOTE on `falsyToDefault`: the timing settings have always been read as
 * `parseFloat(input.value) || <default>`, which means entering 0 yields the
 * DEFAULT, not 0 — you cannot currently set "0 seconds rest" from the panel.
 * That is preserved here so this refactor changes no behaviour, but it is
 * probably a bug worth fixing separately.
 */

/** @type {{title: string, fields: Field[]}[]} */
export const CONFIG_SECTIONS = [
	{
		title: 'Trial Metadata',
		fields: [
			{
				path: 'name', label: 'Sequence Name', type: 'text',
				placeholder: 'Sequence name', default: '',
			},
			{
				path: 'metadata.subjectId', label: 'Subject ID', type: 'text',
				placeholder: 'e.g., subject_001', default: '',
			},
			{
				path: 'metadata.sessionId', label: 'Session ID', type: 'text',
				placeholder: 'e.g., session_001', default: '',
			},
			{
				path: 'metadata.notes', label: 'Notes', type: 'textarea',
				placeholder: 'Additional notes...', default: '',
			},
		],
	},
	{
		title: 'Timing Settings',
		fields: [
			{
				path: 'settings.restBetweenReps', label: 'Rest Between Reps', type: 'number',
				min: 0, max: 30, step: 0.5, unit: 'seconds', default: 2.0, falsyToDefault: true,
			},
			{
				path: 'settings.prepTime', label: 'Prep Time', type: 'number',
				min: 0, max: 10, step: 0.5, unit: 'seconds', default: 1.0, falsyToDefault: true,
			},
			{
				path: 'settings.pauseBetweenItems', label: 'Pause Between Items', type: 'number',
				min: 0, max: 10, step: 0.5, unit: 'seconds', default: 0.5, falsyToDefault: true,
			},
			{
				path: 'settings.playbackSpeed', label: 'Playback Speed', type: 'number',
				min: 0.1, max: 3, step: 0.1, unit: 'x', default: 1.0, falsyToDefault: true,
			},
			{
				path: 'settings.loopSequence', label: 'Loop Sequence', type: 'checkbox',
				default: false,
			},
		],
	},
	{
		// The channel list is appended after these by the panel — it's a repeating
		// sub-editor, not a scalar field.
		title: 'Stimulation',
		fields: [
			{
				path: 'stimulation.enabled', label: 'Enable Stimulation', type: 'checkbox',
				default: false,
			},
			{
				path: 'stimulation.stimulatorType', label: 'Stimulator', type: 'select',
				default: 'science_mode3',
				options: [
					['science_mode3', 'Science Mode 3 (RehaMove3)'],
					['science_mode4', 'Science Mode 4 (P24)'],
					['mock', 'Mock (testing)'],
				],
			},
			{
				path: 'stimulation.controllerUrl', label: 'Controller URL', type: 'text',
				placeholder: 'http://127.0.0.1:11051', default: 'http://127.0.0.1:11051',
				parse: (raw) => raw.trim() || 'http://127.0.0.1:11051',
			},
			{
				path: 'stimulation.port', label: 'Device Port', type: 'text',
				placeholder: '(auto)', default: null,
				// Empty means "let the controller decide" — null, not ''.
				parse: (raw) => raw.trim() || null,
			},
		],
	},
];

/**
 * One stimulation channel. Drives BOTH channel editors — the global list under
 * `stimulation.channels` and the per-item override under `items[i].stim.channels`,
 * which used to be two independent renderers.
 *
 * `label` is used in the roomy global editor, `short` in the compact per-item rows.
 *
 * min/max are advisory only (see the note on Field). The per-item editor never
 * clamped, and the global editor's onchange didn't either — it only fell back to
 * the previous value on unparseable input. Both keep doing exactly that.
 *
 * @type {(Field & {key: string, short: string})[]}
 */
export const STIM_CHANNEL_FIELDS = [
	{ key: 'id', label: 'Ch ID', short: 'Ch', type: 'number', min: 1, max: 8, step: 1, integer: true, default: 1 },
	{ key: 'amplitude', label: 'Amp (mA)', short: 'mA', type: 'number', min: 0, max: 130, step: 1, default: 8.0 },
	{ key: 'pulse_width', label: 'PW (µs)', short: 'µs', type: 'number', min: 10, max: 1000, step: 10, integer: true, default: 200 },
	{ key: 'frequency', label: 'Freq (Hz)', short: 'Hz', type: 'number', min: 0.1, max: 2000, step: 1, default: 50.0 },
	{ key: 'is_biphasic', label: 'Biphasic', short: 'biphasic', type: 'checkbox', default: true },
];

/** A fresh stimulation channel carrying the schema defaults. */
export function defaultStimChannel(id = 1) {
	const ch = {};
	for (const f of STIM_CHANNEL_FIELDS) ch[f.key] = f.default;
	ch.id = id;
	return ch;
}

/**
 * Parse a raw input string for a field.
 *
 * @param {Field} field
 * @param {string} raw
 * @param {*} [previous] - current value, returned when `raw` won't parse as a number
 * @returns {*}
 */
export function parseFieldValue(field, raw, previous) {
	if (field.parse) return field.parse(raw);

	if (field.type === 'number') {
		const n = field.integer ? parseInt(raw, 10) : parseFloat(raw);
		if (Number.isNaN(n)) return previous !== undefined ? previous : field.default;
		// Preserved quirk — see the falsyToDefault note above.
		if (field.falsyToDefault && !n) return field.default;
		return n;
	}

	return raw;
}
