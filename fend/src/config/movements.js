/**
 * Movement Registry
 * =================
 *
 * The single source of truth for the hand animations shipped in `public/`.
 *
 * Every `.glb` the app can play is listed here exactly once, with the name the
 * patient and operator see. Previously the same list lived in four places —
 * WebGLView.AVAILABLE_MODELS plus a hand-maintained name map in each of
 * TutorialOverlay and ProgressSidebar — and they had drifted: several map keys
 * referred to files that don't exist ('wristextension' vs the real 'wristext'),
 * so those movements fell through to the filename fallback and patients were
 * shown "Wristext" instead of "Wrist Extension".
 *
 * ORDER IS SIGNIFICANT. GUIView's model dropdowns bind the selected option's
 * INDEX into this array, so inserting or reordering entries changes what an
 * existing selection resolves to. Append new movements at the end.
 */

/** @typedef {{ name: string, file: string }} Movement */

/** @type {Movement[]} */
export const MOVEMENTS = [
	{ name: 'Wrist Flexion', file: 'wrist.glb' },
	{ name: 'Tripod Pinch', file: 'tripodpinch.glb' },
	{ name: 'Ring + Pinky Flexion', file: 'lastfingers.glb' },
	{ name: 'Tripod Pinch (Fast)', file: 'fasttripodpinch.glb' },
	{ name: 'Ring + Pinky Flexion (Fast)', file: 'fastlastfingers.glb' },
	{ name: 'Wrist Abduction', file: 'abduction.glb' },
	{ name: 'Wrist Adduction', file: 'adduction.glb' },
	{ name: 'Wrist Extension', file: 'wristext.glb' },
	{ name: 'Finger Extension', file: 'fingerext.glb' },
	{ name: 'Wrist Extension (Fast)', file: 'fastwristext.glb' },
	{ name: 'Finger Extension (Fast)', file: 'fastfingerext.glb' },
	{ name: 'Fist', file: 'fist.glb' },
	{ name: 'All motions', file: 'all.glb' },
];

/** file (lowercased, no extension) -> display name */
const NAME_BY_KEY = new Map(
	MOVEMENTS.map((m) => [m.file.replace(/\.glb$/i, '').toLowerCase(), m.name]),
);

/**
 * The display name for a movement.
 *
 * @param {string|{model?: string}} model - a filename ('wristext.glb' or
 *   'wristext'), or a sequence item carrying one.
 * @param {string} [fallback='Exercise'] - returned when there is no model at all.
 * @returns {string}
 */
export function movementName(model, fallback = 'Exercise') {
	const file = typeof model === 'string' ? model : model?.model;
	if (!file) return fallback;

	const key = file.replace(/\.glb$/i, '').toLowerCase();
	const known = NAME_BY_KEY.get(key);
	if (known) return known;

	// Unknown file (e.g. a model dropped into public/ but not registered above):
	// split camelCase and capitalise, so it at least reads as words.
	return file
		.replace(/\.glb$/i, '')
		.replace(/([A-Z])/g, ' $1')
		.replace(/^./, (c) => c.toUpperCase())
		.trim();
}
