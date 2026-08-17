/**
 * Segment Configuration for Animations
 *
 * All animations follow the same timing structure:
 * - 0-2 seconds: Rest phase
 * - 2-12 seconds: Movement phase
 * - 12-14 seconds: Rest phase
 */

const segmentConfig = {
	// Default segment template (14 seconds total)
	defaultSegments: [
		{ name: 'Rest', start: 0, end: 2 },
		{ name: 'Movement', start: 2, end: 12 },
		{ name: 'Rest', start: 12, end: 14 },
	],

	// Per-model segment overrides, keyed by model file then animation index. Anything
	// absent falls back to `defaultSegments` (see getSegments below), so only add an
	// entry when a model genuinely deviates from the 0-2 / 2-12 / 12-14 structure:
	//
	//   'wrist.glb': {
	//     0: [
	//       { name: 'Rest', start: 0, end: 3 },
	//       { name: 'Movement', start: 3, end: 11 },
	//       { name: 'Rest', start: 11, end: 14 },
	//     ],
	//   },
	models: {},

	/**
	 * Get segments for a specific model and animation
	 * @param {string} modelFile - The model filename (e.g., 'wrist.glb')
	 * @param {number} animationIndex - The animation index
	 * @returns {Array} Array of segment objects
	 */
	getSegments(modelFile, animationIndex = 0) {
		const modelConfig = this.models[modelFile];
		if (modelConfig && modelConfig[animationIndex]) {
			return modelConfig[animationIndex];
		}
		// Return default segments if no specific config exists
		return this.defaultSegments;
	},
};

export default segmentConfig;