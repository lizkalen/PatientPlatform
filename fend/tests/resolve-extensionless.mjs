/**
 * ESM resolve hook: retry a failed relative import with a `.js` extension.
 *
 * The application is bundled by Vite, which resolves extensionless relative
 * specifiers (`import { PHASE } from '../ui/SequencePlayer'`). Plain Node does not.
 * Rather than edit production source to suit the tests, the suites register this
 * hook (see harness.mjs) so they can import the REAL modules unmodified.
 *
 * Node built-ins only — no dependencies.
 */
export async function resolve(specifier, context, next) {
	try {
		return await next(specifier, context);
	} catch (err) {
		if (specifier.startsWith('.') && !specifier.endsWith('.js') && context.parentURL) {
			return { url: new URL(`${specifier}.js`, context.parentURL).href, shortCircuit: true };
		}
		throw err;
	}
}
