import { defineConfig } from 'vite';
import { execSync } from 'node:child_process';
import { readFileSync } from 'node:fs';

// Build-time provenance, compiled in as literal strings and stamped into every
// recording by SequencePlayer._buildTrialMetadata so a file traces to its code.
//
// Read name/version from package.json directly. process.env.npm_package_* is
// only set when Vite runs via an npm script; a bare `vite build` would stamp
// undefined.
const pkg = JSON.parse(
	readFileSync(new URL('./package.json', import.meta.url), 'utf-8'),
);

// git commit of this build. `-dirty` marks an uncommitted working tree, so the
// SHA alone won't reproduce it. Falls back to a marker if git is unavailable
// rather than failing the build; `-unknown` means the commit is known but its
// cleanliness isn't.
function gitProvenance() {
	const git = (args) =>
		execSync(`git ${args}`, { stdio: ['ignore', 'pipe', 'ignore'] })
			.toString()
			.trim();
	try {
		const sha = git('rev-parse HEAD');
		if (!sha) return 'unknown';
		try {
			return git('status --porcelain') ? `${sha}-dirty` : sha;
		} catch {
			return `${sha}-unknown`;
		}
	} catch {
		return 'unknown';
	}
}

export default defineConfig({
	server: {
		port: '8080',
	},
	define: {
		APP_NAME: JSON.stringify(pkg.name),
		APP_VERSION: JSON.stringify(pkg.version),
		GIT_SHA: JSON.stringify(gitProvenance()),
	},
});
