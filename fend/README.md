# PatientGUI — frontend (`fend/`)

Operator/patient interface for EMG recording and electrical-stimulation sessions.
It cues the subject through a movement sequence, drives the stimulator, and streams
EMG from the backend (`bend/`) over a WebSocket. The product of a session is the
recording it produces, so correctness and traceability of that data come first.

See [docs/FEND_REFACTORING_PLAN.md](../docs/FEND_REFACTORING_PLAN.md) for the current
work and its ordering.

## Requirements

- Node.js (with npm)
- The backend (`bend/`) running, for a live EMG/stim connection

## Setup

```
npm install
```

Optional per-machine config: copy `.env.example` to `.env.local` and fill in the
values (e.g. `VITE_DECOMP_MODEL_PATH`). `.env.local` is gitignored.

## Run

- `npm run dev` — Vite dev server on http://localhost:8080 (also `npm start`)
- `npm run build` — production build to `dist/`
- `npm run preview` — serve the built output

Builds stamp the app version and git commit SHA into recording metadata; a `-dirty`
SHA marks a build made with uncommitted changes.

## Keyboard shortcuts

- `G` / `P` — toggle the engineering (Tweakpane) panel
- `H` — reset the camera (home)

## Dependencies

- [Three.js](https://github.com/mrdoob/three.js/) — 3D engine
- [Tweakpane](https://cocopon.github.io/tweakpane) — engineering GUI panel
- [stats.js](https://github.com/mrdoob/stats.js) — performance monitor
- [AsyncPreloader](https://github.com/dmnsgn/async-preloader) — asset loader
- [SASS](https://sass-lang.com/) — CSS extension
- [Vite](https://vitejs.dev/) — build system
