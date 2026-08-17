# PatientGUI — `bend/` refactoring plan

**Goal:** make the repository structure understandable by a human reading the directory
listing. Deduplication is a side effect, not the objective. Every package should have one
obvious home, and the name on the folder should say what is inside it.

**Status:** Phase 0 complete (`0145aeb`). Phases A–G below are outstanding.

---

## The organizing rule

> **Shared libraries live in `bend/`. `processing/` depends on `bend/`, never the reverse.**

Why this direction:

- `bend/` is the deployed artifact — it is what `start.bat` runs in front of a patient. A
  deployed system must not reach into a research scratch folder.
- `processing/` consuming the exact code the server ran is *desirable*: it is what makes
  offline analysis reproduce online behaviour.
- `processing/` has no environment of its own (no `.yml`, no `requirements.txt`); it already
  runs in `bend`'s `patientgui` env. The folder split is organizational, not technical.
- The repo already works this way: `processing/scripts/movement_disc_1307/` imports
  `mvdecoder` out of `bend/src`.

---

## Target layout

```
bend/
├─ pyproject.toml         import root = bend/src
├─ README.md              currently 0 bytes — document the structure here
├─ src/
│   ├─ server/            the live WebSocket server (13 modules)
│   ├─ dsp/               shared signal processing (used by server AND legacy)
│   ├─ legacy_pipeline/   the superseded CLI pipeline, retained deliberately
│   ├─ muniverse/         vendored, github.com/pranavm19/muniverse
│   └─ mvdecoder/         movement decoding library
└─ tools/                 hand-run bench/hardware diagnostics

processing/
├─ src/analysis/          research helpers (incl. SVM.py)
├─ src/weber/, blanking_helpers.py
└─ scripts/, notebooks/, configs/     → import muniverse & mvdecoder from bend
```

**Import convention after this work:** `bend/src` is the only import root. Everything is a
top-level package — `server.*`, `dsp.*`, `legacy_pipeline.*`, `muniverse.*`, `mvdecoder.*`.
No `src.` prefix anywhere. No `sys.path` manipulation anywhere.

---

## How the current files map onto it

Derived from static reachability over the three `.bat` entrypoints versus the two legacy
entrypoints (`run_cli_pipeline.py`, `online_decomp.py`).

### `src/server/` — 13 modules, reachable from the shipped entrypoints

```
lsl_ripple/server_cli.py                lsl_ripple/websocket_server.py
lsl_ripple/quattrocento_server_cli.py   lsl_ripple/simulated_websocket_server.py
lsl_ripple/simulate_server_cli.py       lsl_ripple/decomposition_manager.py
lsl_ripple/device.py                    lsl_ripple/movement_classifier_manager.py
lsl_ripple/quattrocento_device.py       lsl_ripple/movement_engine.py
lsl_ripple/simulated_device.py          lsl_ripple/movement_training.py
lsl_ripple/stimulation_client.py
```

Plus `__version__.py`. `lsl_ripple/__init__.py` must be **rewritten** — it currently exports
`RippleStream`/`SimulatedStream`, which are part of the dead generation (see orphans below),
so the package's advertised public API is the code that no longer runs.

### `src/dsp/` — 2 modules, reachable from **both** server and legacy

```
online/loading.py        load_pretrained_model
online/processing.py     process_chunk, design_filters, apply_filters, calculate_firing_rates
```

These are the reason "is `online/` legacy?" has no clean answer today. Giving them their own
home is what makes the server/legacy boundary honest.

Also moves here: `firing_rate_sliding_window` (28 lines, pure numpy) lifted out of
`analysis/classification.py`. It is the *only* thing `bend` uses from `analysis`, and it is a
DSP primitive. Complementary to the existing `calculate_firing_rates` — that one is chunkwise
for the online path, this one is whole-signal sliding window.

### `src/legacy_pipeline/` — 8 modules, legacy entrypoints only

```
online/pipeline_manager.py   online/cue_game.py         online/dof_config.py
online/streaming.py          online/video_game.py       online/plotting.py
online/decomposition.py      online/video_game_serious.py
```

Retained deliberately. The folder name states its status so nobody has to ask.

### Orphans — reachable from no entrypoint

**Delete (~856 lines).** These are a *middle* generation, between the CLI pipeline and the
current server: an LSL-outlet streaming layer, all February, all superseded.

```
lsl_ripple/cli.py                RippleStream over pylsl
lsl_ripple/stream.py
lsl_ripple/simulated_stream.py
lsl_ripple/simulate_cli.py
online/emg_websocket_server.py   docstring: "Trellis (LSL) --> This Server --> Frontend"
online/utils.py                  0 bytes
```

This generation is what gave `lsl_ripple` its name. Every LSL module in the package is now an
orphan and the live path contains none — the name is a fossil, which is the case for renaming.

**Move to `bend/tools/`** — hand-run diagnostics. This category currently has no home, which
is why `bt_open_close.py` reads as dead code when it is not.

```
online/otb_realtime_plot.py   pyqtgraph viewer for the OTBioLab+ socket (July 2026)
src/exo/bt_open_close.py      Tenoexo open/close bring-up over Bluetooth
test_decomp_ws.py             (from bend/ root)
```

### Deletions justified by the layout, not by a diff

| path | reason |
|---|---|
| `bend/src/muniverse/` | stale mirror, frozen 2026-02-17; replaced by the moved copy |
| `processing/src/online/` | stale mirror; `bend`'s copy is live (2026-07-02) |
| `bend/src/analysis/` | identical blobs to processing's, which additionally has `SVM.py` |
| `bend/src/utils/` | dead: only reachable via `consensus.py`, which imports a nonexistent `src.formento` |

`bend/src/neuroparser/` (Ripple's vendored `pyns`, 2,318 lines, zero consumers) is left for a
separate decision — it is unused but it is a vendored library, not rot.

---

## Phases

### Phase 0 — canonicalize `muniverse` ✅ `0145aeb`

- `core.py:302` — `[DEBUG] Using centroids` print re-commented (runs per-chunk in the live path)
- `__init__.py:7` — `from src.muniverse import (...)` → `from . import (...)`, so the package
  is name-agnostic and works before and after relocation
- `min_firing_rate` needed no change: processing's copy already had `10`

### Phase A — packaging skeleton

Add `bend/pyproject.toml` declaring `bend/src` as the package root, then `pip install -e ./bend`
into the `patientgui` env. No files move and no imports change yet.

**Verify:** `python -c "import muniverse, mvdecoder; print(muniverse.__file__)"` from a
directory that is *not* inside the repo.

This is what makes `processing` importing from `bend` a declared dependency rather than a
`sys.path` trick, and it must exist before the moves so there is always a working import path.

### Phase B — relocate `muniverse` (single atomic commit)

> **Prerequisite — resolved during Phase A.** The name `muniverse` on PyPI belongs to an
> unrelated reinforcement-learning project (`github.com/unixpickle/muniverse`, Alex Nichol),
> and `muniverse==0.1.0` was pinned in **both** `patientgui.yml` and `environment.yml` — almost
> certainly a `pip freeze` artifact that matched the local package name against PyPI. It was
> installed in the env and shadowed the repo copy, because the editable install appends
> `bend/src` to `sys.path` while `site-packages` is searched earlier.
>
> Normalizing to `import muniverse` without removing it would have silently bound the
> decomposition code to a game-playing RL library. The `src.` prefix was, accidentally, the only
> thing keeping them apart.
>
> Done: package uninstalled, pin removed from both `.yml` files. **Never `pip install muniverse`.**
> Related landmine: `muniverse/utils/logging.py:207` contains a bare `import muniverse` inside
> `_get_package_root()`. It is currently inert — guarded by `if "site-packages" in
> str(current_path)` — but would return the wrong package's directory if muniverse were ever
> installed into site-packages properly.

Split into two commits, because the two sides have different prerequisites.

**B1 — `bend` import rewrite ✅ (can be done before the move, and was)**

`bend` only needs `algorithms.core`, `.cbss` and `.decomposition`, all of which exist in the
stale mirror, so its imports can be repointed while the old files are still in place.

- `bend/src/muniverse/__init__.py:7` → `from . import (...)` so `import muniverse` resolves
- `src.muniverse.*` → `muniverse.*` in `online/decomposition.py` (×3), `online/processing.py`,
  `online/streaming.py`, `analysis/control.py`

**B2 — the move, plus the `processing` rewrite (single atomic commit)**

`processing` additionally needs `algorithms.muap` (`estimate_stim_template`,
`remove_stim_artifact`) and `decompose_muap`, which are among the 14 modules **absent** from the
stale mirror. Repointing it before the move breaks `blanking_decompose.py` and
`blanking_discriminability.py` immediately, so its rewrite must travel with the move.

1. `git rm -r bend/src/muniverse` **and** `git mv processing/src/muniverse bend/src/muniverse`
   in the *same* commit.
2. Rewrite `src.muniverse.*` → `muniverse.*` in `processing`:
   `scripts/blanking_decompose.py`, `scripts/blanking_discriminability.py`,
   `scripts/physiomio_decompose.py`, `src/analysis/control.py`, `src/blanking_helpers.py`,
   and 3 notebooks (`sandbox.ipynb`, `weber.ipynb`, `blanking_tests/blanking_comp_muap.ipynb`)
3. Copy `processing/src/configs/muap.json` into `bend/src/configs/` — `decompose_muap` resolves
   its default config as `Path(__file__).parent.parent.parent / "configs"`, which becomes
   `bend/src/configs/` after the move, and that directory does not currently contain it.

> **Hazard — must not be split.** `websocket_server.py:60` and `decomposition_manager.py:25` do
> `sys.path.insert(0, bend/src)` at **position zero**, ahead of site-packages. If
> `bend/src/muniverse/` still exists while imports point at `muniverse`, Python silently
> resolves to the stale February copy. The server would start, decomposition would run, and
> verification would pass — against the wrong code.

### Phase C — delete the stale mirrors and dead code

`processing/src/online/`, `bend/src/analysis/` (after lifting `firing_rate_sliding_window`),
`bend/src/utils/`, and the six orphans listed above.

### Phase D — the renames and import normalization

Do these as **one pass per package**, not as separate move-then-fix steps: moves, renames and
import rewrites all edit the same lines.

- `src/lsl_ripple/` → `src/server/`
- `src/online/` → split into `src/dsp/` (2) and `src/legacy_pipeline/` (8)
- normalize every import to the top-level convention

**Files that must be updated or they will break:**

| location | current | note |
|---|---|---|
| `start.bat`, `start_sim.bat`, `start_quattrocento.bat` | `python src/lsl_ripple/*_server_cli.py` | all three hardcode the path |
| `websocket_server.py:43` | `from bend.src.lsl_ripple.device import RippleDevice` | hardcoded fallback inside a `try/except` |
| `websocket_server.py:62`, `simulated_websocket_server.py:46`, `decomposition_manager.py:27-28` | `from online.processing import ...` | → `dsp.processing` |
| 26 bare sibling imports across the server modules | `from decomposition_manager import ...` | only resolve because the `.bat` makes that directory `sys.path[0]` |

> **Optional, and deliberately deferred: a `devices/` subpackage** grouping `device.py`,
> `quattrocento_device.py`, `simulated_device.py` — three duck-typed siblings implementing the
> same implicit interface, and the natural place to hang a `Protocol` documenting it. It is a
> real legibility win but it **breaks the bare sibling imports** (`from device import
> RippleDevice`, `from simulated_device import SimulatedDevice`). Only do it in the same pass
> that converts those to explicit package imports, never before.

### Phase E — `bend/tools/`

Create it and move the three hand-run diagnostics into it. Add a one-paragraph README saying
what the folder is for: *scripts you run by hand to check hardware; not imported by anything.*
That sentence is what prevents the next reader deleting them as dead code.

### Phase F — documentation and the vendored-code marker

- `bend/README.md` is **0 bytes**. Document the layout and the one-way dependency rule there.
- Add `bend/src/muniverse/README.md`: upstream is `github.com/pranavm19/muniverse`, when it was
  vendored, and what was patched locally (the float64 SVD cast in `core.py`, the
  `min_firing_rate` value, and `decompose_muap`). Without a `third_party/` folder to signal it,
  this file is the only thing that stops the next person hand-editing a vendored library.
- Lift `decompose_muap` (148 lines) out of the vendored `muniverse/algorithms/decomposition.py`
  into first-party code that imports muniverse. It currently sits under a hand-written
  `#-------------MINE-------------` banner, which is a symptom of having nowhere else to put it.

### Phase G — verification

In this order:

1. **Offline first** — re-run a `processing/scripts/movement_disc_1307/` script and diff its
   output against `old_out/`. This validates the shared tree with no realtime variables.
2. **Then the server** — run all three `.bat` files. Confirm decomposition runs, the frontend
   connects, and stdout contains no `[DEBUG] Using centroids`.
3. **Exo path** — confirm the COM-port field still drives `trigger_open_close` on REST↔MOVE
   transitions, since `decomposition_manager.py` is touched in phases B and D.

---

## Notes on scope

**Line-ending caveat.** `.gitattributes` is `* text=auto`, so `bend` checks out LF and
`processing` CRLF. Working-tree diffs between the two trees show *every* line as changed while
git sees identical blobs. Compare with `diff --strip-trailing-cr`, or `git rev-parse HEAD:<path>`
to compare blobs directly. Two files that looked like 598- and 1,198-line divergences were
byte-identical.

**Actual divergence between the forked trees is ~190 lines**, not thousands: `muniverse` 155
across 3 files, `online` 37 in 1 file, `analysis` zero. These are stale mirrors, not diverged
branches.

**Static reachability is not sufficient on this repo.** It cannot see cross-folder consumers
arriving via `sys.path`, hand-run bench tools, or vendored libraries. Every "dead code" call
here should be checked against those three categories before acting on it.
