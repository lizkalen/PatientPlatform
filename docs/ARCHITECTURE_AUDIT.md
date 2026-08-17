# PatientGUI — Architecture audit (cross-process)

**Date:** 2026-08-17. **Anchor:** all file:line references are against commit `0ed49db`
("moved files form old folder"). Line numbers will drift; they are anchors for the first
reader, not a contract.

**Method:** three targeted read-only audits over `bend/src/server/`, `bend/src/dsp/`, the
three CLI entrypoints, the three `.bat` launchers, and all 26 files under `fend/src/`.
Duplication percentages were measured with `difflib.SequenceMatcher`, not eyeballed.

**Relationship to the other plans:** `REFACTORING_PLAN.md` covers the `bend/` layout
(mostly executed); `FEND_REFACTORING_PLAN.md` covers in-file frontend defects (Phase 0
executed, A–H outstanding). This document covers what neither does: the architecture
*between* the processes — who owns stimulation, what happens when a process dies, and
where the same logic exists twice. Findings are numbered (S/P/D/B + number) so work items
can reference them.

**Headline:** the stimulator is 100% browser-owned. The backend is a stateless HTTP relay
with zero policy, zero validation, and zero lifecycle binding to the commanding client.
Every stimulation train is started with infinite duration, and the only off-switch in the
system is a WebSocket message from a browser tab.

---

## 1. Stimulation ownership is inverted — safety-critical

### S1 — The browser decides everything; trains are infinite by design

```
fend timer/decision → client.stimulateStart({channels, amplitude, pw, freq})
  → ws {command:"stimulate_start"}
  → websocket_server.py:576-588   (no validation, no state, no interlock)
  → StimulationClient.start_train → PUT {controller_url}/api/control/stimulate
```

- `stimulation_client.py:82-83` forces `duration_sec: -1` on every channel. The docstring
  (`:14-17`) is explicit: an infinite train that runs until explicitly stopped.
- The complete server-side stim surface is `stimulation_client.py` (112 lines) plus four
  call sites: `websocket_server.py:577`/`:591`, `simulated_websocket_server.py:439`/`:453`.
  There is no backend closed loop — `movement_engine.py` only *emits* decisions
  (`movement_classifier_manager.py:161-165`); the browser decides and actuates.
- `stimulation_client.py:19-22` states the ownership outright: parameters "are defined in
  the PatientGUI frontend and passed straight through."

### S2 — No teardown path stops stimulation

The only thing that ever calls `StimulationClient.stop()` is the `stimulate_stop`
WebSocket command. Every failure path leaves the train running:

- **Client disconnect:** `handle_client`'s `finally` (`websocket_server.py:395-397`) does
  exactly one thing — `self.clients.discard(websocket)`. A closed/crashed tab, a dropped
  socket, or a laptop lid leaves the train running indefinitely.
- **Detection latency compounds it:** `serve()` (`:1343`) passes no overrides, so
  `websockets` defaults apply (`ping_interval=20, ping_timeout=20`) — a hard-crashed tab
  is not even *noticed* for up to 40 s, and nothing happens when it is.
- **Server shutdown:** `shutdown()` (`:1318-1326`) deletes the device and LSL outlet and
  nothing else. It is synchronous, so it structurally *cannot* `await self.stim.stop()`.
  Ctrl+C, window close, `taskkill`: the last instruction the stimulator received is
  "stimulate forever". The controller (`samolator`, `http://127.0.0.1:11051`,
  `stimulation_client.py:32`) is a separate process that outlives the backend.
- **No watchdog anywhere.** `grep watchdog|deadline|max_duration|atexit|signal.` over
  `bend/src/server/` → zero hits. `StimulationClient.active` (`:51`, set at `:92`) is
  never read by anything — dead state that looks like a safety interlock and is not one.
- **Frontend side:** no `beforeunload`/`visibilitychange` handler stops stim; the two
  disconnect handlers that exist (`GUIView.js:448`, `EMGChannelView.js:283`) only update a
  status badge. A crashed tab runs no JS at all.

### S3 — The closed loop starves silently

`MovementStimController._onDecision` (`fend/src/emg/MovementStimController.js:69-80`)
closes the gate only when a **new** decision arrives with `p_move < offThreshold`. If
decisions stop arriving, the train runs forever. The server has three ways to stop
emitting decisions without telling the stim controller:

- device lost: `_on_device_lost` (`websocket_server.py:1123-1134`) broadcasts
  `device_status` — which the stim controller does not subscribe to — and never sends
  `movement_status active:false`;
- a `movement.process` exception is caught and printed (`:1313-1314`), loop continues;
- any multi-second event-loop stall (see P7).

### S4 — Three frontend stim drivers, one global stimulator, no arbitration

All three call the same `EMGClient.stimulateStart/Stop` (`EMGClient.js:242-260`); the
server-side stop is a bare `DELETE /api/control/stop` (`stimulation_client.py:101`) that
stops *everything*. There is no train id and no ownership token.

| # | Driver | Trigger | Private state | Refs |
|---|---|---|---|---|
| A | `SequencePlayer` auto-stim | MOVE-phase enter/leave | `_stimActive` | `SequencePlayer.js:507-546` |
| B | `MovementSessionController` sensor pass | `setTimeout` pre/post delays | `_sensorStimOn` | `MovementSessionController.js:596-665` |
| C | `MovementStimController` closed loop | per `movement_decision` | `stimming` | `MovementStimController.js:69-121` |

Mutual exclusion is by *convention only* — the session mutes driver A by rewriting the
config (`_noAutoStimConfig`, `MovementSessionController.js:569-574`). The convention has a
hole: `_calibConfig` (`:867-884`) spreads `...config` **without** stripping
`stimulation.enabled`, so the pre-online re-calibration (`beginOnline()`, `:356-363`) runs
with driver A live. Additionally, any driver's stop kills any other driver's train while
both private flags desync (`MovementStimController.js:45` calls `stopAll()` on any
inactive `movement_status`, reaching through to session-driven timed trains). A and B also
compute the *same* enter/leave-MOVE predicate independently (`SequencePlayer.js:512-513`
vs `MovementSessionController.js:597-598`).

### S5 — Stim state is never mirrored; after a reconnect, emergency stop sends nothing

Stim state appears in no connect payload (`_connected_payload`, `websocket_server.py:362-380`)
and no status message. On reconnect, `MovementStimController.stimming` initialises `false`
(`:32`) and `stopAll()` (`:116-121`) is guarded by `if (this.stimming)` — **the emergency
stop issues no command at all** while the train from before the blip is still running.
This is the highest-impact state-drift in the system.

### S6 — Stimulation acks exist, carry hardware errors, and have zero subscribers

`onStimulationStatus` is implemented end-to-end (`EMGClient.js:526-528`, dispatch
`:609-611`; server emits at `websocket_server.py:584-588`, `:594-598`) and `grep` finds
**no subscriber in the entire frontend**. A `PUT` that failed at the hardware
(`stimulation_client.py:95-97` returns `{"status":"error", ...}`) is broadcast to the
browser and dropped. All three drivers set their flag on the local send, ignoring the
reply. Worse in reverse: `stop()` failures are swallowed (`:109-111`) and the server then
broadcasts `{"active": False, **result}` (`websocket_server.py:594-598`) — the literal
`"active": False` is written before `**result` splats in `"status": "error"`, so **a
failed stop is reported to the UI as "stimulation off"**, with no retry and no alarm.
`onDeviceStatus` (`EMGClient.js:534`) and `onStreamEnded` (`:542`) are equally
subscriber-less.

### S7 — No validation, no authentication

- The only amplitude limit in the system is an HTML `max` attribute:
  `sequenceSchema.js:165-169` declares `max: 130` mA, and its own comments (`:58-61`,
  `:158-160`) state values "are NOT clamped on read" / "min/max are advisory only".
  Backend: `start_train` (`stimulation_client.py:57-97`) checks `if not channels` and
  nothing else. Amplitude, pulse width, frequency go from the browser to the hardware
  untouched.
- `controller_url` is client-supplied per command (`websocket_server.py:581`) — a client
  can redirect stimulation commands to an arbitrary host.
- Unlimited unauthenticated clients (`self.clients`, `:179`; `serve()` `:1343` — no
  `origins`, no limit, no handshake). Any tab that can reach the port can
  `stimulate_start`, `start_recording`, `set_otb_config`, `load_model`. Loopback-only by
  default (`server_cli.py:17`), and `--host 0.0.0.0` is one flag away.
- **Two backends against one stimulator is reachable** (second instance on `--port 8766`,
  or a crashed-then-relaunched pair): the controller API has no session concept — either
  backend's bare DELETE cancels the other's train.

### S8 — Prescription

The frontend cannot be made safe by frontend fixes: a crashed tab runs no JavaScript.
Safety invariants must live in the last process that survives. The backend must become the
stimulation **authority**:

1. server-side clamps on amplitude/pulse-width/frequency/channels, from a server-owned
   limits file (see D6);
2. every train bound to the client/session that started it; stop-on-disconnect;
3. a dead-man deadline on every train, armed at start, independent of decisions;
4. an **async** teardown path (`run()`'s `finally` must await `stim.stop()`, flush
   recordings, close the online writer and the exo COM port);
5. stim + recording state in the connect payload so a reconnecting client rehydrates;
6. a single-controller policy (ownership token, or refuse concurrent commanders).

`FEND_REFACTORING_PLAN.md` Phase A remains necessary — as the operator-facing layer on
top of these guarantees, not as the safety mechanism.

---

## 2. Nothing is ever shut down — process & resource lifecycle

### P1 — The launchers have no stop path and orphan in both directions

All three bats `start` two detached `cmd /k` consoles (`start.bat:70,74`;
`start_sim.bat:70,74`; `start_quattrocento.bat:78,82`) and exit. No `stop.bat`, no PID
capture, no job object; the README never mentions shutdown. Closing the frontend window
leaves the backend holding the device, port 8765, any open recording, and possibly an
active train. Closing the backend window leaves the browser in an infinite fixed-3s
reconnect loop (`EMGClient.js:965-971`). Window-close terminates python via the console
path *without* reliably running `finally`/`atexit` — so even the minimal `shutdown()` is
skipped.

### P2 — Port collision produces a healthy-looking window and a stale connection

`serve(...)` at `websocket_server.py:1343` has no `OSError` handling and no pre-bind
check; `server_cli.py:87-90` catches only `KeyboardInterrupt`. Relaunch with a stale
backend still bound → the new one dies with a traceback **inside a `/k` window titled
"Backend Server" that stays open**, while the frontend silently connects to the *old*
instance (different config, possibly a half-open recording). The bat prints "Both servers
are starting" unconditionally (`start.bat:77`). Vite side: `port: '8080'` without
`strictPort` (`vite.config.js:38`) silently increments to 8081 while the README says open
8080. A stale Quattrocento backend also keeps the single-consumer OTBioLab TCP session;
the loser serves an empty stream instead of exiting (`websocket_server.py:1332-1337`).

### P3 — A clean-machine start is broken end-to-end

- The bats never run the editable install that `bend/pyproject.toml:11` documents as
  required; `patientgui.yml`'s pip block does not include `-e ./bend`. On a fresh machine
  setup "succeeds" and `python -m server.server_cli` dies with `No module named server` —
  into a `/k` prompt that looks like a healthy server window.
- `README.md:98-109`'s manual path (`cd bend; python src/server/server_cli.py`) is also
  broken for the same reason.
- `start_sim.bat:70` passes `%~dp0sim\datafile1_filtered.npz` — no `sim/` exists in the
  repo. Simulation mode fails out of the box (`simulated_websocket_server.py:904-906`).
- `start.bat:8`/`start_sim.bat:8` and `start_quattrocento.bat:12` hardcode two *different*
  conda paths (`miniconda3` vs `AppData\Local\miniconda3`, the latter with a doubled
  backslash) — on any given machine at least one launcher is dead.
- `%CONDA_PATH%` is quoted in the setup check (`start.bat:14`) but unquoted at launch
  (`:70`); a space in `%USERPROFILE%` passes the check and fails the launch. The npm setup
  branch leaks `cd fend` into the launch cwd (`:52`). Declining setup plows on into the
  two `start` lines anyway (`:31`, `:66`).

### P4 — In-process lifecycle: nothing is joined, several things leak

No `subprocess`, no signal handlers, no `atexit` anywhere in `bend/src/server/`.

| Resource | Started | Stop mechanism | Actual behavior |
|---|---|---|---|
| LSL time-sync thread | `websocket_server.py:322-324` | `_running=False`, polled every 5 s (`:237`) | daemon, never joined |
| Quattrocento TCP reader | `devices/quattrocento.py:200-201` | `stop()` called **only from `__del__`** (`:239-248`) | **leaks forever — see below** |
| mv train/clean/finalize tasks | `_spawn_mv_job` (`:721-743`) | none | never awaited on shutdown; killed silently |
| mv save task | `:498-499` | none | fire-and-forget, no reference kept |
| Exo serial COM port | `decomposition_manager.py:208` | `stop_classification()` (`:246-251`) | **never closed by `shutdown()`** |
| Online-run raw file | `_open_online_writer` (`:916`) | `_close_online_writer` (`:926`) | **never closed by `shutdown()`** |
| LSL outlet | `:344` | `del` in `shutdown()` (`:1324`) | leaked on hard kill |

**The Quattrocento device can never be garbage-collected.** The reader thread holds a
strong reference to the device (bound-method target), `stop()` is only called from
`__del__`, and `__del__` never fires while the thread runs — a circular ownership
deadlock. `_close_device()` (`websocket_server.py:1113-1121`) drops the reference; the old
device keeps its TCP connection, keeps parsing frames, and keeps appending to `_blocks` —
a `deque()` with **no maxlen** (`quattrocento.py:192`). The deliberate reconfiguration
path (`set_otb_config`) therefore leaves two concurrent TCP consumers and unbounded memory
growth. Related: `start_classification` called twice leaks the first serial handle and
silently disables exo control (`decomposition_manager.py:221-230`, `:215-217`); same leak
in `unload_model` (`:142-157`).

### P5 — Recordings are RAM-only and written once, at the end

`websocket_server.py:1226-1227` appends every chunk to a list; the file is created only in
`stop_recording` (`:1046`). At Quattrocento rates that is ~2 MB/s ≈ 7 GB/hour, and the
`np.hstack` at `:1030` doubles peak RSS at stop. Consequences:

- process death or window close mid-recording loses **100%** of the block;
- `pkl.dump` failure (disk full, unpicklable client metadata) escapes → kills the client
  socket (see D3), the reset at `:1107-1111` never runs, and the next `start_recording`
  wipes the chunks (`:971`) — silent total loss plus a truncated `.pkl`;
- filenames are second-resolution timestamps (`:1045`) with no subject/session token —
  two stops in one second silently overwrite (contrast `_save_mv_recording:862`, which
  does it right);
- the sim server drops any active recording wordlessly when playback ends (`:778-784`);
- `mv_record_start` unconditionally clears an in-progress capture (`:475-477`);
- a killed process leaves the online-run `.raw` without its `.pkl` sidecar — which is the
  only record of shape/dtype (`:926-952`), so the file is unreadable; a double
  `mv_online_start` produces the same orphan (`:905-910`).

Other unbounded buffers: `_mv_raw` (`:1189`), `recording.decomp.sources` (`:1257`).

### P6 — Event-loop stalls block everything, including stim commands

- `stop_recording` pickles the entire session **synchronously on the event loop**
  (`:1086-1087`). The authors knew: the comment at `:495-496` says exactly this froze
  streaming/stim, and the fix (`to_thread`) was applied only to `_save_mv_recording`.
- `broadcast` (`:347-360`) awaits `client.send()` per client inside the stream loop; with
  `websockets`' default 64 KiB write limit, **one slow browser tab backpressures the
  entire acquisition loop** — device data stops draining and queued `stimulate_stop`
  commands stop being processed.
- `shutdown()` runs `RippleDevice.__del__`'s `time.sleep(1.0)` (`devices/ripple.py:75`) on
  the loop thread — the async paths were careful to use `to_thread` for this (`:1128`);
  the shutdown path was not.

### P7 — Prescription

One supervisor entry point (or at minimum a stop script + job object + single-instance
guard), port-bind error handling that *fails loudly*, the missing `pip install -e ./bend`
step in setup, one correct conda path, incremental recording flush (append-to-disk per
chunk or periodic checkpoint), sidecar-first online writer, and the async teardown from
S8.4. `README.md` and the bats must agree on one invocation form (see D2).

---

## 3. Duplication by fork, not parameterization

### D1 — `simulated_websocket_server.py` is a stale ~70% fork of the live server

Measured: 722 identical lines; 65–79% of the 915-line sim server is verbatim copy of the
1346-line main server (largest blocks: the recording handlers, `handle_client`, the
decomposition accumulation, `_build_movement_timeline` — byte-identical and `@staticmethod`
in both).

**Seven commands exist only in the live server:** `mv_record_pause` (`websocket_server.py:482`),
`mv_record_resume` (`:485`), `mv_online_start` (`:501`), `mv_online_stop` (`:508`),
`mv_train_clean` (`:518`), `mv_set_stim_binary` (`:524`), `set_otb_config` (`:600`). The
frontend sends them unconditionally; the sim answers "Unknown command", which the UI
swallows (B6) — so the entire movement training/online flow *appears* to work against the
simulator and records nothing.

**Same-name commands that behave differently** (a bug fixed in one fork persists in the
other): `mv_record_stop` saves nothing in sim (`simulated:391-397` vs live `:488-499`);
`mv_train` runs inline on the loop and leaks the raw block (`simulated:539-566` vs
`_spawn_mv_job`); `start_classification` catches only `ValueError` in sim (`:411-429`), so
other exceptions kill the socket; sim `get_status` omits movement status (`:462-475`); the
sim stream loop is `while True` with no stop flag and no error handling (`:765`, `:787`) —
its `_running` is dead (initialised `:142`, set only under `enable_lsl` at `:259`, read
only by the time-sync loop).

**The repo already contains the fix pattern:** `quattrocento_server_cli.py:171-190`
injects a `device_factory`/`config_loader` into the *same* `RippleWebSocketServer` class.
The sim fork should collapse into a `SimulatedDevice` behind that factory — deleting
~800 duplicated lines and making the movement flows testable without hardware. (The class
is misnamed and prints "Connected to Ripple Trellis" while running a Quattrocento —
`websocket_server.py:279`, `:1125`, `:1141`.)

### D2 — The three CLIs

`server_cli.py` vs `simulate_server_cli.py`: 78 of 95 lines identical.
`quattrocento_server_cli.py` re-declares the same 11 options with reworded help. Two real
divergences hide in the noise: **filtering defaults on for Ripple/sim but off for
Quattrocento** (`server_cli.py:34-38` vs `quattrocento_server_cli.py:82-86`,`:177`) — the
same GUI records physically different signals depending on which `.bat` was clicked — and
all three docstrings advertise `uv run` console scripts that don't exist (no
`[project.scripts]` in `pyproject.toml`). Four documented invocation forms exist across
docstrings/README/bats; at most one works (P3).

### D3 — Shared-by-copy fragility in both forks

`handle_message` catches only `json.JSONDecodeError` (`websocket_server.py:623`,
`simulated:483`); `handle_client` catches only `ConnectionClosed` (`:393`/`:304`). Any
other exception — `int(ch)` on a non-numeric channel (`:532`), ragged `np.hstack`
(`:490`), `pkl.dump` failure (`:1087`) — kills that client's socket with a 1011.

### D4 — The frontend render pipeline exists three times

| Stage | `EMGChannelView` | `DecompositionView` | `TriggerMonitor` |
|---|---|---|---|
| Ring buffer | reads client's (`EMGClient.js:741-765`) | own copy `:308-342` | own copy `:71-85` |
| Min/max autoscale | `:437-443` | `:543-548` | `:93-96` |
| Decimation | nearest-neighbour `:445-458` | nearest-neighbour `:556-567` | **min/max envelope (correct)** `:105-115` |
| RAF start/stop | `:360-374` | `:406-420` — char-for-char identical | none — draws synchronously in the socket callback (`:84`) |

`DecompositionView.js:315` acknowledges the copy in a comment. The duplication is *why*
the correct decimation landed only in the throwaway debug widget — there was no shared
path to fix. Also: `EMGClient.onData` fan-out (`:49`, `:474`, `:768-774`) has zero
subscribers — buffers are maintained and an empty callback list iterated ~20×/s forever.
`SequencePlayer` itself contains two structurally identical countdown-timer loops
(`:571-586`, `:828-851`), and `onPhaseChange` fires on every 100 ms tick, driving a full
session-view re-render at 10 Hz (`MovementSessionController.js:684-695`).

### D5 — The protocol is two hand-maintained string tables

A 23-branch `if/elif` chain (`websocket_server.py:405-621`) vs 23 hand-written methods
(`EMGClient.js:164-434`) vs a 17-case `switch` (`:569-644`). No shared schema, no
generated types. Unknown commands come back as an error the UI drops unless it matches
`/config/i` (`MovementSessionController.js:700-702`) — the mechanism that made D1's drift
silent.

### D6 — Constants duplicated across the boundary (the dangerous ones)

| Value | Backend | Frontend |
|---|---|---|
| `trig_ch = 192` | `movement_engine.py:30,69,297`; `movement_training.py:535`; `websocket_server.py:763,861` | `EMGClient.js:373`; `GUIView.js:417`; `MovementSessionController.js:78` |
| sample rate 2048 | `movement_engine.py:29`; `quattrocento.py:39`; `quattrocento_server_cli.py:153` | `EMGClient.js:44`; `MovementSessionController.js:204` |
| class labels | `movement_engine.py:32-33` | `EMGClient.js:34` (hardcoded fallback) |
| classification defaults | `decomposition_manager.py:34-37,163-169`; `websocket_server.py:543-546` | `EMGClient.js:419-422`; `DecompositionController.js:31-34` |
| controller URL / stimulator type / port 8765 | `stimulation_client.py:32-33`; `*_cli.py` | `sequenceConfig.js:42-69`; `sequenceSchema.js:129-139`; `App.js:115` |

`trig_ch` drives stim-artifact blanking — six independent copies of a safety-relevant
number. **And one real cross-boundary numeric bug:** the blanking window is sized for
30 Hz stim (`movement_engine.py:31` `STIM_HZ = 30` — itself dead code; `pre=8, post=23`
samples at `:69,80,204-211`), while the frontend default stim frequency is **50 Hz**
(`sequenceConfig.js:50`, `sequenceSchema.js:168`). At 50 Hz the blank covers ~75% of every
pulse period and the clean/stim two-stage routing (`movement_engine.py:225`, `:276`)
collapses — essentially every window classifies as "stim-contaminated". Nothing on either
side validates configured frequency against what the model was blanked/trained for.

Fix direction: one machine-readable `protocol/limits` file (message types, trig channel,
sample rates, stim parameter ceilings, class labels) read by both sides; the backend
clamps against it (S8.1).

### D7 — The config shape exists in ~8 places

(1) `sequenceConfig.js:8-92` defaults; (2) `sequenceSchema.js:71-170` — second set of
defaults, currently in agreement, nothing checks it; (3) `configStore.normalizeConfig`
(`configStore.js:33-37`) — **invents an 8 mA stim channel** for any config lacking one,
which describes all four files in `fend/configs/`; (4) `localStorage`, loaded unvalidated
at boot (`App.js:102`); (5) `fend/configs/*.json` + `configs/old/*.json` — eight files in
a pre-stimulation shape that **no code path references** (not imported, not fetched, not
under `public/`); (6–8) the session controller's derived configs — `_nostimConfig`
(`:541-558`), `_noAutoStimConfig`/`_calibConfig` (`:569-574`, `:867-884`),
`_guidedOnlineConfig` (`:396-427`) — plus a fourth literal stim-channel copy at `:189`.
`sequenceSchema.js:17-22` names the fields it deliberately skips — `items[]`,
`onlineStimulation`, `nostimTrigger` — exactly the structures that carry stimulation
parameters and movement identity.

---

## 4. Split-brain state across the process boundary

### B1 — Reconnect rehydrates neither recording nor stim state

`_connected_payload` (`websocket_server.py:362-380`) omits `recording` and all stim state.
`get_status` includes `recording` (`:608`) but the client discards it
(`EMGClient.js:799-809` reads only the decomposition/classification fields). After a
drop + auto-reconnect, `EMGClient.isRecording` is stale; `SequencePlayer.stop()` guards
`stopRecording` on it (`:232-236`), so a recording can be left open and never saved. The
stim half of this is S5.

### B2 — Server-side latches a disconnected client can never clear

`_mv_paused` is set by `mv_record_pause` (`:482-483`) and cleared only by
`mv_record_resume`/`mv_record_start`; **`mv_record_stop` does not clear it** (`:488-499`),
and the frontend deliberately skips resume-before-stop
(`MovementSessionController.js:474-481`). It gates the *general* recorder (`:1226`). Abort
a session at the wrong moment and every subsequent `start_recording` produces a file
containing only the pre-trigger buffer — with a cheerful "Recording started" broadcast and
no error, indefinitely. The invariant claimed in the comment at `:1224-1225` is enforced
nowhere.

### B3 — The session controller hijacks the player and never gives it back

`MovementSessionController` is not a second engine — it drives the single `SequencePlayer`
(`_runSequence`, `:504-513`) by installing **derived** configs and never restoring the
operator's (`reset()` at `:462-492` clears its own fields only). After any movement
session, Play replays the derived variant (1-rep calibration, zeroed amplitudes) and
`getPlan()` (`:142-155`) reads back the config it itself installed (`_config()`, `:138`).

The sharpest chain: during guided training the patient sidebar stays live
(`beginTraining()` → `setMode('patient')`, `:245`) with a Stop wired directly to
`player.stop()` (`ProgressSidebar.js:127`), bypassing the controller. `stop()` does not
fire `onSequenceComplete`, so `setExternalRecording(false)` (`:675`) never runs — the flag
**latches**. The server-side `mv_record_start` capture (`:246`) stays open; press Start
again and the re-cued pass appends to it, then `_onTrainRecorded()` (`:515-526`) sends a
`classSequence` describing one pass over a recording containing one-and-a-fraction —
**bout labels shift positionally and the classifier trains on mislabelled data with no
error anywhere.** Abandon instead, and every later ordinary run silently records nothing
(`SequencePlayer.js:183`, `:232`) while the UI looks normal. (Supersedes/extends fend plan
C6, which covers only the config-panel route.)

### B4 — Two classification pipelines actuating from opposite sides of the boundary

| | MU-decomposition | Movement classifier |
|---|---|---|
| Backend | `decomposition_manager.py` (filtered samples, `websocket_server.py:1245`) | `movement_engine.py` (raw samples, `:1285`, re-runs its own filters `:177,212`) |
| Labels | ints 0/1/2 | strings close/trp/ext/rest |
| Actuator | **exo hand, from the backend over serial** (`decomposition_manager.py:381-391`) | **FES stimulator, from the frontend over HTTP** |

They share no state, have no mutual exclusion (`load_model` `:413` vs
`load_movement_model` `:444`; independent flags `:1242`/`:1282`), and can run
simultaneously — at which point two writers drive the same 3D hand:
`DecompositionController` (`:209-240`, `:246-328`, plus `App.js:184` runs its `update()`
unconditionally every frame) versus `SequencePlayer._playCurrentItem` (`:603-661`). The
frontend also re-derives the move/rest decision the backend already computed, with
different thresholds (`movement_classifier_manager.py:157` vs
`MovementStimController.js:87-94`, `:34-35`) — and the browser's version is the one that
actuates the patient.

### B5 — Recording has three server subsystems and a frontend-only mutex

Every chunk is independently offered to three sinks: plain recording (`:1226`,
filtered), mv capture (`:1188`, raw→RAM), online writer (`:1190`, raw→disk). They don't
collide on files, but arbitration lives only in the frontend
(`setExternalRecording`, `SequencePlayer.js:82-84`) and only covers one of the pairings —
`DecompositionController.stopClassification()` still stops recordings it didn't start
(`:103-111`). `MovementSessionController` sends `mvOnlineStop` unconditionally on
reset/end (`:455`, `:483`), and the server broadcasts `online_saved` even when nothing was
open (`websocket_server.py:513`) — the operator log claims saves that didn't happen.

### B6 — Failures are swallowed at every layer

Per-chunk exceptions in decomposition and movement processing are printed and ignored
(`:1276-1277`, `:1313-1314`) — a model that throws on every chunk looks like a stale-but-
calm UI with a stimulator gate that never closes (S3). Frontend: server errors are dropped
unless they match `/config/i` (`MovementSessionController.js:700-702`); `index.js:27`
swallows all boot failures into `console.log` (black screen); `index.js:31-34` has no
`.catch` at all on the manifest fetch.

---

## 5. Security posture

Acceptable only because of the loopback bind, and worth stating plainly: unauthenticated
clients can make the server `pickle.load` an arbitrary client-supplied path
(`websocket_server.py:414,445,822`; `movement_engine.py:315-317`; `dsp/loading.py:33-34`;
`decomposition_manager.py:100`) — arbitrary code execution by design; `set_otb_config`
parses any XML path (`:601,653`; `quattrocento.py:107`); `mv_record_start` embeds the
whole client message verbatim into the saved pickle (`:479`, `:866`). If this server is
ever bound to a LAN, all of this is remote.

---

## 6. Revised priority stack

The two existing plans stay valid; this reorders around them.

1. ~~Commit the tree~~ — done (`0ed49db`).
2. **Server-side stimulation authority (S8) + fend Phase A + one visible banner** — done
   (`d3a90de..3426ffd`: implementation, adversarial QA, and fix rounds; both sides
   harness-tested). Still open from this package, and closable only outside the code:
   clinical review of the `StimLimits` ceilings, and the bench tests in the scope note
   below (samolator behaviour on client death, retry-give-up visibility, Windows Ctrl+C
   teardown).
3. **Recording integrity (P5, B1, B2):** incremental flush, `_mv_paused` fix, recording
   state in the connect payload, sidecar-first online writer, filename collision fix.
4. **Process lifecycle (P1–P3):** fix or replace the launchers; port-bind failure must be
   loud; add the missing install step; restore/replace the sim data file.
5. **The dedup pass (D1, D2, D4, D5–D7):** sim fork → `SimulatedDevice` factory; one CLI;
   shared protocol/limits file (backend clamps against it); one render module; delete the
   unreachable `configs/`.
6. **Session-control ownership (B3, B5, S4):** single stim requester with ownership token;
   derived configs restored on session end; recording token (fend plan C5/C6 fold in here).
7. Then continue `FEND_REFACTORING_PLAN.md` B, C, D… as written — several items shrink
   once ownership moves server-side.

---

## 7. Smaller items worth batching into adjacent work

- `RippleWebSocketServer` misnomer + wrong-hardware console prints (D1).
- Deprecated `websockets` legacy API (`websocket_server.py:38-39`) — breaks on next major.
- `apply_filters` loops 256 channels in Python per 50 ms chunk (`dsp/processing.py:70-79`).
- `process_chunk` mutates the caller's `centroids` in place (`:226-231`) while
  `start_recording` snapshots it as static config (`websocket_server.py:978-984`).
- `EMGChannelView` hardcodes 64-channel fallbacks (`:19,313,386`) against a 256-channel
  Quattrocento; reconnect race renders the wrong channels with correct-looking labels.
- `DecompositionView` allocation churn on a permanent subscription (`:260`, `:330-332`).
- Recording-error messages broadcast to all clients while `load_model` errors are unicast
  (`websocket_server.py:957,1017` vs `:431`).
- Dead: `StimulationClient.active`; `MovementClassifierManager.set_trig_mid`; unused
  `import sys`/`Path` leftovers in three modules; `EMGClient.onData` fan-out.
- `window.app` global + append-only subscriber arrays (`SequencePlayer.js:108-134` return
  no unsubscribers) — safe only while nothing is ever re-created; documented boot-order
  coupling in `App.js:37-38,50-51,122` with no assertion.

---

## Notes on scope

Three audit passes were independent and overlapped deliberately; where they disagreed,
the file was re-read. Everything above carries at least one direct file:line read.
Not audited: `muniverse/` internals, `mvdecoder/`, both `legacy_*` trees, and the
`samolator` stimulation controller (outside this repo — its behavior on client death is
unknown and matters for S2; worth testing on a bench load).

One process-boundary behavior was **not** bench-verified: whether the stimulator hardware
latches an infinite train when the controller loses its client. Verify with a scope on a
bench load before relying on any software fix in section S8.
