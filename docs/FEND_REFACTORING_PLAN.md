# PatientGUI — `fend/` refactoring plan

**Goal:** take a rushed prototype to something that can be trusted to run experiments
unattended. The deliverable of this module is *EMG recordings and stimulation events that
are correct, complete, and traceable*. Everything else — the 3D hand, the timeline, the
panels — exists to serve that. Where UI and data integrity conflict, data integrity wins.

**Status (2026-08-17):** Phase 0 is complete — `package.json`, `index.html`, the
provenance defines in `vite.config.js`, and `_buildTrialMetadata` are all done (note the
stamped `GIT_SHA` only became meaningful with commit `0ed49db`, when the tree first
entered version control). Phases A–H are outstanding; A1 (`emergencyStop` not terminal)
and A4 (`skipToNext` doesn't stop stim) were re-verified live in the source at audit time.

**Read `docs/ARCHITECTURE_AUDIT.md` before starting Phase A.** Its conclusion: Phase A's
guarantees cannot be anchored in the browser — a crashed tab runs no JavaScript, and the
server currently stops stimulation on no teardown path whatsoever (audit S2). Phase A
remains necessary as the operator-facing layer, but it lands together with the
server-side stimulation authority described in audit S8, not instead of it.

---

## The two organizing rules

> **1. Every piece of state has exactly one owner. Components read it through that owner
> and never keep their own copy.**

Nearly every defect found in review is two components disagreeing about who is in charge:
`currentModelIndex` (3 writers, one of which forgets), the trigger channel (3 owners, and
the code comments document a past bug from exactly this), `isAddingSegment` (2 hand-synced
copies), `nMUs` (3 copies), player state (mirrored into 3 views), recording control (3
independent callers of `start_recording`). Fixing the instances without fixing the pattern
just moves the bug.

> **2. Nothing fails to the console.**

Today: a dead socket is a `console.warn` in `_send`; a server `error` message is routed by
running a regex over its human-readable text (`MovementSessionController.js:700-702`) and
dropped if it doesn't match `/config/i`; a failed model load logs and hangs; a failed
export silently produces nothing. An operator running an experiment has no way to learn
that anything went wrong. Every failure path must terminate in something the operator can
see without opening DevTools.

---

## Ordering principle

Phases are ordered by **cost of delay**, not by difficulty.

- Phase 0 first because every session run without it produces permanently untraceable data.
- Phases A–C before any structural work, because they are the ones that can hurt a subject
  or void a session, and they are all small.
- Phase D before E–H, because the later phases are much harder to do correctly on top of
  duplicated state.

Do not start the UI restructure (F) first. It is the most visible work and the least urgent.

---

## Phase 0 — Provenance and identity

**Why first:** `SequencePlayer._buildTrialMetadata` (`:337-368`) sends subject, session and
settings to the server for permanent storage and **includes no software version**. Every
recording made so far cannot be traced to the code that produced it. This is the only item
in the plan whose cost accrues retroactively — each session run without it is permanently
un-attributable.

- `package.json:2,4` — real `name`, real `version`. It is still `"vite-threejs"` / `"0.0.0"`
  from the starter template.
- Add app version **and git commit SHA** to `_buildTrialMetadata`. Vite `define` already
  wires `APP_VERSION` (`vite.config.js:7-9`); add the SHA the same way.
- `index.html:7` — title is `Vite App`. `GUIView.js:56` writes `document.title = "APP 0.0.0"`.
- `GUIView.js:22` — **a hardcoded personal absolute path**
  (`C:\Users\velar\SynologyDrive\Personal\...`) is compiled into every build. Move to config.
- `fend/README.md` is the unmodified starter README. Replace or delete; it documents
  shortcuts that no longer exist.

**Files:** `package.json` · `vite.config.js` · `index.html` · `src/ui/SequencePlayer.js`
(`_buildTrialMetadata` ~:337) · `src/gui/GUIView.js` (`:22`, `:56`) · `README.md`

**Verify:** record a trial, open the resulting file, confirm version + SHA are present.

---

## Phase A — Stimulation safety

Everything here concerns current delivered to a human subject. All four were read directly
in the source, not inferred.

**Files (phase):** `src/emg/MovementSessionController.js` · `src/emg/MovementStimController.js`
· `src/emg/EMGClient.js` · `src/ui/SequencePlayer.js`

### A1 — Emergency stop must be terminal

`MovementSessionController.emergencyStop()` (`:494-501`) clears timers, stops the timed
stim and calls `stim.emergencyStop()` — but **does not change `this.state` and does not
stop the player**. The sequence keeps cueing. On the next MOVE phase, `:687` still matches
`state === SESSION.SENSOR_SIM`, calls `_onSensorPhase`, which re-arms `_stimStartTimer`
(`:605`) and fires `stimulateStart` again at `:642`.

The sensor pass drives the stimulator **directly through `this.client`**, so
`MovementStimController`'s `enabled = false` latch (`MovementStimController.js:61-65`)
never sees this path at all.

**Net effect: press STOP, and the subject is stimulated again 1–2 seconds later.**

Fix: `emergencyStop()` sets a terminal `HALTED` state, calls `this.player.stop()`, and sets
a `_halted` flag that `_onSensorPhase` and `_startTimedStim` both bail on. Clearing `_halted`
must require an explicit operator action, not a state transition.

→ `MovementSessionController.js` (`emergencyStop` :494, `_onSensorPhase` :596,
`_startTimedStim` :631, the `SESSION` enum) · `MovementStimController.js` (the bypassed latch)

### A2 — A stop that never reached the server must not be recorded as success

`MovementStimController.stopAll()` (`:116-121`) sets `this.stimming = false`
unconditionally. `EMGClient._send` (`:550-558`) returns `false` on a dead socket and **no
caller anywhere checks the return value** (~20 command methods). Socket drops while
stimming → stop is discarded → local state says "stopped" → nothing retries on reconnect.
The train is documented as infinite until stopped (`EMGClient.js:230-241`).

Fix: `stopAll()` only clears `stimming` on a confirmed send; on failure it stays true, is
retried on reconnect, and raises a visible alarm.

→ `MovementStimController.js` (`stopAll` :116) · `EMGClient.js` (`_send` :550 return value;
the reconnect hook) · every `client.stimulate*` call site that ignores the boolean

### A3 — Dead-man watchdog

`_onDecision` (`MovementStimController.js:69-80`) is the only thing that can close the gate.
If decisions stop arriving — server stall, socket drop, model unloaded without a status
message — the train runs indefinitely. There is no timeout independent of the message stream.

Fix: hard timeout on any active train, armed at `_start()`, independent of decisions. This
is the backstop for every failure mode not enumerated here.

→ `MovementStimController.js` (`_start` :103, `_onDecision` :69) · the same watchdog also
covers the sensor-pass train in `MovementSessionController.js` (`_startTimedStim`)

### A4 — Stimulation outliving playback control

- `skipToNext` / `skipToPrevious` / `jumpToItem` (`SequencePlayer.js:256-309`) call
  `_clearPhaseTimer()` but **not `_stopStimulation()`** — unlike `pause()`, `stop()`,
  `reset()` and `_onSequenceComplete()`, which all do.
- `_playCurrentItem` (`:596`) is `async` and awaits model load and tutorial. The
  `prepTime > 0` branch re-checks `isPlaying` inside its callback (`:648`); the
  **`prepTime === 0` branch at `:652-656` does not**, and runs `_setPhase(PHASE.MOVE)` +
  `webglView.play()` unconditionally after the awaits. With `prepTime = 0`, a Stop pressed
  during the tutorial is undone when the subject then presses "I'm Ready".

> **Scope note.** The original review claimed this fires regardless of `prepTime`. It does
> not — the guarded branch is genuinely guarded. The hole is real but requires
> `prepTime === 0`. Verified by reading `SequencePlayer.js:645-656`.

→ `SequencePlayer.js` (`skipToNext` :256, `skipToPrevious` :280, `jumpToItem` :298,
`_playCurrentItem` :596 — the `prepTime === 0` branch :652, `_stopStimulation`)

**Verify Phase A:** with the stimulator on a bench load and a scope on the output — press
STOP mid-sensor-pass and confirm no further output; pull the network cable mid-train and
confirm the watchdog fires; set `prepTime = 0`, Stop during a tutorial, then confirm.

---

## Phase B — Failure visibility

**Files (phase):** `src/emg/EMGClient.js` (reconnect, connect, error routing, status
rehydration) · `src/emg/MovementSessionController.js` (disconnect handling, error filter,
train timeout, calibration correlation) · `src/emg/MovementSessionView.js` (spinner, sticky
STOP) · `src/emg/EMGChannelView.js` (`isRecording`) · a shared connection-banner component
usable in patient mode (new — see B2). **Backend touch point:** B3/B7 want the server to send
an error *category* and a calibration *id* — `bend/` change, coordinate with that side.

### B1 — The session layer never learns the connection dropped

Only `EMGChannelView` (`:283`) and `GUIView` (`:445`) subscribe to `onDisconnect`. Neither
`MovementSessionController` nor `SequencePlayer` does. A drop mid-recording means the
subject performs every cued rep, the pulsing red RECORDING indicator stays up
(`EMGChannelView.js:136-150`, `MovementSessionView.js:139-147`), and the block is lost with
no signal to anyone.

`isRecording` is also only ever assigned from a server message (`EMGClient.js:778`), so it
is never cleared on disconnect — the UI actively asserts that data is being captured while
nothing is connected.

Fix: session layer subscribes to `onDisconnect`; halts the run; clears `isRecording`;
raises a blocking banner **that is visible in patient mode** (see B2).

→ `MovementSessionController.js` (`_subscribe` :668) · `SequencePlayer.js` (halt on drop) ·
`EMGClient.js` (`isRecording` :778, `onDisconnect` fan-out :953) · `EMGChannelView.js` (:283)

### B2 — Patient mode has no connection indicator at all

The only indicators are inside the `EMGChannelView` modal and the Tweakpane field — both
unavailable in patient mode, which is the mode experiments actually run in.

→ new shared banner component · mounted by `App.js` / `ProgressSidebar.js` so it renders in
patient mode · fed by `EMGClient` connect/disconnect + reconnect state

### B3 — Error routing

- `MovementSessionController.js:700-702` filters server errors with `/config/i.test(msg)`
  and drops everything else. Routing by regex over human-readable text is not a routing
  mechanism — the server should send a category, and the frontend should display anything
  it cannot classify rather than discarding it.
- `trigger_monitor_status` is an **empty case** (`EMGClient.js:606-607`) — a refused monitor
  request shows a permanently blank box.
- `_send` failures are `console.warn` only.

→ `MovementSessionController.js` (`onError` handler :700) · `EMGClient.js`
(`trigger_monitor_status` :606, `_send` :550, generic `error` dispatch) · `GUIView.js`
(current error display) · **`bend/`** (send an error category, not free text)

### B4 — Training and calibration can hang forever

`MovementSessionView` renders an indefinite spinner for `TRAINING` / `TRAINING_CLEAN` /
`CALIBRATING` (`:149-159`). If `mv_train_status` never arrives, the operator waits forever
with no timeout and no escape but ✕, which discards the block.

→ `MovementSessionController.js` (arm a timeout on entering those states) ·
`MovementSessionView.js` (:149 spinner → timed-out error state)

### B5 — Emergency-stop confirmation is erased within one frame

`MovementSessionView.js:199,219` set `"STIM STOPPED"`, which the next `_onDecision`
(`:434-443`, ~15 Hz) overwrites with the predicted label. The operator gets ~60 ms of
confirmation. Halted state must be sticky and must gate the decision renderer.

→ `MovementSessionView.js` (`_onDecision` :434 gated by the halted flag from A1) — one file

### B6 — Reconnect policy

`EMGClient.js:965-971` — fixed 3000 ms, no backoff, no jitter, no cap, no UI state. Also
`disconnect()` (`:128-136`) sets `autoReconnect = false` permanently with no way to restore
it, so a later `connect()` succeeds once and never reconnects again.

Also: `connect()` (`:80-84`) only early-returns on `readyState === OPEN`. Called while
`CONNECTING`, it overwrites `this.ws` while the old socket's `onclose` closure still holds
`this` — that handler will later flip the UI to "Disconnected" **on a live connection** and
schedule a competing reconnect. Fix with a socket-identity check in every handler.

→ `EMGClient.js` only (`connect` :80, `disconnect` :128, `onclose`/`onerror` handlers :101,
`_scheduleReconnect` :965) — self-contained, but expose reconnect state so B2's banner can read it

### B7 — Spurious closed-loop start on reconnect

`_rehydrateStatusFromConnect` (`EMGClient.js:697-719`) synthesizes a `movement_status` on
every connect from `msg.movement_active`. `MovementSessionController:703-716` reacts to any
`s.active && state === CALIBRATING` by consuming `_calibThenOnline` and starting guided
online → `stim.enable()`. A blip during pending calibration, with a *previous* model still
loaded server-side, jumps straight into closed-loop stimulation driven by an uncalibrated
model.

Fix: correlate against the specific calibration request (id, or a `calibrated` flag in
`mv_train_status`), never a bare `active` boolean.

→ `EMGClient.js` (`_rehydrateStatusFromConnect` :697) · `MovementSessionController.js`
(the `CALIBRATING` reaction :703) · **`bend/`** (attach a calibration id / `calibrated` flag)

---

## Phase C — Session integrity

**Rule for this phase: a run either records correctly or refuses to start.**

**Files (phase):** `src/ui/SequencePlayer.js` (the hub — play/pause, completion, tutorial
await, recording ownership) · `src/webgl/WebGLView.js` (load-error drain) ·
`src/ui/TutorialOverlay.js` (promise cancellation) · `src/emg/DecompositionController.js` &
`src/emg/MovementSessionController.js` (recording arbitration) · `src/emg/EMGClient.js`
(`startRecording` guard) · `src/gui/GUIView.js` & `src/ui/ProgressSidebar.js` (button labels)

### C1 — A failed model load hangs the run permanently

`WebGLView.loadModel`'s error callback (`:209-216`) builds a fallback cube and **never
drains `this._modelLoadedOnce`**, which is only drained in the success path (`:196-197`).
`SequencePlayer._loadModelAndWait` (`:684-704`) awaits a promise resolved solely from that
queue. One bad filename, one transient blip → the await never settles, the sequence stops,
EMG keeps recording, subject watches a grey cube. Same at `DecompositionController.js:325`.

Fix: drain the queue in the error path, add a load timeout, and surface the failure.

→ `WebGLView.js` (`loadModel` error cb :209, `_modelLoadedOnce` :196) · `SequencePlayer.js`
(`_loadModelAndWait` :684) · `DecompositionController.js` (:325)

### C2 — "Pause" is not a pause

`play()` (`:165`) is not a resume:
- `:179` `_initTimeline()` destroys `_timeline.events` and all accumulated phase/rep
  provenance for the session so far.
- `:183-186` calls `startRecording()` again. `EMGClient.startRecording` (`:164`) has **no
  `isRecording` guard** — it blindly sends `start_recording`.
- `:197` restarts the current item from the rep boundary.

`GUIView.js:98-102` labels this button Play/**Pause**. Either make it a real resume or
rename it and remove `pause()` — but the current label is a lie in both directions.

→ `SequencePlayer.js` (`play` :165, `pause` :210, `_initTimeline` :373) · `EMGClient.js`
(`startRecording` guard :164) · `GUIView.js` (button :98) · `ProgressSidebar.js` (footer button)

### C3 — Post-completion Start produces junk recordings

`_onSequenceComplete()` (`:908`) never resets `currentItemIndex`. The next press of the
patient-facing Start button runs `_initTimeline()` + `startRecording()`, hits
`config.items[length] === undefined` (`:600`), immediately re-completes, and stops the
recording. **Every press writes another empty file** and the subject sees a button that
does nothing. Only Stop resets the index (`:224`), and nothing says so.

Fix: reset the index on completion, and add an explicit completion state to the patient
view (see F3).

→ `SequencePlayer.js` (`_onSequenceComplete` :908) · `ProgressSidebar.js` (:118 completion
handler) · `PhaseOverlay.js` (:242 IDLE hide) — completion UI itself lands in F3

### C4 — Tutorial promise deadlock

`TutorialOverlay.hide()` (`:150-152`) never invokes or nulls `_confirmCallback`, and
`stop()` never clears `_tutorialResolve`. The `_showTutorial` promise
(`SequencePlayer.js:661-677`) stays pending forever and `_playCurrentItem` is suspended at
`:623` permanently. Re-entering patient mode and pressing Start creates a **second**
concurrent `_playCurrentItem` chain driving the same `webglView.action`; both register and
deregister the same `finished` listener, making rep counting nondeterministic.

Fix: every await in the player must be cancellable, and cancellation must run on `stop()`,
`reset()` and mode change.

→ `TutorialOverlay.js` (`hide` :150, `_confirmCallback`) · `SequencePlayer.js`
(`_showTutorial` :661, `stop` :220, `reset` :245, `_tutorialResolve`) · `PatientModeController.js`
(mode-change hook that must cancel)

### C5 — Three independent owners of recording

- `SequencePlayer.js:185,235,921`
- `DecompositionController.js:89,106` — `stopClassification()` calls `stopRecording()`
  **unconditionally**, terminating a recording it did not start
- `MovementSessionController` via `mv_record_*`

`setExternalRecording` (`SequencePlayer.js:78-80`) papers over exactly one of the three
pairings. Give recording a single owner with an explicit token; non-owners cannot stop it.

→ `SequencePlayer.js` (:185, :235, :921, `setExternalRecording` :78) ·
`DecompositionController.js` (:89, :106) · `MovementSessionController.js` (`mv_record_*`) ·
`EMGClient.js` (owns the token / current recorder)

### C6 — Stale `_onSeqComplete` misfires

The controller keeps one `player.onSequenceComplete` handler (`:669-677`) and one callback
slot (`:504`). Abandon a block without `reset()`, then play a sequence from the config
panel, and the stale callback fires — `_onTrainRecorded()` issues `mvRecordStop` + `mvTrain`
against whatever the server is holding. Scope callbacks to a run token.

→ `MovementSessionController.js` only (`_onSeqComplete` slot :504, `_subscribe` :669,
`_onTrainRecorded` :515) — same run-token mechanism as C5

---

## Phase D — Single ownership

This is the structural core. It is the phase that makes E–H tractable. The "Current owners"
column below *is* the file map for each row — those are the sites that stop writing the
state and start reading it through one owner.

**Files (phase), beyond the table:** `src/webgl/WebGLView.js` · `src/gui/GUIView.js` ·
`src/ui/SequencePlayer.js` · `src/emg/DecompositionController.js` · `src/emg/TriggerMonitor.js`
· `src/ui/TimelineView.js` · `src/emg/EMGClient.js` · `src/emg/DecompositionView.js` ·
`src/ui/ProgressSidebar.js` · `src/ui/PhaseOverlay.js` · `src/ui/PatientModeController.js` ·
`src/ui/ModeMenu.js` · `src/ui/TutorialOverlay.js` · `src/styles/_patient-mode.scss`

| State | Current owners | Target |
|---|---|---|
| `currentModelIndex` | `WebGLView` field, written by `GUIView:688` and `SequencePlayer:701`; **`DecompositionController:321-327` forgets**, tracking its own `currentlyLoadedFile` | `loadModel()` owns it; nobody writes it externally |
| Trigger channel | `MovementSessionController.params.trigCh` (3 writers), `GUIView.triggerChannel`, `TriggerMonitor.channel` | One owner. The comment at `MovementSessionController.js:741-746` documents a past bug from this exact split — and it is still live: `GUIView.js:424` starts the monitor on one value while `toggleTrigger` (`:126-130`) starts it on another |
| `isAddingSegment` | `GUIView.js:250` ↔ `TimelineView.js:313`, hand-synced | One owner |
| `nMUs` / active flags | `EMGClient`, `DecompositionController`, `DecompositionView` | One store, subscribed to |
| Player state | mirrored into `ProgressSidebar:25-29`, `PhaseOverlay:20-21`, `GUIView` | Read `getState()` on demand |
| Mode | `PatientModeController` knows 2 modes; `ModeMenu` presents 3 | Session is a modal, not a mode — or promote it to a real third mode. Not both |

**Consequence of the `currentModelIndex` bug** (worth calling out because it corrupts
saved data, not just display): `WebGLView.playAnimation:271` uses that index to key the
`SegmentManager` context. After a classification-driven model swap, **segments are saved
and loaded under the wrong model's key.**

Also in this phase, because it is the same disease:

- **Two competing visibility systems.** Mode visibility is CSS-driven off
  `body.config-mode` / `body.patient-mode`, but `ProgressSidebar:229-235` and
  `PhaseOverlay:254-260` also set inline `display`. Inline wins permanently, so one `hide()`
  call makes a component unrecoverable by mode change. `PhaseOverlay` uses `opacity` in
  `_updateDisplay` and `display` in `show`/`hide`. Pick one.
- **Two competing styling systems.** ~975 lines of SCSS and ~215 inline `style` assignments
  across 11 files. This is why `_patient-mode.scss:515-518` (re-centring the phase overlay
  for the sidebar) is **dead code** — it cannot beat the inline `left: 50%` at
  `PhaseOverlay.js:82`. The overlay is never actually re-centred.
- **`TutorialOverlay` drives the renderer directly** (`:179-183`, `:215-224`), including
  writing `app.webgl.isPlaying` and a hardcoded `setLoop(2, Infinity)` with the THREE enum
  meaning in a comment. Animation lifecycle currently has three owners — `WebGLView`,
  `SequencePlayer`, `TutorialOverlay` — which is why C4 exists.
- **No component has a working `destroy()`.** Window/document listeners in `TimelineView`
  (`:156`, `:304-305` — and `attachSegmentEvents` double-registers if called twice),
  `SequenceConfigPanel:122`, `ModeMenu:31`, `EMGChannelView:241`; a `ResizeObserver` in
  `DecompositionView:212`; a `<style>` appended per instance at `EMGChannelView:171-178`.
  `SequencePlayer`'s and `SegmentManager`'s subscribe methods return nothing, so there is no
  unsubscribe. `PatientModeController.onModeChange` is the one that gets this right, and
  nobody uses the unsubscriber it returns.

  → `destroy()` sweep touches: `TimelineView.js` · `SequenceConfigPanel.js` · `ModeMenu.js` ·
  `EMGChannelView.js` · `DecompositionView.js` · `TutorialOverlay.js` · `ProgressSidebar.js` ·
  and the subscribe-returns-unsubscriber change in `SequencePlayer.js` + `SegmentManager.js`.
  Adopt the pattern already correct in `PatientModeController.js:110`.

---

## Phase E — Movement identity and segment timing

**This phase unblocks gesture classification. Do it before accumulating more training data.**

**Files (phase):** `src/config/movements.js` (the registry — add stable ids) ·
`src/config/sequenceConfig.js` & `fend/configs/*.json` (items reference movements) ·
`src/config/segmentConfig.js` (per-model timing) · `src/emg/MovementSessionController.js`
(label derivation :860) · `src/gui/GUIView.js` (dropdowns bind index, :674, :583) ·
`src/webgl/WebGLView.js` (clip duration :256) · `src/ui/SegmentManager.js` (segment defaults)
· `src/config/configStore.js` (migration for saved configs) · `fend/public/*.glb` (asset
deletes). **Note:** trained models live in `bend/` — the label rename must be coordinated so
existing `.pkl` models and their label maps stay valid or are migrated.

### E1 — Labels must not be derived from filenames

`MovementSessionController.js:860` derives the training label by stripping `.glb`:
`(it.label || it.model || 'mov').replace(/\.glb$/i, '')`. `onlineStimulation.patterns` is
keyed by that same derived string (`sequenceConfig.js:63-64`).

Two consequences:
1. Renaming an asset silently invalidates trained classifier labels and stim patterns.
2. `fastwristext` and `wristext` are **two distinct classifier classes for one physiological
   movement** — actively harmful to training.

Fix: an explicit, stable `id` on each movement in the registry. The filename becomes a
rendering detail. Dropdowns must bind that id, not an array index — `movements.js:15-18`
already warns in prose that reordering the array silently changes what a persisted
selection resolves to (`GUIView.js:674-676`, `:583`).

→ `movements.js` (add `id`) · `MovementSessionController.js` (:860 label from `id`) ·
`sequenceConfig.js` + `configs/*.json` (`patterns` keys, item refs) · `GUIView.js` (dropdown
value = id, :583, :674)

### E2 — Per-model segment timing (prerequisite for E3)

`segmentConfig.js:29` has `models: {}` **empty**, so every model gets the same hardcoded
0-2 / 2-12 / 12-14 split (`:12-16`) regardless of its actual clip duration — which
`WebGLView.js:256` computes per clip and which certainly varies across 13 assets. A 6-second
clip gets a "Movement" segment running from 2 s to 12 s, i.e. mostly past the end.

Fix: derive segment defaults from the real clip duration; allow per-model overrides that
are actually populated.

→ `segmentConfig.js` (`models` :29, default split :12) · `WebGLView.js` (clip duration :256,
segment reload :190, :272) · `SegmentManager.js` (`load` :207, defaults)

### E3 — Delete the "fast" assets

Confirmed with the author: the fast variants are *the same movement held for less time*.
That is a **segment duration**, not a different animation — so once E2 is real, a "fast"
item is the same `.glb` with a shorter hold segment.

Delete `fastfingerext.glb`, `fastlastfingers.glb`, `fasttripodpinch.glb`,
`fastwristext.glb` (~1.1 MB of ~4.0 MB total), remove their registry entries, and express
the variants as sequence items. Collapses the duplicate classifier labels from E1 at the
same time.

> **Hazard:** any already-trained model or saved config referencing `fast*` labels breaks.
> Do E1's id mapping first and provide a migration for saved configs in `localStorage`.

→ delete `public/fastfingerext.glb`, `public/fastlastfingers.glb`,
`public/fasttripodpinch.glb`, `public/fastwristext.glb` (and the stale `dist/` copies) ·
`movements.js` (drop entries) · `sequenceConfig.js` + `configs/*.json` (rewrite `fast*` items
as normal item + short hold) · `configStore.js` (`localStorage` migration)

> **Also verify:** `abduction.glb` and `adduction.glb` are byte-for-byte the same *size*
> (273,508) with different hashes. Two anatomically opposite movements exporting to
> identical byte counts is worth one manual check — confusing them corrupts a dataset
> silently and undetectably.
>
> → `public/abduction.glb`, `public/adduction.glb` — open both, confirm the motion matches
> the name. No code change if they check out.

---

## Phase F — The two surfaces

Decision taken: **Tweakpane is demoted to an engineering panel**; a first-party operator
surface owns the experiment flow. `SequenceConfigPanel` + `ProgressSidebar` are already
most of it.

### F1 — Split the surfaces

- **Engineering panel** (Tweakpane, behind a keystroke): COM port, MU indices, trigger
  channel, decomposition config, stats. Things touched during bring-up.
- **Operator surface** (first-party): subject/session, sequence selection, run controls,
  connection + recording state, live progress. Things touched every session.

→ `GUIView.js` (761 lines — the split happens here: keep engineering bindings, move operator
flow out) · `SequenceConfigPanel.js` + `ProgressSidebar.js` (grow into the operator surface) ·
`_tweakpane.scss` (the 47 `!important`) · `package.json` (pin `tweakpane`, `three`)

Tweakpane's specific failures are documented in the source itself: text bindings do not
flush typed values until blur, so `GUIView.js:394-397` and `:553-555` read the raw DOM and
needed separate "Set" buttons; `_rebuildMUDropdowns:633-669` carries a comment explaining
that controls appear at the end of the folder because positional inserts are impossible.
Keeping it for engineering use accepts these; moving the operator flow off it does not.

`_tweakpane.scss` also has **47 `!important` declarations in 147 lines** against Tweakpane's
private class names, under a `^4.0.5` caret range. Pin the version.

### F2 — Repair the timeline and segment editor

Confirmed in use, so it gets fixed rather than cut.

- **Drag resizes the wrong segment.** `TimelineView.js:347-408` captures `index` once at
  pointerdown; every pointermove calls `updateSegment(index, …)`, which **re-sorts the
  array** (`SegmentManager.js:110`). Cross a neighbour's start time and you begin resizing
  a different segment. Fix: stable ids, not array indices.
- **The dragged element is destroyed on every mousemove.** The same call fires
  `_notifyChange()` → `renderSegments()` → `segmentsEl.innerHTML = ''` (`:469`), and
  `_autoSave()` writes `localStorage` synchronously per mousemove. Defer sort, save and
  re-render to pointerup.
- **Existing segments are unclickable in normal mode.** `.timeline__range` is `z-index: 2`
  (`style.scss:104-107`), `.timeline__segments` is `z-index: 1` (`:130-136`). Only
  `.timeline__scrub--adding` lifts segments to `z-index: 10`. So select and resize are
  **unreachable** unless the operator first arms Add mode — whose sole affordance is a
  Tweakpane button re-titling itself to `[ Adding Movement... ]`.
- **Speed sliders desync.** `WebGLView.js:190-191,272-273` reload `segmentManager` on every
  model/animation change; `GUIView.updateSegmentSpeedsFromManager` is called **only** after
  JSON import (`:282`). After any model switch the sliders show stale values while the
  engine uses different ones.
- **No validation on import.** `SegmentManager.importJSON:293-316` and `load:207-241` accept
  anything passing `Array.isArray`. Malformed data yields `NaN` segments that poison
  `getSpeedAtTime` and render as `left: NaN%`.
- **Keyboard-inoperable.** All interaction is pointer-only on absolutely-positioned divs.

→ `TimelineView.js` (drag :347, `renderSegments` :467, `attachSegmentEvents` :294, z-index
interplay) · `SegmentManager.js` (`updateSegment` sort :110, `_autoSave` :112, `importJSON`
:293, `load` :207) · `style.scss` (`.timeline__range`/`.timeline__segments` z-index :104, :130)

### F3 — Patient view

Freedom to change was confirmed, so these are real fixes, not workarounds.

**Files:** `src/ui/PhaseOverlay.js` (labels, contrast, `aria-live`, IDLE/completion) ·
`src/styles/_patient-mode.scss` (contrast, typography, `prefers-reduced-motion`, focus) ·
`src/ui/ProgressSidebar.js` (Stop confirm, completion, EMG-view exposure, item semantics) ·
`src/ui/TutorialOverlay.js` (focus, dialog semantics) · `src/ui/SequenceConfigPanel.js` (dirty
check, dialog semantics) · `src/ui/PatientModeController.js` (`Ctrl+M` guard) · `src/App.js`
(`keyup` `activeElement` guard :212) · `src/ui/ModeMenu.js` (menu semantics) ·
`src/styles/style.scss` (shared pulse animations).

- **The green light says "HOLD."** `PhaseOverlay.js:47-48`. The file's own header (`:8`) says
  green = "Move", the config key is `MOVE`, and `SequencePlayer.js:7` documents it as "Move".
  The string is the outlier. **Confirm the clinical intent before changing** — if subjects
  are meant to sustain a contraction, the label is right and the comments are wrong.
- **Contrast.** MOVE badge is `#fff` on `#00cc44` ≈ **2.1:1** — the go-signal fails WCAG AA
  even at the large-text threshold. Same for the play button and "I'm Ready"
  (`_patient-mode.scss:265-269`, `:409`). PREP and REST pass, so the failure is inconsistent
  and cheap to fix.
- **Typography.** Exercise name 14 px, rep count 12 px (`_patient-mode.scss:212,220`), and
  ellipsis-truncated in a 280 px column — the two things the subject most needs to read, at
  body-copy size, at 1–3 m viewing distance. Completed items at `opacity: 0.5` fall below 3:1.
- **Stop is destructive, unconfirmed, and the only control the subject has** — full-width
  `#cc3333` (`:258-293`), resets both indices and ends the recording. Offer Pause (once C2
  makes it real) and confirm Stop mid-session.
- **No completion feedback.** The overlay hides itself on IDLE and the sidebar button flips
  back to "Start Exercises". The subject finishes a 10-minute protocol and the screen goes
  blank.
- **`prefers-reduced-motion` is absent** and two infinite pulse animations run continuously
  (`style.scss:267`, `_patient-mode.scss:198,297-304`). One media query.
- **Screen reader:** no `aria-live` on the phase badge or countdown, no dialog semantics on
  either modal, no focus management. The `ModeMenu` is a `div` with no menu semantics.
- **"View EMG Data" is one tap away in the subject-facing sidebar** (`ProgressSidebar.js:67-76`).
  Decide deliberately whether that belongs there.
- **Config modal discards edits silently** on Escape, click-outside, × and Cancel — no dirty
  check. `_resetToDefaults` is the only confirmed action in the whole UI layer, and it is
  inconsistent: it persists immediately, so Cancel does not undo it.
- **`Ctrl+M` flips mode with no guard** (`PatientModeController.js:146-151`) — during a run,
  with the modal open, mid-session. Nothing in `setMode` stops the sequence, the recording,
  or a running stim train.
- **Global `keyup` shortcuts fire inside text fields.** `App.js:212-217` binds `g`/`p`/`h` on
  `window` with no `activeElement` check, and there is no `stopPropagation` on any of the ~12
  text inputs. Typing a subject ID containing "p" or "h" toggles the admin pane and resets
  the camera. Uses deprecated `keyCode`.

### F4 — Isolate decomposition

Decision taken: keep, behind a flag.

- It must stop driving the hand directly. `DecompositionController` imports `three` and
  manipulates `LoopOnce`, `action.reset/timeScale/time`, `webgl.animationTime` and
  `webgl.segmentManager.segments` (`:246-328`). Split EMG→label from label→animation.
- `hide()` (`DecompositionView.js:388-392`) stops the RAF but leaves the data subscription
  permanent, so buffer work continues all session for an invisible panel.
- `currentLabel` is committed at `:218` **before** the "no model configured" guard at
  `:233-236`, so state claims MOVE while the hand never moved, and the next REST transition
  reverses an animation that never played.

→ `DecompositionController.js` (three.js import :1, animation control :246–328, `currentLabel`
:218) · `DecompositionView.js` (`hide` :388, permanent subscription :260) · `WebGLView.js`
(the animation API the controller should call instead of reaching into) · a feature flag
(`configStore.js` or an env/config toggle)

---

## Phase G — Signal rendering

The plots are currently not clinically readable, which matters because this module's job is
EMG.

**Files (phase):** `src/emg/EMGChannelView.js` (normalisation, decimation, DPR, page count) ·
`src/emg/DecompositionView.js` (normalisation, drop detection, spike batching, RAF, DPR) ·
`src/emg/TriggerMonitor.js` (already-correct envelope to copy; move its draw into RAF) ·
`src/emg/EMGClient.js` (`sampleRate`-aware buffer sizing :44/:648, gap/`n_samples` checks
:752/:851, hot-path logging) · `src/webgl/WebGLView.js` (`setPixelRatio`, cache-buster :112,
disposal, delta clamp) · `src/App.js` (canvas width authority :197, resize debounce) ·
`src/gui/GUIView.js` (competing width authority :744) · `src/emg/DecompositionController.js`
(hot-path logging :211) · `src/styles/_patient-mode.scss` (canvas width rule).

- **Per-frame auto-normalisation.** `EMGChannelView.js:437-443` and
  `DecompositionView.js:543-548` recompute min/max over the visible window every frame and
  normalise to it. A resting channel of µV noise is drawn at the same full-scale amplitude
  as a maximal contraction, and the gain visibly jumps as the window slides. **You cannot
  distinguish activation from rest by looking at the plot.** Needs fixed or slowly-adapting
  gain with a labelled µV axis.
- **No time axis, and the window length silently depends on the device.**
  `EMGClient.bufferSize = 2048` is commented "~1 second at 2048 Hz", but `sampleRate` arrives
  from the server (`:648`) and is never used to size the buffer. On a 10240 Hz Quattrocento
  the window is 0.2 s; at 512 Hz it is 4 s. Nothing on screen says which.
- **Nearest-neighbour decimation.** `EMGChannelView.js:445-458` takes one sample per pixel
  column, discarding ~60% of samples and aliasing a broadband signal. `samplesPerPixel` is
  computed at `:432` and never used — the envelope approach was clearly intended.
  `TriggerMonitor.js:105-115` does min/max envelope **correctly**; the throwaway debug widget
  is the one that gets it right. Copy it.
- **No drop/gap detection** despite the server providing the data — `decomposition_chunk`
  carries `total_samples` (`EMGClient.js:851`) and `emg_chunk` a timestamp (`:771`); neither
  is checked. A server-side drop renders as a seamless but discontinuous trace, and spike
  rasters silently desync from the waveform. Also no validation that `msg.n_samples` matches
  `data[ch].length` (`:752-758`) — a mismatch writes `NaN` into the `Float32Array` and
  permanently blanks the channel via the min/max normalisation.
- **`TriggerMonitor` draws synchronously on the socket callback** (`:84`) — ~48k reads plus a
  full repaint, ~20×/s, on the main thread, interleaved with the three.js loop. It is the
  only file in the layer ignoring the RAF pattern.
- **RAF decoupled from data.** Both views repaint at 60 fps against ~20 chunks/s — 3× wasted
  full-canvas repaints. Render on a dirty flag.
- **No `devicePixelRatio` handling anywhere**, including no `setPixelRatio` on the WebGL
  renderer. The 3D hand — the thing the subject looks at — renders at 1× and is soft on any
  HiDPI display. `EMGChannelView` uses a fixed 800×600 backing store stretched to fit.
- **Three authorities compute the canvas width and disagree**: `App.js:197-210` (hardcoded
  280/320, ignores the 400 px pane), `GUIView.js:744-746`, `_patient-mode.scss:87-92`. In
  config mode the drawing buffer is `innerWidth` while the container is `innerWidth - 400`,
  so the hand sits visibly off-centre and the camera aspect matches the buffer rather than
  the visible area. No resize debounce either.
- **Console logging in hot paths** — `DecompositionController.js:211-212` logs every
  classification (~20 Hz), `EMGClient.js:878` `JSON.stringify`s every status. DevTools retains
  every logged object graph, so a multi-hour session grows memory steadily. Gate behind a
  debug flag.
- **Three.js hygiene:** `?v=${Date.now()}` on every model URL (`WebGLView.js:112`) defeats all
  caching — with `loopSequence`, ~1.1 MB re-downloaded and re-parsed per loop; textures,
  skeletons and `mixer.uncacheRoot` are never disposed on swap; `this.action` survives model
  swaps pointing at a destroyed mixer; raw `clock.getDelta()` with no clamp means a GC pause
  can skip a whole movement while still counting the rep.

---

## Phase H — Guardrails

There is currently **no linter, no formatter, no test, no type checking, and no CI** over
~11,000 lines driving electrical stimulation.

**Files (phase):** `src/config/configStore.js` (validation :26) · `src/config/sequenceSchema.js`
(`falsyToDefault` :63/:195, already the source of truth) · `src/ui/SequenceConfigPanel.js`
(channel-id collision :560/:813, blob export :976) · `src/ui/SegmentManager.js` (blob export
:326) · `src/gui/GUIView.js` (`stats` :731, `file.path` :381/:511) · `src/index.js` +
`fend/public/data/manifest.json` (delete `async-preloader`) · `package.json` (pin `three`,
pin `tweakpane`, move `stats.js` to dev) · `vite.config.js` (`base`, `sourcemap`, `target`,
port type) · `fend/start_sim.bat` + `start.bat` + `start_quattrocento.bat` (serve built
output, open browser) · new: `.eslintrc` / prettier / a test runner + first specs.

- **Validate configs against the schema that already exists.** `configStore.normalizeConfig`
  (`:26-40`) only ensures containers exist; `loadSavedConfig` (`:46-55`) parses arbitrary
  `localStorage` JSON and hands it straight to the player at boot (`App.js:102`).
  `sequenceSchema.js` already declares types, defaults and min/max for every scalar, so a
  validator is nearly free. **Highest-consequence gap: `stimulation.channels[].amplitude`
  has `max: 130` mA in the schema (`sequenceSchema.js:166`) and nothing enforces it
  anywhere.** Clamp at the boundary, not in the UI.
- Fix the documented `falsyToDefault` bug (`sequenceSchema.js:63-67`, `:195`) — **you cannot
  currently set a rest of 0 seconds**, which is a legitimate value.
- Fix stim channel id collisions: `SequenceConfigPanel.js:560-561` uses
  `Math.min(maxId + 1, 8)`, so a 9th channel silently duplicates id 8. The per-item editor
  uses a different scheme entirely (`:813`), so global and per-item lists can disagree about
  what channel N means.
- **Unit tests for the pure parts first** — they are the highest-value and easiest:
  `sequenceSchema` validation, config derivation and the four transforms in
  `MovementSessionController` (`:541`, `:569`, `:867`, `:396`), segment maths, `_moveProb`
  and the stim gate hysteresis/latch. `MovementStimController._now()` is already overridable
  for tests, which suggests this was intended.
- Pin `three` **exactly** (`^0.181.2` — Three.js ships breaking changes in minors). Pin
  `tweakpane`. Move `stats.js` to devDependencies and gate its instantiation
  (`GUIView.js:731-734` runs it in production).
- **Delete `async-preloader`.** It exists solely to load `public/data/manifest.json`, which is
  `{"items": []}`. It contributes a network round trip, a `loadProgress` variable that is
  never displayed, and a `.catch` that swallows all boot failures into `console.log`
  (`index.js:27`) — a failed fetch gives the operator a permanently black screen. Either
  delete it or make it do real work (warming the ~4 MB of `.glb` before a session).
- **Stop running the Vite dev server in front of subjects.** `start_sim.bat:73` runs
  `npm run dev` and does not open a browser. Build and serve the built output.
- `vite.config.js`: no `base`, no `sourcemap` (so failures from a rig produce minified
  stack traces), no `build.target`, `port: '8080'` is a string.
- `GUIView.js:381,511` read `file.path` — an Electron-only property removed from browsers
  years ago. Both Browse buttons are non-functional; `:383` works around it by asking the
  operator to paste a full path. Decide whether this is an Electron app or not.
- Blob export is broken in some browsers: `SegmentManager.js:326-331` and
  `SequenceConfigPanel.js:976-981` never append the anchor and revoke the URL synchronously
  after `click()`.

---

## Notes on scope

**What was verified directly.** Everything in Phase 0 and Phase A, plus C1, C2, C3, the
`currentModelIndex` gap in D, the `onDisconnect` absence in B1, the styling/ownership counts
in D, and the asset hashes in E3 were read in the source during review. The remainder comes
from a deeper pass and should be re-confirmed at fix time — one claim from that pass was
already found overstated (see the scope note under A4).

**Line numbers will drift.** They are anchors for the first reader, not a contract.

**Do not batch phases D–H into one branch.** D changes ownership of state that E–H all read.
Land D, verify a full session end-to-end, then continue.

**The two files worth reading before starting** are `config/movements.js` and
`config/sequenceSchema.js`. Both document the exact bug they were written to eliminate, and
`sequenceSchema.js:63-67` documents one it still has. That is the standard the rest of this
module should be brought to — not the code style, the honesty about failure modes.
