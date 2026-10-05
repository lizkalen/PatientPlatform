"""Movement Classifier Manager
==============================

Live-stream counterpart of DecompositionManager, for movement discrimination
(close / trp / ext / rest) on stimulated HD-EMG. Loads a frozen model artifact
produced by online_sim (`--save-model`), then turns each incoming RAW EMG chunk
into a movement decision via the shared OnlineMovementClassifier engine.

Used by RippleWebSocketServer exactly like DecompositionManager:
    1. instantiate
    2. load_model(path, sample_rate, n_channels) -> activates
    3. process(raw_samples) -> decision dict per chunk (or None if no new window)
    4. unload_model() -> deactivates

IMPORTANT — feed it the RAW chunk, not the server-filtered one. The engine does
its OWN notch -> stim-blank -> bandpass, and needs the trigger row intact, so it
must see the unfiltered (channels, samples) chunk with EMG rows 0..191 and the
trigger at `trig_ch` (192) — the same channel layout the model was trained on.
"""
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from server.movement_engine import OnlineMovementClassifier, load_artifact, RLABS


class MovementClassifierManager:
    """Manages the live movement-classification engine + its model lifecycle."""

    def __init__(self):
        self.active: bool = False
        self.model_path: Optional[str] = None
        self.engine: Optional[OnlineMovementClassifier] = None
        self.class_order = RLABS
        self.n_classes: int = 0
        self.sample_rate: float = 0.0
        self.total_samples: int = 0
        self.last_decision: Optional[dict] = None
        self.route_clean: int = 0            # windows routed to the clean/initiator model
        self.route_stim: int = 0             # windows routed to the stim model
        self.stim_binary: bool = False       # STIM head reports a move/rest gate (vs 4-class)

    # -------------------------------------------------------------------------
    # MODEL LIFECYCLE
    # -------------------------------------------------------------------------
    def load_model(self, model_path: str, sample_rate: float, n_channels: int) -> dict:
        """Load a frozen movement-model artifact and build the live engine.

        Raises FileNotFoundError / KeyError on a bad artifact, or ValueError if
        the device's channel count can't supply the EMG rows + trigger the model
        expects.
        """
        art = load_artifact(model_path)
        eng = OnlineMovementClassifier.from_artifact(art)

        need = max(int(eng.gidx.max()), int(eng.trig_ch)) + 1
        if n_channels < need:
            raise ValueError(
                f"model needs {need} channels (EMG rows up to {int(eng.gidx.max())} "
                f"+ trigger at {eng.trig_ch}) but the device streams {n_channels}. "
                f"Check the OTBioLab+ configuration includes all EMG grids + trigger."
            )

        eng.reset()
        self.engine = eng
        self.model_path = model_path
        self.sample_rate = sample_rate
        self.class_order = list(eng.class_order or RLABS)
        self.n_classes = len(self.class_order)
        self.total_samples = 0
        self.last_decision = None
        self.route_clean = 0
        self.route_stim = 0
        self.stim_binary = getattr(eng, "stim_binary", False)
        self.active = True

        info = self.get_status()
        info["fixed_latency_ms"] = round(eng.HOLD / eng.fs * 1000, 1)
        two_stage = getattr(eng, "model_clean", None) is not None
        print(f"[MovementManager] Loaded {model_path}: {self.n_classes} classes "
              f"{self.class_order}, {eng.ng} good ch, win={eng.win_ms}ms "
              f"step={eng.step_ms}ms{' [TWO-STAGE: stim+clean specialists]' if two_stage else ''}")
        return info

    def unload_model(self):
        """Unload the model and reset all state."""
        self.active = False
        self.model_path = None
        self.engine = None
        self.n_classes = 0
        self.total_samples = 0
        self.last_decision = None
        print("[MovementManager] Model unloaded")

    def set_trig_mid(self, trig_mid: float):
        """Override the stim-trigger threshold (frozen from offline calibration).
        Useful if the live device's trigger amplitude differs from the recording."""
        if self.engine is not None:
            self.engine.trig_mid = float(trig_mid)

    def set_stim_binary(self, on: bool) -> dict:
        """Toggle the STIM head's live output between 4-class and a move/rest gate
        (1 - P(rest)). Runtime-safe; the clean head is unaffected. Returns status."""
        on = bool(on)
        if self.engine is not None:
            self.engine.set_stim_binary(on)
        self.stim_binary = on
        return self.get_status()

    # -------------------------------------------------------------------------
    # PER-CHUNK PROCESSING
    # -------------------------------------------------------------------------
    def process(self, raw_samples: np.ndarray) -> Optional[dict]:
        """Feed one RAW (channels, samples) chunk to the engine.

        Returns the latest movement decision finalised in this chunk, or None if
        no window boundary was crossed (0 decisions this chunk — the frontend just
        holds the previous state). Multiple windows may finalise per chunk; we
        report the most recent (current state) plus how many fired.
        """
        if not self.active or self.engine is None or raw_samples is None:
            return None
        n = raw_samples.shape[1]
        events = self.engine.process(raw_samples)
        self.total_samples += n
        # Track which specialist each window was routed to (stim=blanked window, else
        # clean/initiator) — for the live routing diagnostic.
        n_stim = sum(1 for e in events if e.get("stim"))
        self.route_stim += n_stim
        self.route_clean += len(events) - n_stim
        if not events:
            return None
        ev = events[-1]                      # most recent decision = current state
        pred = int(ev["pred"])
        probs = ev.get("sm", ev.get("probs"))    # smoothed class probabilities
        decision = {
            "pred": pred,
            "label": self.class_order[pred] if pred < len(self.class_order) else str(pred),
            "probs": np.asarray(probs, float).tolist(),
            "n_decisions": len(events),
            "stim": bool(ev.get("stim")),        # which model: True=stim, False=clean/initiator
            "route_clean": self.route_clean,     # cumulative window routing counts
            "route_stim": self.route_stim,
            "end_sample": int(ev["end"]),
            "n_samples": n,
            "total_samples": self.total_samples,
        }
        # Move/rest readout (1 - P(rest) on the smoothed belief). `binary` marks a
        # STIM-head window reported as a move/rest gate; the 4-class pred/probs above
        # still ride along for logging + actuation.
        if "p_move" in ev:
            decision["p_move"] = float(ev["p_move"])
        if ev.get("binary"):
            decision["binary"] = True
            decision["moving"] = decision["p_move"] >= 0.5
        self.last_decision = decision
        return decision

    def result_to_json(self, result: Optional[dict]) -> Optional[dict]:
        """Wrap a process() result as a broadcastable message."""
        if result is None:
            return None
        return {"type": "movement_decision", "class_order": self.class_order, **result}

    def get_status(self) -> dict:
        """Current movement-classification status for broadcasting. Keys are
        `movement_`-namespaced so this can be merged into the shared connected/
        status messages without clobbering the decomposition manager's keys."""
        status = {
            "movement_active": self.active,
            "movement_model_path": self.model_path,
            "movement_class_order": self.class_order,
            "movement_n_classes": self.n_classes,
            "movement_total_samples": self.total_samples,
            "movement_stim_binary": self.stim_binary,
        }
        if self.active and self.engine is not None:
            e = self.engine
            status["movement_two_stage"] = getattr(e, "model_clean", None) is not None
            status["movement_config"] = {
                "n_good_channels": int(e.ng),
                "trig_ch": int(e.trig_ch),
                "win_ms": e.win_ms,
                "step_ms": e.step_ms,
                "smooth": e.smooth,
            }
        return status
