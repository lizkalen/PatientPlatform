"""Decider: fuse the two heads and drive a move/rest stimulation gate.

Per window it smooths each head's probabilities with an EMA, forms the final label
(rest if the gate says rest, otherwise the gesture argmax), and produces a stim
command from p_move = P(move):

  * START when at rest and p_move >= on_thr
  * STOP  when stimming and p_move < off_thr and the onset latch has elapsed
  * HOLD  otherwise

on_thr >= off_thr gives hysteresis around the boundary; the latch holds a freshly
started train on for at least `latch_ms` so a brief early rest cannot stop it. This is
the control logic from MovementStimController.js, as a pure policy: it returns the
command and does not talk to a stimulator. Time is passed in (`t_ms`), so it works off
sample time offline and wall-clock time live.

`update` is the stim-only path. `update_routed` adds the clean (no-stim) head: it picks
the head that owns each window from the stim flag and keeps a third EMA for the clean
head. Both return the same event shape.

numpy only.
"""
from __future__ import annotations

import numpy as np

from .config import DecoderConfig

START, STOP, HOLD = "START", "STOP", "HOLD"


class Decider:
    def __init__(self, cfg: DecoderConfig):
        self.cfg = cfg
        self.n_move = cfg.rest_idx
        self.enabled = True
        self.reset()

    def reset(self):
        """Clear smoothing and gate state for a new recording/stream."""
        self.gate_ema = self._gate_prior()
        self.gest_ema = self._gest_prior()
        self.clean_ema = self._clean_prior()                 # routed path only
        self.prev_stim = None                                # regime of the last window
        self.stimming = False
        self.stim_started_ms = 0.0

    def _gate_prior(self):
        return np.array([0.5, 0.5])                          # [P(move), P(rest)]

    def _gest_prior(self):
        return np.full(self.n_move, 1.0 / self.n_move)

    def _clean_prior(self):
        """Rest-neutral prior for the clean head: half the mass on rest, the other half
        split evenly over the move classes.

        NOT the flat 1/n_classes prior. The clean head drives the gate through
        p_move = 1 - P(rest), so a flat prior over 4 classes would start each clean run
        at p_move = 0.75 and ask for START before seeing any evidence. Half on rest
        starts it at 0.5, matching the gate's own [0.5, 0.5] prior, and the even split
        over the move classes keeps the prior neutral between gestures."""
        p = np.full(self.cfg.n_classes, 0.5 / self.n_move)
        p[self.cfg.rest_idx] = 0.5
        return p

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False
        self.stimming = False

    def emergency_stop(self):
        """Stop stimulating and stop reacting until re-enabled."""
        self.enabled = False
        self.stimming = False

    def update(self, gate_proba, gest_proba, t_ms, new_run=False) -> dict:
        """Advance the decider by one window.

        gate_proba : [P(move), P(rest)]
        gest_proba : probabilities over move classes 0..n_move-1
        t_ms       : current time in ms (sample time offline, wall-clock live)
        new_run    : True if this window starts a new contiguous run
        """
        a = self.cfg.ema_alpha
        gate_proba = np.asarray(gate_proba, float)
        gest_proba = np.asarray(gest_proba, float)

        if new_run and self.cfg.ema_reset:
            self.gate_ema = gate_proba.copy()
            self.gest_ema = gest_proba.copy()
        else:
            self.gate_ema = a * gate_proba + (1 - a) * self.gate_ema
            self.gest_ema = a * gest_proba + (1 - a) * self.gest_ema

        is_rest = bool(self.gate_ema.argmax() == 1)
        p_move = float(self.gate_ema[0])
        final = self.cfg.rest_idx if is_rest else int(self.gest_ema.argmax())
        return self._event(final, is_rest, p_move, t_ms, "stim")

    def _command(self, p_move, t_ms) -> str:
        """START/STOP/HOLD from p_move, with hysteresis and the onset latch."""
        if not self.enabled:
            return HOLD
        if not self.stimming and p_move >= self.cfg.on_thr:
            self.stimming = True
            self.stim_started_ms = t_ms
            return START
        if (self.stimming and p_move < self.cfg.off_thr
                and (t_ms - self.stim_started_ms) >= self.cfg.latch_ms):
            self.stimming = False
            return STOP
        return HOLD

    def _event(self, final, is_rest, p_move, t_ms, head) -> dict:
        """The decision dict, after running the stim command policy on p_move."""
        cmd = self._command(p_move, t_ms)
        return dict(final=int(final), label=self.cfg.class_names[int(final)],
                    p_move=float(p_move), is_rest=bool(is_rest), head=head,
                    gate_ema=self.gate_ema.copy(), gest_ema=self.gest_ema.copy(),
                    clean_ema=self.clean_ema.copy(), stim_cmd=cmd, stimming=self.stimming)

    def update_routed(self, stim_on, t_ms, gate_proba=None, gest_proba=None,
                      clean_proba=None, new_run=False) -> dict:
        """Advance by one window with the head that owns its regime.

        stim_on True  -> the gate and gesture heads decide, exactly as `update`.
        stim_on False -> the clean head decides all classes on its own, and p_move is
                         1 - P(rest) from its smoothed probabilities.

        The stim command policy runs in both regimes on purpose: before a train starts
        there is no stimulation, so the clean head is the head that must see the onset
        and ask for START. With cfg.ema_regime_reset the EMA of the head that takes over
        restarts from its uniform prior, so it does not inherit stale smoothing from the
        regime it was idle through."""
        stim_on = bool(stim_on)
        fresh = self.cfg.ema_regime_reset and self.prev_stim is not stim_on
        self.prev_stim = stim_on

        if stim_on:
            if fresh:
                self.gate_ema = self._gate_prior()
                self.gest_ema = self._gest_prior()
            return self.update(gate_proba, gest_proba, t_ms, new_run)

        if fresh:
            self.clean_ema = self._clean_prior()
        a = self.cfg.ema_alpha
        clean_proba = np.asarray(clean_proba, float)
        if new_run and self.cfg.ema_reset:
            self.clean_ema = clean_proba.copy()
        else:
            self.clean_ema = a * clean_proba + (1 - a) * self.clean_ema
        rest = self.cfg.rest_idx
        final = int(self.clean_ema.argmax())
        return self._event(final, final == rest, 1.0 - self.clean_ema[rest], t_ms, "clean")
