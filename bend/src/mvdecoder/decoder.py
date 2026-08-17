"""The two-stage ExtraTrees decoder: a gate (move vs rest) and a gesture head.

The gate is trained on all stim windows with the label rest=(gt==rest_idx); the
gesture head is trained on the moving stim windows (gt < rest_idx). Both are
StandardScaler + ExtraTreesClassifier pipelines. Inference returns per-head class
probabilities aligned to a fixed label order.

An optional third head, the CLEAN head, decides all classes on its own for the windows
where stimulation is off. It reads the same gesture feature vector and comes from a
no-stim training recording. Fit it with `fit_clean_from_cache`; without it the decoder
stays stim-only and every path behaves as before.

sklearn is imported inside `fit` only, so importing this module does not require it;
inference uses the fitted estimators through predict_proba. `calibration` (trigger
threshold and PARRM period) is stored alongside the models for persistence.
"""
from __future__ import annotations

import pickle

import numpy as np

from .config import DecoderConfig
from .features import vectors_from_cache


class TwoStageDecoder:
    def __init__(self, cfg: DecoderConfig, good: np.ndarray):
        self.cfg = cfg
        self.good = np.asarray(good, int)
        self.gate_model = None
        self.gest_model = None
        self.clean_model = None            # optional no-stim head (see fit_clean)
        self.thr: float | None = None      # trigger threshold (calibration)
        self.P: float | None = None        # PARRM period (calibration)

    @property
    def has_clean(self) -> bool:
        """True once a clean (no-stim) head is fitted, which turns on routing."""
        return self.clean_model is not None

    # -- calibration -------------------------------------------------------------
    def set_calibration(self, thr, P):
        """Record the session's trigger threshold and PARRM period."""
        self.thr = None if thr is None else float(thr)
        self.P = float(P)
        return self

    # -- training ----------------------------------------------------------------
    def _pipeline(self):
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
        from sklearn.ensemble import ExtraTreesClassifier
        return make_pipeline(
            StandardScaler(),
            ExtraTreesClassifier(self.cfg.n_trees, random_state=self.cfg.random_state,
                                 n_jobs=1))

    def fit(self, Xgate, Xgest, gt, S=None, gate_mask=None, gest_mask=None):
        """Fit both heads. Xgate/Xgest are the per-window gate/gesture feature matrices,
        gt the class labels (rest = rest_idx).

        By default the gate trains on the stim windows S and the gesture on the moving
        stim windows (S & gt<rest) -- the experlangen setup, where rest is stimulated.
        Pass explicit `gate_mask` / `gest_mask` to override this, e.g. when rest is
        stim-off and the gate must train on all windows (move and rest)."""
        Xgate = np.asarray(Xgate, float)
        Xgest = np.asarray(Xgest, float)
        gt = np.asarray(gt, int)
        rest = self.cfg.rest_idx
        if gate_mask is None or gest_mask is None:
            if S is None:
                raise ValueError("pass S, or both gate_mask and gest_mask")
            S = np.asarray(S, bool)
        gm = np.asarray(gate_mask, bool) if gate_mask is not None else S
        sm = np.asarray(gest_mask, bool) if gest_mask is not None else (S & (gt < rest))
        self.gate_model = self._pipeline().fit(Xgate[gm], (gt[gm] == rest).astype(int))
        self.gest_model = self._pipeline().fit(Xgest[sm], gt[sm])
        return self

    def fit_from_cache(self, F, A, E, gt, S):
        """Fit from a prebuilt feat_v2 cache: assemble the gate/gesture vectors, then
        fit. Uses this decoder's good-channel indices and config."""
        Xgate, Xgest = vectors_from_cache(F, A, E, self.good, self.cfg)
        return self.fit(Xgate, Xgest, gt, S)

    def fit_clean(self, Xgest, gt, mask=None):
        """Fit the CLEAN (no-stim) head: one model over all classes, for the windows
        where stimulation is off. It reads the SAME gesture feature vector as the
        gesture head, so no extra features are computed at run time.

        Xgest : gesture feature matrix
        gt    : class labels (rest = rest_idx)
        mask  : which windows to train on; None uses all of them."""
        Xgest = np.asarray(Xgest, float)
        gt = np.asarray(gt, int)
        m = np.ones(len(gt), bool) if mask is None else np.asarray(mask, bool)
        self.clean_model = self._pipeline().fit(Xgest[m], gt[m])
        return self

    def fit_clean_from_cache(self, F, A, E, gt, S, mask=None):
        """Fit the clean head from a no-stim training cache.

        The default mask takes the moving windows that carry stim (S & gt < rest) plus
        the rest windows that do not (~S & gt == rest). In the no-stim training block the
        trigger still runs during each movement, so its movement windows are stim-flagged
        and artifact-cleaned, while rest sits in the untriggered gaps. Pass `mask` to
        override this for a recording labelled another way."""
        _, Xgest = vectors_from_cache(F, A, E, self.good, self.cfg)
        gt = np.asarray(gt, int)
        if mask is None:
            S = np.asarray(S, bool)
            rest = self.cfg.rest_idx
            mask = (S & (gt < rest)) | (~S & (gt == rest))
        return self.fit_clean(Xgest, gt, mask)

    # -- inference ---------------------------------------------------------------
    @staticmethod
    def _align(proba_row, classes, n):
        out = np.zeros(n)
        for j, c in enumerate(classes):
            out[int(c)] = proba_row[j]
        return out

    def gate_proba(self, x) -> np.ndarray:
        """[P(move), P(rest)] for one gate feature row."""
        pr = self.gate_model.predict_proba(np.atleast_2d(x))[0]
        return self._align(pr, self.gate_model.classes_, 2)

    def gesture_proba(self, x) -> np.ndarray:
        """Probabilities over the move classes (0..rest_idx-1) for one gesture row."""
        pr = self.gest_model.predict_proba(np.atleast_2d(x))[0]
        return self._align(pr, self.gest_model.classes_, self.cfg.rest_idx)

    def clean_proba(self, x) -> np.ndarray:
        """Probabilities over all classes (0..rest_idx) from the clean head, for one
        gesture feature row."""
        pr = self.clean_model.predict_proba(np.atleast_2d(x))[0]
        return self._align(pr, self.clean_model.classes_, self.cfg.n_classes)

    def gate_proba_batch(self, Xgate) -> np.ndarray:
        pr = self.gate_model.predict_proba(np.asarray(Xgate, float))
        out = np.zeros((len(pr), 2))
        for j, c in enumerate(self.gate_model.classes_):
            out[:, int(c)] = pr[:, j]
        return out

    def gesture_proba_batch(self, Xgest) -> np.ndarray:
        pr = self.gest_model.predict_proba(np.asarray(Xgest, float))
        out = np.zeros((len(pr), self.cfg.rest_idx))
        for j, c in enumerate(self.gest_model.classes_):
            out[:, int(c)] = pr[:, j]
        return out

    def clean_proba_batch(self, Xgest) -> np.ndarray:
        pr = self.clean_model.predict_proba(np.asarray(Xgest, float))
        out = np.zeros((len(pr), self.cfg.n_classes))
        for j, c in enumerate(self.clean_model.classes_):
            out[:, int(c)] = pr[:, j]
        return out

    # -- persistence -------------------------------------------------------------
    def save(self, path, meta=None):
        art = dict(cfg=self.cfg, good=self.good, thr=self.thr, P=self.P,
                   gate_model=self.gate_model, gest_model=self.gest_model,
                   clean_model=self.clean_model, meta=meta or {})
        with open(path, "wb") as f:
            pickle.dump(art, f)
        return path

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            art = pickle.load(f)
        dec = cls(art["cfg"], art["good"])
        dec.thr, dec.P = art["thr"], art["P"]
        dec.gate_model, dec.gest_model = art["gate_model"], art["gest_model"]
        dec.clean_model = art.get("clean_model")        # absent in stim-only artifacts
        return dec
