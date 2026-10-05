"""
Low-level primitives for the MUAP-based EMG decomposition of
Chen, Li & Xia (2025), "A motor unit action potential-based method for
surface electromyography decomposition", J. NeuroEng. Rehabil. 22:60.

These functions parallel the existing ``core`` module and implement the
pieces that the convolution-kernel-compensation (CKC) MUAP pipeline needs on
top of it:

    * ``compute_cov_inverse``      -- C_x̄x̄^{-1}              (Eq. 3 denominator)
    * ``gckc``                     -- gradient CKC stage-1 decomposition (Eq. 6)
    * ``spike_triggered_average``  -- MUAP estimation by STA   (Eq. 7)
    * ``least_squares_muap``       -- MUAP estimation by LS    (Eq. 8)
    * ``pca_smooth_muap``          -- optional MUAP denoising  (>=90 % variance)
    * ``reconstruct_mu_filter``    -- MUAP -> MU filter        (step 5 of Algorithm 1)
    * ``lmmse_source``             -- apply a MU filter        (Eq. 3)

Conventions
-----------
* ``sig``   : raw / band-passed EMG, shape (n_channels, n_samples).
* ``x_ext`` : extended EMG produced by ``core.extension(sig, K)``, shape
  (n_channels * K, n_samples). The paper does NOT whiten before decomposition,
  so everything here works in the *extended-but-not-whitened* domain.
* ``Cx``    : the precomputed product  C_x̄x̄^{-1} @ x_ext, shape
  (n_channels * K, n_samples). Computing it once lets the source for any MU
  filter ``c`` be obtained as a single matmul:  source = c @ Cx   (= cᵀ Cinv x̄).
"""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks, fftconvolve
from scipy.ndimage import shift as _ndshift

from .core import est_spike_times, peel_off as _peel_off
from ..evaluation.evaluate import get_bin_spikes, max_xcorr, match_spike_trains

_EPS = 1e-12


def _scatter_add(kernel: np.ndarray, spikes, n_samp: int) -> np.ndarray:
    """Add a short ``(n_feat, Nw)`` kernel, centred on each spike, into an
    ``(n_feat, n_samp)`` zero array.

    The centring reproduces ``core.peel_off`` exactly: with ``Nw = 2*width+1``
    the kernel spans ``[s-width, s+width]`` (sample ``s-width`` <-> kernel index
    0), so ``_scatter_add(waveform, sp, L)`` equals that function's ``comp_sig``
    for interior spikes (and ``_scatter_add(Cinv @ waveform, sp, L)`` therefore
    equals ``Cinv @ comp_sig`` -- the deflation cancels in time while the inverse
    acts only on the feature axis).
    """
    kernel = np.asarray(kernel, float)
    n_feat, Nw = kernel.shape
    half = Nw // 2
    out = np.zeros((n_feat, n_samp))
    for s in np.asarray(spikes, dtype=int):
        a = int(s) - half
        lo = max(0, a)
        hi = min(n_samp, a + Nw)
        if hi <= lo:
            continue
        out[:, lo:hi] += kernel[:, lo - a:hi - a]
    return out


def _best_duplicate_match(spk, accepted, n_samp, fsamp,
                          max_shift=0.1, tol=0.001):
    """Best-matching accepted train for ``spk``: ``(best_idx, best_score)``.

    Uses the *same* binary-train matching as ``core.remove_duplicates`` -- best
    cross-correlation lag (``max_xcorr``) then tolerance-masked true positives
    (``match_spike_trains``), scored as ``tp / max(len_i, len_j)`` -- so the
    training-side gate and the online dedup agree on what "duplicate" means.
    ``best_idx == -1`` when there is no overlap with any accepted train (or
    ``accepted``/``spk`` is empty); the caller compares ``best_score`` against
    its duplicate threshold.
    """
    spk = np.asarray(spk, dtype=int)
    best_idx, best_score = -1, 0.0
    if spk.size == 0:
        return best_idx, best_score
    st1 = get_bin_spikes(spk, n_samp)
    msh = int(max_shift * fsamp)
    for u, prev in enumerate(accepted):
        prev = np.asarray(prev, dtype=int)
        if prev.size == 0:
            continue
        st2 = get_bin_spikes(prev, n_samp)
        _, shift = max_xcorr(st1, st2, max_shift=msh)
        tp, _, _ = match_spike_trains(st1, st2, shift=shift, tol=tol, fsamp=fsamp)
        denom = max(spk.size, prev.size)
        score = tp / denom if denom > 0 else 0.0
        if score > best_score:
            best_score, best_idx = score, u
    return best_idx, best_score


# --------------------------------------------------------------------------- #
# Covariance / LMMSE                                                          #
# --------------------------------------------------------------------------- #
def compute_cov_inverse(x_ext: np.ndarray, reg="auto", rank_trunc=None) -> np.ndarray:
    """Inverse of the (extended) EMG covariance matrix C_x̄x̄ = E(x̄ x̄ᵀ).

    A small ridge term is added for numerical stability, mirroring the
    regularised inversion used throughout CKC implementations.

    ``rank_trunc`` (float in (0,1) or None): if set, the covariance is inverted on
    a RANK-TRUNCATED basis -- eigen-components with eigenvalue <= rank_trunc * max
    are dropped (pseudo-inverse) instead of amplified by the tiny ridge. This is the
    fix for the near-singular extended-EMG covariance (see mu_xclean/diagnosis): the
    default ridge is ~1e-8, far below the noise floor, so the inverse amplifies the
    noise subspace and the LMMSE sources become noise. A truncation of ~1e-3 keeps
    only the signal subspace. None (default) preserves the original behavior.
    """
    n_feat, n_samp = x_ext.shape
    cov = (x_ext @ x_ext.T) / n_samp
    if rank_trunc is not None:
        w, V = np.linalg.eigh(cov)
        keep = w > rank_trunc * w.max()
        w_inv = np.where(keep, 1.0 / np.where(keep, w, 1.0), 0.0)
        return (V * w_inv) @ V.T
    if reg == "auto":
        lam = 1e-6 * np.trace(cov) / n_feat
    else:
        lam = float(reg)
    return np.linalg.inv(cov + lam * np.eye(n_feat))


def lmmse_source(mu_filter: np.ndarray, Cx: np.ndarray) -> np.ndarray:
    """LMMSE pulse-train estimate  θ̂ = c_θx̄ᵀ C_x̄x̄^{-1} x̄  (Eq. 3).

    ``Cx`` must equal ``C_x̄x̄^{-1} @ x_ext`` so this is a single dot product.
    """
    return mu_filter @ Cx


def _cov_isi(spikes, fsamp) -> float:
    """Coefficient of variation of the inter-spike interval."""
    spikes = np.asarray(spikes)
    if spikes.size < 3:
        return np.inf
    isi = np.diff(spikes / fsamp)
    m = np.mean(isi)
    return np.std(isi) / m if m > 0 else np.inf


def _spike_change(prev, cur, tol_samples: int = 1) -> float:
    """Relative change between two spike trains (1 - rate of agreement).

    Used as the iteration-stopping criterion (paper: "change in the estimated
    MUST is less than 1 % for two consecutive iterations").
    """
    prev = np.asarray(prev)
    cur = np.asarray(cur)
    if prev.size == 0 and cur.size == 0:
        return 0.0
    if prev.size == 0 or cur.size == 0:
        return 1.0
    matched = 0
    j = 0
    sp = np.sort(prev)
    sc = np.sort(cur)
    for s in sc:
        while j < sp.size and sp[j] < s - tol_samples:
            j += 1
        if j < sp.size and abs(sp[j] - s) <= tol_samples:
            matched += 1
            j += 1
    roa = 2.0 * matched / (prev.size + cur.size)
    return 1.0 - roa


# --------------------------------------------------------------------------- #
# Stage 1: gradient CKC decomposition                                         #
# --------------------------------------------------------------------------- #
def _dfdtheta(theta: np.ndarray, contrast: str = "square") -> np.ndarray:
    """Derivative ∂f/∂θ of the contrast function used by gradient CKC.

    The paper requires an even, peak-reinforcing nonlinearity. Chen et al.
    (2025) use ∂f/∂θ = θ² (i.e. f = θ³/3, the same choice Holobar & Zazula
    used in their gradient-CKC experiments); ∂f/∂θ = |θ| is the other option
    discussed by Holobar.
    """
    if contrast == "square":      # f = θ³/3  ->  ∂f/∂θ = θ²
        return theta ** 2
    if contrast == "abs":         # ∂f/∂θ = |θ|
        return np.abs(theta)
    raise ValueError(f"Unknown contrast: {contrast}")


def _skew_sign(x: np.ndarray) -> float:
    """Sign-bearing skewness; used to orient θ̂ so discharges are positive peaks.

    The pulse-train amplitude/sign is ambiguous (Holobar & Zazula 2007). Both
    the t³ and t·|t| contrasts reward *positive* peaks, so before refining we
    flip the filter when the source is negatively skewed.
    """
    m = x.mean()
    s = x.std() + _EPS
    return float(np.mean(((x - m) / s) ** 3))


def _contrast_value(source: np.ndarray, contrast: str) -> float:
    """Cost F(θ̂) = Σ_n f(θ̂(n)) whose derivative is ``_dfdtheta`` (Eq. 9/10).

    * ``square``: f = θ³/3  -> use Σ θ³           (∂f/∂θ = θ²)
    * ``abs``   : f = θ|θ|/2 -> use Σ θ|θ|        (∂f/∂θ = |θ|)

    Returned up to a positive constant; only its *increase* matters for the
    line search, so the constant is dropped.
    """
    if contrast == "square":
        return float(np.sum(source ** 3))
    if contrast == "abs":
        return float(np.sum(source * np.abs(source)))
    raise ValueError(f"Unknown contrast: {contrast}")


def _line_search(c, grad, Cx, contrast, base_eta, max_backtracks):
    """Bisection (step-halving) line search of the learning rate η.

    Holobar & Zazula (ICA 2007) "adjust η(k) by bisection" each iteration. With
    a unit-norm ``grad`` and unit-norm ``c`` the step η is a pure rotation
    magnitude, so we try η ∈ {base_eta, base_eta/2, base_eta/4, ...} and accept
    the largest step that increases the contrast F(θ̂). If none does, the
    (renormalised) input filter is returned and ``improved=False`` signals the
    caller that the source has converged.

    Each trial costs one ``c @ Cx`` matmul; ``max_backtracks`` bounds that cost.
    """
    c_unit = c / (np.linalg.norm(c) + _EPS)
    f0 = _contrast_value(c_unit @ Cx, contrast)
    eta = base_eta
    for _ in range(max_backtracks):
        c_try = c + eta * grad
        c_try /= np.linalg.norm(c_try) + _EPS
        s_try = c_try @ Cx
        if _contrast_value(s_try, contrast) > f0:
            return c_try, s_try, True
        eta *= 0.5
    return c_unit, c_unit @ Cx, False


def gckc(
    x_ext: np.ndarray,
    Cx: np.ndarray,
    fsamp: float,
    n_mu: int = 100,
    max_iter: int = 45,
    tol: float = 0.01,
    sil_th: float = 0.9,
    cov_th: float = 0.35,
    min_firing_rate: float = 0.5,
    min_spikes: int = 5,
    base_eta: float = 1.0,
    max_backtracks: int = 8,
    contrast: str = "square",
    noise_floor_frac: float = 0.01,
    max_consec_rejects: int = 20,
    cluster: str = "kmeans",
    opt_init: str = "activity_idx",
    peel_off: bool = False,
    peel_win: float = 0.04,
    recur_peel_th: float = float("inf"),
    peel_recompute_every: int = 0,
    dedup: bool = True,
    dedup_th: float = 0.3,
    dedup_max_shift: float = 0.1,
    dedup_tol: float = 0.001,
    Cinv: np.ndarray | None = None,
    verbose: bool = False,
):
    """Decompose extended EMG into MUSTs with the *gradient* CKC algorithm.

    This is the stage-1 decomposition the MUAP pipeline averages to obtain
    MUAPs (steps 2-3 of Algorithm 1). It follows the classic CKC of Holobar &
    Zazula (2007, IEEE T-SP, Fig. 1) and its gradient extension (2007, ICA),
    which is what Chen et al. (2025) build on:

      1. The activity index γ(n) = x̄(n)ᵀ C_x̄x̄^{-1} x̄(n) indexes global pulse
         activity. A seed n1 is chosen according to ``opt_init``:
           * ``'activity_idx'`` (default, paper): n1 = argmax γ over the
             not-yet-deflated γ.
           * ``'biased_random'``: n1 is *sampled* with probability ∝ γ over the
             not-yet-deflated samples -- stochastic, but still steered away from
             already-extracted units, reusing the step-4 deflation.
           * ``'random'``: each search starts from a random unit vector
             (mirroring CBSS); γ is then used only for the step-4 deflation and
             the loop runs the full ``n_mu`` iterations.
      2. The cross-correlation vector is initialised as ĉ_θx̄ = x̄(n1) (Eq. 5)
         (for ``'random'`` it is the random vector), and θ̂ is oriented so
         discharges are positive peaks.

      Optionally (``peel_off=True``, requires ``Cinv``) the contribution of each
      ACCEPTED train is subtracted from x̄ (STA-based, ``peel_win`` s window) and
      C_x̄x̄^{-1}x̄ / γ are refreshed, so a re-seed cannot land on a unit already
      pulled out. This costs one ``Cinv @ x̄`` matmul per accepted MU.

      Independently, ``recur_peel_th`` guards against *dominating attractors*:
      short non-MU-like peaks (1-2 discharges) that many independent searches
      keep re-converging onto -- e.g. a periodic stim/movement artifact left in
      x̄. Each rejected discharge is bucketed by the ±5 ms guard; once a bucket
      has re-attracted the search ``recur_peel_th`` times it is peeled from x̄
      (only that location, same STA mechanism) and the source/γ refreshed, so the
      search stops wasting iterations on it. ``inf`` (default) disables this; it
      requires ``Cinv``. Only short trains are eligible, so a real multi-spike MU
      is never removed this way.
      3. ĉ is refined by the natural-gradient update (Holobar Eq. 10):
             ĉ ← ĉ + η · Σ_n (∂f/∂θ)(θ̂(n)) · x̄(n),   θ̂ = ĉᵀ C_x̄x̄^{-1} x̄,
         with ∂f/∂θ = θ² (the peak-reinforcing nonlinearity) and η chosen by a
         bisection (step-halving) line search on the contrast F each iteration.
      4. DEFLATION (Fig. 1 step 8): the activity index is zeroed at the
         discharge instants of *every* reconstructed train -- accepted or not --
         so the search advances and never re-seeds a unit already pulled out.
         A short guard window around each discharge prevents re-seeding an
         adjacent sample of the same firing. The loop repeats until γ falls to
         the noise floor (step 9). No Gram-Schmidt / separation-vector
         orthogonalisation is used -- duplicate trains are reconciled afterwards
         by the post-processing spike-train comparison (Fig. 1 step 7).

    The natural-gradient metric G = C_x̄x̄^{-1} cancels the inverse covariance
    when the update is written in the cross-correlation domain, so the update
    term is the plain weighted observation sum ``x_ext @ (∂f/∂θ)`` with no extra
    C_x̄x̄^{-1}. ``c`` and ``grad`` are renormalised so ``base_eta`` is a pure
    rotation magnitude that behaves consistently across recordings; the actual
    step is then found by halving ``base_eta`` until the contrast increases.

    Returns
    -------
    spikes_list : list[np.ndarray]   accepted MU discharge sample indices
    sil_list    : list[float]        pseudo-silhouette per accepted MU
    cov_list    : list[float]        ISI CoV per accepted MU
    filt_list   : list[np.ndarray]   the (iteratively estimated) MU filters
    """
    n_feat, n_samp = x_ext.shape

    if opt_init not in ("activity_idx", "random", "biased_random"):
        raise ValueError(f"Unknown opt_init: {opt_init!r} (expected "
                         "'activity_idx', 'biased_random' or 'random')")
    if peel_off and Cinv is None:
        raise ValueError("peel_off=True requires Cinv (C_x̄x̄^{-1}) to refresh "
                         "the source/activity index after each peel")
    if np.isfinite(recur_peel_th) and Cinv is None:
        raise ValueError("recur_peel_th requires Cinv (C_x̄x̄^{-1}) to refresh "
                         "the source/activity index after peeling an attractor")

    # Activity index  γ(n) = x̄(n)ᵀ C_x̄x̄^{-1} x̄(n)
    gamma = np.sum(x_ext * Cx, axis=0)
    work = gamma.astype(float).copy()          # γ minus deflation (step 8)
    deflate_mask = np.zeros(n_samp, dtype=bool)  # persistent -inf positions

    # Stop criterion (Fig. 1 step 9, "until γ is exhausted"), made scale-free:
    # a small fraction of the *initial peak* activity. A median/MAD baseline is
    # deliberately NOT used -- it assumes sparse discharges over a quiet
    # baseline, which fails for densely firing / gap-free signals where almost
    # every sample is active. The practical terminators are this floor plus
    # ``max_consec_rejects`` and ``n_mu``.
    gmax0 = float(np.max(work)) if np.isfinite(work).any() else 0.0
    noise_floor = noise_floor_frac * gmax0
    guard = max(1, int(round(0.005 * fsamp)))  # ±5 ms refractory guard

    spikes_list, sil_list, cov_list, filt_list = [], [], [], []
    consec_rejects = 0
    peels_since_full = 0                        # for the periodic Cx re-sync
    width_peel = int(peel_win * fsamp)

    def _apply_peel(x_ext, Cx, gamma, peels_since_full, peel_spikes):
        """Subtract the STA reconstruction of ``peel_spikes`` from x̄ and update
        Cx / γ incrementally.

        Instead of recomputing ``Cx = Cinv @ x̄`` (O(n_feat²·L)) after the peel,
        we exploit its linearity:  ΔCx = Cinv @ comp = scatter(Cinv @ waveform),
        i.e. one n_feat²·Nw matmul plus a sparse scatter (Nw = 2·win·fs+1 ≪ L).
        γ changes only inside the peeled windows, so only those columns are
        refreshed. ``peel_recompute_every`` triggers an exact full recompute
        every N peels to bound floating-point drift (0 = never).
        """
        peel_spikes = np.asarray(peel_spikes, dtype=int)
        x_ext, _comp, waveform = _peel_off(x_ext, peel_spikes,
                                           win=peel_win, fsamp=fsamp)
        sp_in = peel_spikes[(peel_spikes >= width_peel + 1)
                            & (peel_spikes < n_samp - width_peel - 1)]
        if sp_in.size:
            Cx = Cx - _scatter_add(Cinv @ waveform, sp_in, n_samp)
            cols = np.zeros(n_samp, dtype=bool)
            for s in sp_in:
                cols[max(0, int(s) - width_peel):
                     min(n_samp, int(s) + width_peel + 1)] = True
            gamma[cols] = np.sum(x_ext[:, cols] * Cx[:, cols], axis=0)
        peels_since_full += 1
        if peel_recompute_every and peels_since_full >= peel_recompute_every:
            Cx = Cinv @ x_ext
            gamma = np.sum(x_ext * Cx, axis=0)
            peels_since_full = 0
        return x_ext, Cx, gamma, peels_since_full

    # Dominating-attractor registry: how often each ±5 ms bucket has re-attracted
    # a (short, non-MU-like) rejected search. Buckets crossing ``recur_peel_th``
    # are peeled from x̄ once so the search stops re-converging onto them.
    bucket_w = 2 * guard + 1
    attractor_hits: dict[int, int] = {}
    peeled_buckets: set[int] = set()

    def _seed_index():
        """Pick a seed sample from the (deflated) activity index, or None when
        γ is exhausted. ``argmax`` for 'activity_idx', γ-weighted draw for
        'biased_random'."""
        if not np.isfinite(work).any() or float(np.max(work)) <= noise_floor:
            return None
        if opt_init == "activity_idx":
            return int(np.argmax(work))
        p = np.clip(np.where(np.isfinite(work), work, 0.0), 0.0, None)
        total = p.sum()
        if total <= 0:
            return None
        return int(np.random.choice(n_samp, p=p / total))

    if verbose:
        print(f"[gCKC] target MUs={n_mu}, contrast={contrast}, base_eta={base_eta}, "
              f"sil_th={sil_th}, cov_th={cov_th}, noise_floor={noise_floor:.3g}, "
              f"init={opt_init}, peel_off={peel_off}")

    for k in range(n_mu):
        # --- seed the MU search --------------------------------------------
        if opt_init == "random":
            # random unit vector; the activity index is then used only for the
            # discharge-based deflation below, not for choosing the seed.
            idx = None
            c = np.random.randn(n_feat).astype(float)
        else:
            # activity-index seed:  ĉ_θx̄ = x̄(n1)  (Eq. 5)
            idx = _seed_index()
            if idx is None:
                if verbose:
                    print(f"[gCKC] activity index exhausted "
                          f"(max={np.max(work):.3g} <= floor={noise_floor:.3g}); "
                          f"stopping")
                break
            c = x_ext[:, idx].astype(float).copy()
        c /= np.linalg.norm(c) + _EPS
        source = lmmse_source(c, Cx)           # θ̂ = ĉᵀ C_x̄x̄^{-1} x̄  (Eq. 3)
        if _skew_sign(source) < 0:             # orient discharges positive
            c, source = -c, -source
        spikes, sil, _ = est_spike_times(source, fsamp, cluster=cluster)

        # --- gradient refinement, η by bisection line search (Eq. 10) ------
        stable = 0
        for _ in range(max_iter):
            prev = spikes
            s = source / (np.max(np.abs(source)) + _EPS)   # scale-free θ̂
            grad = x_ext @ _dfdtheta(s, contrast)          # Σ_n (∂f/∂θ) x̄(n)
            grad /= np.linalg.norm(grad) + _EPS            # unit direction
            c, source, improved = _line_search(
                c, grad, Cx, contrast, base_eta, max_backtracks)
            if not improved:                               # converged
                break
            spikes, sil, _ = est_spike_times(source, fsamp, cluster=cluster)
            if len(spikes) == 0:
                break
            if _spike_change(prev, spikes) < tol:
                stable += 1
                if stable >= 2:    # <1% change for two consecutive iterations
                    break
            else:
                stable = 0

        # --- quality gating (practical filter on top of the paper) ---------
        cov = _cov_isi(spikes, fsamp)
        fr = len(spikes) / (n_samp / fsamp) if len(spikes) else 0.0
        accept = (
            len(spikes) >= min_spikes
            and sil > sil_th
            and cov < cov_th
            and fr >= min_firing_rate
        )
        # --- uniqueness gate (training-side dedup) -------------------------
        # A quality-passing source that duplicates an already-accepted train
        # (best-lag spike agreement >= dedup_th) is rejected here, so the
        # reconstructed filter bank stays non-redundant and the online
        # remove_duplicates step has little left to do. Doing it incrementally
        # (one vs the few accepted) avoids the O(n_units²) post-hoc pass.
        is_dup = False
        gate_checked = bool(accept and dedup and len(spikes_list) > 0)
        if gate_checked:
            dup_idx, dup_score = _best_duplicate_match(
                spikes, spikes_list, n_samp, fsamp,
                max_shift=dedup_max_shift, tol=dedup_tol)
            is_dup = dup_score >= dedup_th
            if is_dup:
                accept = False
            if verbose:
                match_str = (f"best match MU {dup_idx} agreement={dup_score:.2f}"
                             if dup_idx >= 0 else "no overlap with any accepted MU")
                verdict = (f"REJECT duplicate (>= {dedup_th:.2f})" if is_dup
                           else f"unique, keep (< {dedup_th:.2f})")
                print(f"[gCKC] dedup gate: candidate sil={sil:.3f} "
                      f"n={len(spikes)} vs {len(spikes_list)} accepted -> "
                      f"{match_str} -> {verdict}")

        if accept:
            spikes_list.append(np.asarray(spikes, dtype=int))
            sil_list.append(float(sil))
            cov_list.append(float(cov))
            filt_list.append(c.copy())
            consec_rejects = 0
            if verbose:
                print(f"[gCKC] MU {len(spikes_list)-1}: ACCEPT "
                      f"sil={sil:.3f} cov={cov:.3f} fr={fr:.2f} n={len(spikes)}")
        else:
            consec_rejects += 1
            # duplicates were already reported by the dedup-gate line above;
            # here we only log quality rejects (sil/cov/fr/min_spikes)
            if verbose and not is_dup:
                _sp = np.asarray(spikes, dtype=int)
                _loc = ",".join(str(int(s)) for s in _sp[:6])
                if _sp.size > 6:
                    _loc += ",..."
                print(f"[gCKC] reject  sil={sil:.3f} cov={cov:.3f} fr={fr:.2f} "
                      f"n={len(spikes)} seed={idx} spikes=[{_loc}]")

        # --- deflation (Fig. 1 step 8): zero γ at this train's discharges --
        # Done for EVERY reconstructed train (accepted or not) so the seed
        # always advances. The guard clears a small neighbourhood around each
        # discharge -- and the seed itself, which guarantees progress even when
        # a rejected candidate produced few/garbage spikes.
        sp = np.asarray(spikes, dtype=int)
        deflate = sp if idx is None else np.append(sp, idx)
        for n0 in deflate:
            lo = max(0, int(n0) - guard)
            hi = min(n_samp, int(n0) + guard + 1)
            deflate_mask[lo:hi] = True
            work[lo:hi] = -np.inf

        # --- peel-off (optional): remove the accepted unit from x̄ ----------
        # Subtract the STA reconstruction of the accepted train and refresh
        # C_x̄x̄^{-1}x̄ and γ (Cinv held fixed) so a later re-seed cannot land on
        # a unit already extracted. Gated on ACCEPT to bound the matmul cost.
        if peel_off and accept and sp.size:
            x_ext, Cx, gamma, peels_since_full = _apply_peel(
                x_ext, Cx, gamma, peels_since_full, sp)
            work = gamma.copy()
            work[deflate_mask] = -np.inf
            if verbose:
                print(f"[gCKC] peel-off MU {len(spikes_list)-1}; "
                      f"γ_max now {float(np.max(work)):.3g}")

        # --- dominating-attractor peel (reject-only) -----------------------
        # Short (1-2 discharge) rejected trains that many independent searches
        # keep re-converging onto are periodic artifacts, not MUs. Count each
        # discharge's ±5 ms bucket; once a bucket re-attracts the search
        # ``recur_peel_th`` times, peel ONLY that location from x̄ (same STA
        # mechanism) so the search stops wasting iterations on it. Real
        # multi-spike MUs are excluded by the ``sp.size <= 2`` gate, and only
        # proven attractors are removed, so legitimate peaks survive.
        if (not accept and np.isfinite(recur_peel_th)
                and 0 < sp.size <= 2):
            to_peel = []
            for n0 in sp:
                b = int(n0) // bucket_w
                attractor_hits[b] = attractor_hits.get(b, 0) + 1
                if attractor_hits[b] >= recur_peel_th and b not in peeled_buckets:
                    peeled_buckets.add(b)
                    to_peel.append(int(n0))
            if to_peel:
                x_ext, Cx, gamma, peels_since_full = _apply_peel(
                    x_ext, Cx, gamma, peels_since_full,
                    np.asarray(to_peel, dtype=int))
                work = gamma.copy()
                work[deflate_mask] = -np.inf
                if verbose:
                    print(f"[gCKC] peeled {len(to_peel)} recurrent attractor(s) "
                          f"at {to_peel}; γ_max now {float(np.max(work)):.3g}")

        # if consec_rejects >= max_consec_rejects:
        #     if verbose:
        #         print(f"[gCKC] {consec_rejects} consecutive rejects; stopping")
        #     break

    if verbose:
        print(f"[gCKC] accepted {len(spikes_list)} MU(s)")
    return spikes_list, sil_list, cov_list, filt_list


# --------------------------------------------------------------------------- #
# MUAP estimation                                                             #
# --------------------------------------------------------------------------- #
def spike_triggered_average(sig: np.ndarray, spikes, n_samples: int = 128) -> np.ndarray:
    """Multichannel MUAP by spike-triggered averaging (Eq. 7).

    MUAP_ij(n) = (1/|Ψ|) Σ_{n_k∈Ψ} x_i(n_k - N/2 + n),   n = 1..N

    Returns array of shape (n_channels, n_samples).
    """
    sig = np.asarray(sig)
    n_ch, n_t = sig.shape
    half = n_samples // 2
    spikes = np.asarray(spikes, dtype=int)

    acc = np.zeros((n_ch, n_samples))
    count = 0
    for s in spikes:
        a, b = s - half, s - half + n_samples
        if a >= 0 and b <= n_t:
            acc += sig[:, a:b]
            count += 1
    if count == 0:
        return acc
    return acc / count


def least_squares_muap(x_ext: np.ndarray, spike_trains, n_channels: int,
                       n_samples: int = 128) -> list:
    """MUAP estimation by least squares (Eq. 8):  Ĥ = (θ̄ᵀ θ̄)^{-1} θ̄ᵀ x̄.

    Solves jointly for all MUs to account for waveform overlap. Each MUAP is
    returned as (n_channels, n_samples). Kept as an alternative to STA; the
    paper found no significant difference and STA is the default.
    """
    n_feat, n_t = x_ext.shape
    J = len(spike_trains)
    # Build extended pulse-train design matrix Θ̄ (n_t x J*N)
    theta = np.zeros((n_t, J * n_samples))
    half = n_samples // 2
    for j, spk in enumerate(spike_trains):
        for s in np.asarray(spk, dtype=int):
            for n in range(n_samples):
                t = s - half + n
                if 0 <= t < n_t:
                    theta[t, j * n_samples + n] = 1.0
    # Use the un-extended channels: reconstruct base signal from x_ext top block
    base = x_ext[:n_channels, :]  # first delay block ≈ original channels
    H = np.linalg.pinv(theta.T @ theta) @ theta.T @ base.T  # (J*N x n_channels)
    muaps = []
    for j in range(J):
        muaps.append(H[j * n_samples:(j + 1) * n_samples, :].T)  # (n_ch, N)
    return muaps


def pca_smooth_muap(muap: np.ndarray, var_keep: float = 0.9) -> np.ndarray:
    """Denoise / smooth a MUAP with PCA, keeping >= ``var_keep`` of variance.

    PCA is taken across channels (each channel is a sample of the temporal
    waveform), as in the paper's STA+PCA variant.
    """
    mu = muap.mean(axis=0, keepdims=True)
    centred = muap - mu
    # SVD over channels x time
    U, S, Vt = np.linalg.svd(centred, full_matrices=False)
    if S.sum() == 0:
        return muap
    ratio = np.cumsum(S ** 2) / np.sum(S ** 2)
    k = int(np.searchsorted(ratio, var_keep) + 1)
    k = max(1, min(k, S.size))
    recon = (U[:, :k] * S[:k]) @ Vt[:k, :]
    return recon + mu


# --------------------------------------------------------------------------- #
# Stimulation-artifact removal                                                #
# (experimental: an electrical-stim artifact superimposed on the test EMG)    #
# --------------------------------------------------------------------------- #
def _infer_stim_period(env: np.ndarray, fsamp: float,
                       min_hz: float = 1.0, max_hz: float = 200.0) -> int:
    """Estimate the inter-pulse interval (samples) of a periodic train from a
    1-D rectified envelope by autocorrelation, restricted to [min_hz, max_hz].
    Returns 0 if no plausible lag is found.
    """
    env = np.asarray(env, float)
    env = env - env.mean()
    n = env.size
    if n < 4:
        return 0
    ac = np.correlate(env, env, mode="full")[n - 1:]
    lo = max(1, int(fsamp / max_hz))
    hi = min(n - 1, int(fsamp / min_hz))
    if hi <= lo:
        return 0
    return lo + int(np.argmax(ac[lo:hi]))


def _matched_filter(sig: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Multichannel matched-filter response: the summed cross-correlation of
    each channel with its template waveform ('same' alignment), so a peak marks
    the centre of a template match even when EMG is superimposed.
    """
    n_ch, n_t = sig.shape
    mf = np.zeros(n_t)
    for ch in range(n_ch):
        mf += fftconvolve(sig[ch], template[ch][::-1], mode="same")
    return mf


def _detect_stim_times(sig: np.ndarray, fsamp: float, template=None,
                       stim_hz=None, refractory_frac: float = 0.7,
                       height_frac: float = 0.3):
    """Locate stim pulses. With a ``template`` a multichannel matched filter is
    used (robust when EMG is superimposed); otherwise the cross-channel
    rectified envelope is used (for a stim-only reference recording). A relative
    height threshold tolerates per-pulse amplitude jitter. Returns
    (pulse_indices, period_samples).
    """
    sig = np.asarray(sig, float)
    score = _matched_filter(sig, template) if template is not None \
        else np.abs(sig).sum(0)
    period = (fsamp / float(stim_hz)) if stim_hz \
        else _infer_stim_period(np.abs(sig).sum(0), fsamp)
    distance = max(1, int(refractory_frac * period)) if period else 1
    smax = float(score.max()) if score.size else 0.0
    if not np.isfinite(smax) or smax <= 0:
        return np.array([], dtype=int), period
    peaks, _ = find_peaks(score, distance=distance, height=height_frac * smax)
    return peaks.astype(int), period


def estimate_stim_template(stim_sig: np.ndarray, fsamp: float,
                           win: float = 0.015, stim_hz=None,
                           refractory_frac: float = 0.7,
                           height_frac: float = 0.3):
    """Estimate the multichannel stimulation-artifact waveform from a recording
    that contains ONLY the stimulation (no / minimal voluntary EMG).

    Pulses are detected on the rectified cross-channel envelope (refined by
    ``stim_hz`` if given, else inferred), and a spike-triggered average over a
    ``win``-second window centred on each pulse yields the template.

    Returns (template (n_ch, N), pulse_indices, period_samples).
    """
    stim_sig = np.asarray(stim_sig, float)
    N = max(2, int(round(win * fsamp)))
    peaks, period = _detect_stim_times(
        stim_sig, fsamp, template=None, stim_hz=stim_hz,
        refractory_frac=refractory_frac, height_frac=height_frac)
    template = spike_triggered_average(stim_sig, peaks, N)
    return template, peaks, period


def _blank_regions(sig: np.ndarray, times, half: int, N: int,
                   pad: int = 0, fill: str = "linear") -> np.ndarray:
    """Replace the artifact window around each pulse in ``times`` instead of
    subtracting a fitted template ('blanking').

    The window is the template span ``[t-half, t-half+N)`` widened by ``pad``
    samples each side. ``fill`` chooses the replacement: ``"zero"`` sets the
    region to 0; ``"linear"`` interpolates each channel linearly between the
    samples just outside the (merged) region, avoiding the hard discontinuity
    that zeroing leaves on downstream covariance / MUAP estimation.

    Returns the removed component ``comp`` (so ``residual = sig - comp``,
    matching :func:`remove_stim_artifact`).
    """
    n_ch, n_t = sig.shape
    pad = max(0, int(pad))
    # build a boolean mask of all samples to blank, merging overlaps
    mask = np.zeros(n_t, bool)
    for t in times:
        lo = max(0, int(t) - half - pad)
        hi = min(n_t, int(t) - half + N + pad)
        if hi > lo:
            mask[lo:hi] = True
    if not mask.any():
        return np.zeros_like(sig)

    repl = sig.copy()
    if fill == "zero":
        repl[:, mask] = 0.0
    elif fill == "linear":
        # interpolate each contiguous masked run from its bounding samples
        edges = np.flatnonzero(np.diff(np.r_[0, mask.view(np.int8), 0]))
        for lo, hi in zip(edges[0::2], edges[1::2]):  # run = [lo, hi)
            l = lo - 1                       # last good sample before the run
            r = hi                           # first good sample after the run
            n_gap = hi - lo
            xs = np.arange(1, n_gap + 1, dtype=float)
            if l >= 0 and r < n_t:
                w = xs / (n_gap + 1)
                repl[:, lo:hi] = (sig[:, l:l + 1] * (1.0 - w)
                                  + sig[:, r:r + 1] * w)
            elif l >= 0:                      # run touches the right edge
                repl[:, lo:hi] = sig[:, l:l + 1]
            elif r < n_t:                     # run touches the left edge
                repl[:, lo:hi] = sig[:, r:r + 1]
            else:
                repl[:, lo:hi] = 0.0
    else:
        raise ValueError(f"Unknown blank fill mode: {fill!r}")
    return sig - repl


def remove_stim_artifact(sig: np.ndarray, template: np.ndarray, fsamp: float,
                         stim_hz=None, scale_per_pulse: bool = True,
                         align_jitter: int = 2, subsample_align: bool = True,
                         refractory_frac: float = 0.7,
                         height_frac: float = 0.3,
                         blank: bool = False, blank_fill: str = "linear",
                         blank_pad: int = 0):
    """Detect and subtract a stimulation-artifact ``template`` from ``sig``.

    Pulses are located with the multichannel matched filter (tolerates timing
    jitter from a nominally constant frequency); each is re-aligned within
    +/-``align_jitter`` samples and, if ``scale_per_pulse``, LS-scaled
    (a = <seg, template> / <template, template>, one scalar across channels) to
    track amplitude jitter before subtraction.

    With ``subsample_align`` the integer alignment is refined to a fractional
    offset by parabolic interpolation of the matched-filter peak, and the
    template is shifted by that fraction (cubic spline) before subtraction.
    This suppresses the derivative-shaped residual that integer-only alignment
    leaves on the sharp artifact edges when a pulse falls between samples.

    If ``blank`` is True the detected windows are *replaced* (zeroed or linearly
    interpolated, per ``blank_fill``) rather than template-subtracted -- the
    template is then used only to detect pulses, not to model their shape. This
    is more robust when the artifact is too variable to template well, at the
    cost of discarding the EMG inside each window. ``blank_pad`` widens each
    blanked window by that many samples on both sides.

    Returns (residual, comp, times).
    """
    sig = np.asarray(sig, float).copy()
    template = np.asarray(template, float)
    n_ch, n_t = sig.shape
    N = template.shape[1]
    half = N // 2
    times, _ = _detect_stim_times(
        sig, fsamp, template=template, stim_hz=stim_hz,
        refractory_frac=refractory_frac, height_frac=height_frac)

    if blank:
        comp = _blank_regions(sig, times, half, N, pad=blank_pad,
                              fill=blank_fill)
        return sig - comp, comp, times

    comp = np.zeros_like(sig)
    J = max(0, int(align_jitter))
    for t in times:
        a0 = t - half
        # integer-shift matched-filter scores (in-bounds shifts only)
        corrs = {}
        for sh in range(-J, J + 1):
            a = a0 + sh
            if a < 0 or a + N > n_t:
                continue
            corrs[sh] = float(np.sum(sig[:, a:a + N] * template))
        if not corrs:
            continue
        best_shift = max(corrs, key=corrs.get)

        # sub-sample refinement: parabolic vertex of the 3 scores straddling
        # the integer peak gives the fractional offset delta in [-0.5, 0.5].
        delta = 0.0
        if (subsample_align and (best_shift - 1) in corrs
                and (best_shift + 1) in corrs):
            y_m, y_0, y_p = (corrs[best_shift - 1], corrs[best_shift],
                             corrs[best_shift + 1])
            denom = y_m - 2.0 * y_0 + y_p
            if denom < 0.0:                      # concave => genuine maximum
                delta = float(np.clip(0.5 * (y_m - y_p) / denom, -0.5, 0.5))

        a = a0 + best_shift
        if a < 0 or a + N > n_t:
            continue
        # window stays at integer ``a``; the matched-filter peak sits delta
        # samples past the integer best, so the artifact in this window is the
        # template shifted by +delta -- subtract exactly that.
        tmpl = (_ndshift(template, (0.0, delta), order=3, mode="nearest")
                if delta else template)
        seg = sig[:, a:a + N]
        if scale_per_pulse:
            scale = float(np.sum(seg * tmpl)) / (float(np.sum(tmpl ** 2)) + _EPS)
        else:
            scale = 1.0
        comp[:, a:a + N] += scale * tmpl

    return sig - comp, comp, times


# --------------------------------------------------------------------------- #
# MU filter reconstruction (step 5 of Algorithm 1)                            #
# --------------------------------------------------------------------------- #
def _window_center(channel: np.ndarray, K: int, method: str) -> int:
    """Locate the centre n0 of the K-sample window on the dominant channel."""
    N = channel.size
    half = K // 2
    lo, hi = half, N - half  # valid centres so the window stays in-bounds

    if method == "centered_spike":
        return N // 2

    if method == "centered_peak":
        # centre on the absolute peak of the channel
        return int(np.clip(np.argmax(np.abs(channel)), lo, hi - 1))

    # Sliding-window scores for Maximum PPV / Maximum Product
    best_n0, best_val = lo, -np.inf
    for n0 in range(lo, hi):
        seg = channel[n0 - half:n0 + half]
        if seg.size == 0:
            continue
        vmax, vmin = seg.max(), seg.min()
        if method == "max_ppv":
            val = vmax - vmin
        elif method == "max_product":
            val = abs(vmax * vmin)
        else:
            raise ValueError(f"Unknown reconstruction method: {method}")
        if val > best_val:
            best_val, best_n0 = val, n0
            argmax = n0 - half + int(np.argmax(seg))
            argmin = n0 - half + int(np.argmin(seg))
            center_between = (argmax + argmin) // 2
    # tie-break: centre between extrema
    return int(np.clip(center_between, lo, hi - 1))


def reconstruct_mu_filter(
    muap: np.ndarray,
    K: int,
    method: str = "centered_peak",
    reverse_time: bool = True,
) -> np.ndarray:
    """Reconstruct an MU filter from a multichannel MUAP (step 5).

    A K-sample window is located on the channel with the largest absolute
    amplitude (the dominant channel) using ``method``; the SAME sample
    positions are then taken from every channel and cascaded into a single
    vector of length n_channels * K.

    Parameters
    ----------
    muap : (n_channels, N) array
    K : window length per channel; must match the extension factor used for
        the EMG so the filter aligns with the extended observation.
    method : 'centered_peak' (paper-recommended) | 'max_ppv' | 'max_product'
        | 'centered_spike'.
    reverse_time : flip each channel's window in time before cascading so the
        ordering matches ``core.extension``'s [x(n), x(n-1), ...] delay
        convention. Verify against your ``extension`` implementation.
    """
    muap = np.asarray(muap, dtype=float)
    n_ch, N = muap.shape
    half = K // 2

    dominant = int(np.argmax(np.max(np.abs(muap), axis=1)))
    n0 = _window_center(muap[dominant], K, method)

    start = int(np.clip(n0 - half, 0, N - K))
    window = muap[:, start:start + K]  # (n_ch, K), same positions across channels
    if reverse_time:
        window = window[:, ::-1]
    return window.reshape(-1)  # cascade channel-major -> (n_ch * K,)