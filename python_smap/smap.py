"""S-map (local linear) coupling for two physiological series -- Python reference port.

Faithful port of the MATLAB reference used in the MMV heart-brain work:
    mmv_heartBrainSmap.m      -> bin_zscore, smap_pair, smap_pairs, run_session
    mmv_smapLambda.m          -> smap_lambda
    mmv_heartBrainTimeLocal.m -> smap_timelocal_pair (matched time-local control)
    local_miaaft              -> miaaft
Validated against MATLAB R2026a by validate_vs_matlab.py (see HANDOFF_SMAP_PYTHON.md).

Conventions (do not change without reading the handoff):
  * DIRECTION is the forecasting sense: pair (src, tgt) = "src -> tgt" means src's past improves
    the forecast of tgt; IS = d tgt(t+h) / d src. This is OPPOSITE to the CCM reconstruction
    convention, although the arrow reads the same.
  * State z(t) = [y(t) .. y(t-E+1), x(t) .. x(t-E+1)]; target y(t+h). Query rows t = E-1 .. n-h-1
    (0-based); the MATLAB tBin (1-based, state closes at second tBin*bin) is t + 1.
  * Theiler exclusion |t_i - t_j| <= theiler INCLUDES SELF (the diagonal).
  * Weights w = exp(-theta * d / dbar), dbar = mean distance from the query to its (non-excluded)
    library. theta = 0 is the global lagged-linear regression (keep it as the built-in control).
  * Ridge on the slopes only (not the intercept), scaled per query by the mean slope diagonal of
    the weighted Gram matrix: lam = ridge * trace(G_slopes) / (p - 1).
Dependencies: numpy, scipy (cdist only). Memory is O(nT^2) per pair -- fine to nT ~ 10k.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist

DEFAULTS = dict(bin=5, E=3, h=1, theiler=8, theta=(0, 0.25, 0.5, 1, 2, 4, 8, 16, 32),
                ridge=1e-4, nSurr=99, maxIter=100)


# --------------------------------------------------------------------------- preprocessing
def bin_zscore(X: np.ndarray, b: int) -> np.ndarray:
    """Average non-overlapping blocks of b samples, then z-score each column (ddof=1, as MATLAB).
    X: (n, nVar) on a uniform grid with NaNs already filled. Trailing partial block is dropped."""
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    n = (X.shape[0] // b) * b
    Xb = X[:n].reshape(n // b, b, X.shape[1]).mean(axis=1)
    return (Xb - Xb.mean(0)) / Xb.std(0, ddof=1)


def fill_gaps(X: np.ndarray) -> np.ndarray:
    """Linear interpolation over NaNs, nearest-value at the ends (MATLAB fillmissing 'linear')."""
    X = np.array(X, float, copy=True)
    if X.ndim == 1:
        X = X[:, None]
    idx = np.arange(X.shape[0])
    for j in range(X.shape[1]):
        ok = np.isfinite(X[:, j])
        if not ok.all():
            X[:, j] = np.interp(idx, idx[ok], X[ok, j])   # np.interp holds end values = 'nearest'
    return X


# --------------------------------------------------------------------------- core S-map
def _design(x, y, E, h):
    n = len(y)
    t = np.arange(E - 1, n - h)                                   # 0-based query rows
    Zy = np.column_stack([y[t - l] for l in range(E)])
    Zx = np.column_stack([x[t - l] for l in range(E)])
    Z = np.hstack([Zy, Zx])
    A = np.hstack([np.ones((len(t), 1)), Z])
    return t, Z, A, y[t + h]


def _solve(W, A, yt, ridge):
    """Weighted ridge LS for every query row at once. W: (nT, nT) weights (row = query)."""
    p = A.shape[1]
    # G[i] = A' diag(W[i]) A ; b[i] = A' diag(W[i]) yt   -- via einsum, (nT, p, p)
    G = np.einsum('ij,jk,jl->ikl', W, A, A, optimize=True)
    bv = W @ (A * yt[:, None])
    d = np.arange(1, p)
    lam = ridge * G[:, d, d].sum(1) / (p - 1)
    G[:, d, d] += lam[:, None]
    beta = np.linalg.solve(G, bv[..., None])[..., 0]              # (nT, p)
    pred = (A * beta).sum(1)
    rho = np.corrcoef(pred, yt)[0, 1]
    return rho, beta, pred


def smap_pair(x, y, *, E=3, h=1, theiler=8, theta=DEFAULTS['theta'], ridge=1e-4,
              keep_ar=False, keep_beta=False):
    """S-map for one directed pair x -> y on binned, z-scored series.

    Returns dict:
      t     (nT,)            0-based query rows (state closes at bin t)
      rho   (nTheta,)        Pearson r of local forecast vs y(t+h)
      IS    (nT, nTheta)     interaction strength = sum of coefficients on x lags
      AR    (nT, E, nTheta)  coefficients on y(t), y(t-1), ... (if keep_ar)
      beta  (nT, p, nTheta)  full local coefficients [const, y lags, x lags] (if keep_beta)
      neff  (nTheta,)        mean Kish effective library size
    """
    x = np.asarray(x, float); y = np.asarray(y, float)
    t, Z, A, yt = _design(x, y, E, h)
    nT = len(t); theta = np.atleast_1d(np.asarray(theta, float))
    mask = np.abs(t[:, None] - t[None, :]) <= theiler              # includes self
    D = cdist(Z, Z)
    Dl = np.where(mask, np.nan, D)
    dbar = np.nanmean(Dl, axis=1)
    out = dict(t=t, rho=np.full(len(theta), np.nan), IS=np.full((nT, len(theta)), np.nan),
               neff=np.full(len(theta), np.nan))
    if keep_ar:
        out['AR'] = np.full((nT, E, len(theta)), np.nan)
    if keep_beta:
        out['beta'] = np.full((nT, A.shape[1], len(theta)), np.nan)
    for k, th in enumerate(theta):
        W = np.exp(-th * D / dbar[:, None])
        W[mask] = 0.0
        rho, beta, _ = _solve(W, A, yt, ridge)
        out['rho'][k] = rho
        out['IS'][:, k] = beta[:, 1 + E:1 + 2 * E].sum(1)
        out['neff'][k] = np.mean(W.sum(1) ** 2 / (W ** 2).sum(1))
        if keep_ar:
            out['AR'][:, :, k] = beta[:, 1:1 + E]
        if keep_beta:
            out['beta'][:, :, k] = beta
    return out


def smap_pairs(Xb, pairs, **kw):
    """Run smap_pair over a list of (src, tgt) column indices of Xb. Returns rho (nPair, nTheta),
    IS (nT, nTheta, nPair) and, with keep_ar=True, AR (nT, E, nTheta, nPair) -- MATLAB layout."""
    res = [smap_pair(Xb[:, s], Xb[:, g], **kw) for s, g in pairs]
    out = dict(t=res[0]['t'], rho=np.stack([r['rho'] for r in res]),
               IS=np.stack([r['IS'] for r in res], axis=-1))
    if kw.get('keep_ar'):
        out['AR'] = np.stack([r['AR'] for r in res], axis=-1)
    return out


# --------------------------------------------------------------------------- relaxation rate
def smap_lambda(AR, bin_s=None):
    """Dominant-root modulus of the companion matrix of the target-lag block.
    AR: (..., E) with the lag axis LAST, or the (nT, E, nTheta[, nPair]) layout (lag axis 1).
    Returns lam and tau (decay time; in bins, or seconds if bin_s given), NaN where lam >= 1."""
    AR = np.asarray(AR, float)
    if AR.ndim >= 3:                                              # MATLAB layout -> lag last
        AR = np.moveaxis(AR, 1, -1)
    E = AR.shape[-1]
    flat = AR.reshape(-1, E)
    lam = np.full(flat.shape[0], np.nan)
    ok = np.isfinite(flat).all(1)
    if E == 1:
        lam[ok] = np.abs(flat[ok, 0])
    else:
        C = np.zeros((ok.sum(), E, E))
        C[:, 0, :] = flat[ok]
        C[:, np.arange(1, E), np.arange(E - 1)] = 1.0
        lam[ok] = np.abs(np.linalg.eigvals(C)).max(1)
    lam = lam.reshape(AR.shape[:-1])
    with np.errstate(divide='ignore', invalid='ignore'):
        tau = -1.0 / np.log(lam)
    tau[~(lam < 1)] = np.nan
    if bin_s is not None:
        tau = tau * bin_s
    return lam, tau


# --------------------------------------------------------------------------- time-local control
def _kish(W):
    return W.sum(1) ** 2 / (W ** 2).sum(1)


def smap_timelocal_pair(x, y, *, E=3, h=1, theiler=8, theta=DEFAULTS['theta'], ridge=1e-4):
    """Same model, library, Theiler and ridge, but neighbours weighted by |dt| with a per-query
    bandwidth matched to the state kernel's Kish effective N. Returns rhoS/rhoT, ISS/IST,
    NeffS/NeffT, bT (median bandwidth, bins), spreadS/spreadT (weighted mean |dt|, bins)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    t, Z, A, yt = _design(x, y, E, h)
    nT = len(t); theta = np.atleast_1d(np.asarray(theta, float))
    dt = np.abs(t[:, None] - t[None, :]).astype(float)
    mask = dt <= theiler
    D = cdist(Z, Z); dbar = np.nanmean(np.where(mask, np.nan, D), 1)
    bg = np.exp(np.linspace(np.log(0.25), np.log(50 * nT), 120))
    Nt = np.empty((nT, len(bg)))
    for g, b in enumerate(bg):
        w = np.exp(-dt / b); w[mask] = 0; Nt[:, g] = _kish(w)
    Nt = np.maximum.accumulate(Nt, axis=1)
    lNt, lbg = np.log(Nt), np.log(bg)
    nb = len(bg)
    O = {k: np.full(len(theta), np.nan) for k in ('rhoS', 'rhoT', 'NeffS', 'NeffT', 'bT', 'spreadS', 'spreadT')}
    O['ISS'] = np.full((nT, len(theta)), np.nan); O['IST'] = O['ISS'].copy(); O['t'] = t
    for k, th in enumerate(theta):
        Ws = np.exp(-th * D / dbar[:, None]); Ws[mask] = 0
        Ns = _kish(Ws)
        O['rhoS'][k], beta, _ = _solve(Ws, A, yt, ridge)
        O['ISS'][:, k] = beta[:, 1 + E:].sum(1)
        O['NeffS'][k] = Ns.mean(); O['spreadS'][k] = np.mean((Ws * dt).sum(1) / Ws.sum(1))
        if th == 0:
            for a, b_ in (('rhoT', 'rhoS'), ('NeffT', 'NeffS'), ('spreadT', 'spreadS')):
                O[a][k] = O[b_][k]
            O['IST'][:, k] = O['ISS'][:, k]; O['bT'][k] = np.inf
            continue
        lNs = np.log(Ns)                                          # log-log interpolation per query
        g = np.clip((lNt < lNs[:, None]).sum(1), 1, nb - 1) - 1   # 0-based lower grid index
        r = np.arange(nT)
        fr = (lNs - lNt[r, g]) / np.maximum(lNt[r, g + 1] - lNt[r, g], np.finfo(float).eps)
        fr = np.clip(fr, 0, 1)
        b = np.exp(lbg[g] + fr * (lbg[g + 1] - lbg[g]))
        Wt = np.exp(-dt / b[:, None]); Wt[mask] = 0
        O['rhoT'][k], beta, _ = _solve(Wt, A, yt, ridge)
        O['IST'][:, k] = beta[:, 1 + E:].sum(1)
        O['NeffT'][k] = _kish(Wt).mean(); O['bT'][k] = np.median(b)
        O['spreadT'][k] = np.mean((Wt * dt).sum(1) / Wt.sum(1))
    return O


# --------------------------------------------------------------------------- surrogates
def miaaft(X, rng: np.random.Generator, max_iter=100):
    """Multivariate IAAFT (Schreiber & Schmitz 2000): preserves each marginal exactly and the
    amplitude spectra + cross-spectra (common random phase added to all columns). Apply to the
    UNBINNED series, then bin_zscore -- as the reference does."""
    X = np.asarray(X, float)
    n = X.shape[0]
    srt = np.sort(X, 0)
    F = np.fft.fft(X, axis=0); Amp = np.abs(F); ph = np.angle(F)
    k = np.arange(1, int(np.ceil(n / 2)))                         # MATLAB 2:ceil(n/2), 0-based

    def synth(alpha):
        Y = F.copy()
        Y[k] = Amp[k] * np.exp(1j * (ph[k] + alpha[k, None]))
        Y[n - k] = np.conj(Y[k])
        return np.real(np.fft.ifft(Y, axis=0))

    def remap(s):
        ord_ = np.argsort(s, 0, kind='stable')
        y = np.empty_like(s); rk = np.empty(s.shape, int)
        for j in range(s.shape[1]):
            y[ord_[:, j], j] = srt[:, j]; rk[ord_[:, j], j] = np.arange(n)
        return y, rk

    alpha = np.zeros(n); alpha[k] = 2 * np.pi * rng.random(len(k))
    s, rk = remap(synth(alpha))
    for _ in range(max_iter):
        Sf = np.fft.fft(s, axis=0)
        alpha = np.zeros(n); alpha[k] = np.angle((np.conj(F[k]) * Sf[k]).sum(1))
        rk_old = rk
        s, rk = remap(synth(alpha))
        if np.array_equal(rk, rk_old):
            break
    return s


# --------------------------------------------------------------------------- one session
def run_session(X, pairs, *, bin=5, E=3, h=1, theiler=8, theta=DEFAULTS['theta'], ridge=1e-4,
                nSurr=99, maxIter=100, seed=0, keep_ar=False):
    """Full per-session run as in mmv_heartBrainSmap: observed S-map + multivariate-IAAFT null.
    X: (nSamples, nVar) raw series on a uniform grid (e.g. 1 Hz). Returns observed rho/IS/(AR),
    rhoNull (nPair, nTheta, nSurr), sdIS (nPair, nTheta), sdISNull (nPair, nTheta, nSurr)."""
    X = fill_gaps(X)
    kw = dict(E=E, h=h, theiler=theiler, theta=theta, ridge=ridge)
    obs = smap_pairs(bin_zscore(X, bin), pairs, keep_ar=keep_ar, **kw)
    obs['sdIS'] = np.nanstd(obs['IS'], axis=0, ddof=1).T
    rng = np.random.default_rng(seed)
    nP, nTh = obs['rho'].shape
    obs['rhoNull'] = np.full((nP, nTh, nSurr), np.nan); obs['sdISNull'] = obs['rhoNull'].copy()
    for s in range(nSurr):
        r = smap_pairs(bin_zscore(miaaft(X, rng, maxIter), bin), pairs, **kw)
        obs['rhoNull'][:, :, s] = r['rho']
        obs['sdISNull'][:, :, s] = np.nanstd(r['IS'], axis=0, ddof=1).T
    obs['opts'] = dict(bin=bin, **kw, nSurr=nSurr, maxIter=maxIter, seed=seed)
    return obs
