"""Compare smap.py against MATLAB outputs of the reference functions on the same input.
Usage: python validate_vs_matlab.py syn.mat   (expects syn_ml.mat and syn_tl.mat beside it)
Pass criterion: max |diff| < 1e-8 on every array (float64 vs float64/single where noted)."""
import sys, numpy as np, scipy.io as sio
from smap import bin_zscore, fill_gaps, smap_pairs, smap_lambda, smap_timelocal_pair, miaaft

f = sys.argv[1]
X = sio.loadmat(f)['X']; ml = sio.loadmat(f.replace('.mat', '_ml.mat')); tl = sio.loadmat(f.replace('.mat', '_tl.mat'))
theta = (0, 0.25, 0.5, 1, 2, 4, 8, 16, 32)
Xb = bin_zscore(fill_gaps(X), 5)
r = smap_pairs(Xb, [(0, 1), (1, 0), (2, 1)], E=3, h=1, theiler=8, theta=theta, keep_ar=True)
lam, tau = smap_lambda(r['AR'])
rows = []
def cmp(name, a, b, tol=1e-8):
    a = np.asarray(a, float); b = np.asarray(b, float).reshape(a.shape)
    both = np.isfinite(a) & np.isfinite(b)
    d = np.max(np.abs(a[both] - b[both])) if both.any() else 0.0
    ok = d < tol and np.array_equal(np.isfinite(a), np.isfinite(b))
    rows.append(ok); print(f'{name:10s} max|diff| = {d:.2e}  {"PASS" if ok else "FAIL"}')
cmp('Xb', Xb, ml['Xb']); cmp('tBin', r['t'] + 1, ml['tBin'].ravel())
cmp('rho', r['rho'], ml['rho']); cmp('IS', r['IS'], ml['IS'])
cmp('AR', r['AR'], ml['AR'], 1e-6)          # MATLAB stores AR as single
cmp('lambda', lam, ml['lam'], 1e-6)
# tau = -1/log(lam) amplifies MATLAB's single-precision AR near lam -> 1: compare relatively
cmp('tau (rel)', tau / np.asarray(ml['tau']).reshape(tau.shape), np.where(np.isfinite(tau), 1.0, np.nan), 1e-4)
o = smap_timelocal_pair(Xb[:, 0], Xb[:, 1], E=3, h=1, theiler=8, theta=(0, 0.5, 2, 4, 16))
for k in ('rhoS', 'rhoT', 'ISS', 'IST', 'NeffS', 'NeffT', 'spreadS', 'spreadT'):
    cmp(k, o[k], tl[k] if tl[k].shape != (1, 5) else tl[k].ravel(), 1e-6)
cmp('bT', o['bT'][1:], tl['bT'].ravel()[1:], 1e-6)
# surrogate properties (RNG differs from MATLAB; check the invariants instead)
Xf = fill_gaps(X); s = miaaft(Xf, np.random.default_rng(0))
marg = all(np.array_equal(np.sort(s[:, j]), np.sort(Xf[:, j])) for j in range(X.shape[1]))
P = lambda Z: np.abs(np.fft.fft(Z - Z.mean(0), axis=0))
spec = np.median(np.abs(P(s) - P(Xf)) / P(Xf).mean(0))
xs = lambda Z: np.corrcoef(Z.T)[0, 1]
print(f'miaaft     marginals exact: {marg}; median rel spectral err {spec:.3f}; r01 obs {xs(Xf):.3f} surr {xs(s):.3f}')
rows.append(marg)
print('ALL PASS' if all(rows) else 'SOME FAILED'); sys.exit(0 if all(rows) else 1)
