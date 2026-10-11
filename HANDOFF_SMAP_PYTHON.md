# S-map in Python — implementation handoff for cloud agents

Written 2026-10-11. For an agent running **without MATLAB and without the Box tree** (a cloud
sandbox, a batch worker). It gives a validated Python reference, the exact algorithm, the data
contract, how to scale it, and the traps. **What the numbers mean** and how to test them is in
`HANDOFF_SMAP_GENERAL_METHOD.md`; read it before interpreting anything. This file is about getting
the computation right.

Ship these together:

```
python_smap/
  smap.py                    # the reference module (numpy + scipy only)
  validate_vs_matlab.py      # regression test against MATLAB outputs
  fixtures/syn.mat           # synthetic input (3000 s, 3 series, NaN gap)
  fixtures/syn_ml.mat        # MATLAB outputs: Xb, rho, IS, tBin, AR, lam, tau
  fixtures/syn_tl.mat        # MATLAB outputs of the time-local control
  fixtures/make_synthetic.py, smap_harness.m, tl_harness.m   # how the fixtures were made
HANDOFF_SMAP_GENERAL_METHOD.md   # interpretation, inference, design rules
```

---

## 0. First thing to do in a new environment

```bash
pip install "numpy>=1.24" "scipy>=1.10"
cd python_smap && python validate_vs_matlab.py fixtures/syn.mat
```

It must print `ALL PASS`. Measured on 2026-10-11 against MATLAB R2026a (`mmv_heartBrainSmap.m`,
`mmv_smapLambda.m`, `mmv_heartBrainTimeLocal.m`):

| array | max abs diff vs MATLAB |
|---|---|
| binned z-scored series | 1e-15 |
| rho (9 θ × 3 pairs) | 6e-14 |
| IS | 3e-11 |
| AR (MATLAB stores single) | 2e-7 |
| λ | 4e-7 |
| τ (relative) | 3e-5 |
| time-local control: rhoS/rhoT, ISS/IST, Neff, spread, bT | ≤ 4e-12 |
| multivariate IAAFT | marginals exact; RNG differs from MATLAB by design, so invariants are checked instead |

**If you change anything in `smap.py`, rerun this before using the output.** Do not "simplify" the
code until it passes again.

---

## 1. Data contract

Input per session: `X`, an `(nSamples, nVar)` float array on a **uniform grid** (MMV used 1 Hz:
one row per second), one column per series, NaN where missing. Typical columns: a heart series
(RR, HR) and EEG features (FOOOF residual band power, aperiodic exponent).

- NaN gaps → `fill_gaps` (linear interpolation, nearest value at the ends). Long gaps should be
  excluded upstream, not interpolated through.
- The **surrogates are made from the unbinned series** and binned afterwards. Keep that order.
- `bin_zscore(X, bin)` averages non-overlapping blocks of `bin` samples, drops the trailing
  partial block, then z-scores each column with `ddof=1` (MATLAB `zscore`). Using `ddof=0`
  breaks the bit-for-bit match.
- Pairs are `(source_col, target_col)`, 0-based.

Store inputs and outputs as `.npz` or `.mat` (`scipy.io.savemat`). The MATLAB layouts are
kept: `IS` is `(nT, nTheta, nPair)`, `AR` is `(nT, E, nTheta, nPair)`.

---

## 2. The algorithm, exactly

For one directed pair x → y on binned, z-scored series of length n:

1. **Query rows** `t = E−1 … n−h−1` (0-based). MATLAB's `tBin` is `t + 1`; the state at t
   closes at second `(t+1)·bin`.
2. **State** `z(t) = [y(t), y(t−1), …, y(t−E+1), x(t), …, x(t−E+1)]` — target lags first.
   Design row `A(t) = [1, z(t)]`, so `p = 1 + 2E`. Response `y(t+h)`.
3. **Distances** `D = cdist(Z, Z)` (Euclidean, all queries vs all queries).
4. **Theiler mask** `|tᵢ − tⱼ| ≤ theiler`, **including the diagonal** (self). Masked pairs get
   weight 0 and are excluded from d̄.
5. **d̄ᵢ** = mean of Dᵢⱼ over the non-masked library of query i.
6. **Weights** `Wᵢⱼ = exp(−θ·Dᵢⱼ/d̄ᵢ)`, zero on the mask. θ = 0 gives uniform weights = global
   lagged-linear regression.
7. **Per-query weighted ridge LS**: `Gᵢ = Aᵀ diag(Wᵢ) A`, `bᵢ = Aᵀ diag(Wᵢ) y(t+h)`.
   Ridge is added to the **slope diagonal only** (not the intercept) with a per-query scale:
   `λᵢ = ridge · trace(Gᵢ[1:,1:]) / (p−1)`, ridge = 1e-4. Solve `βᵢ = Gᵢ⁻¹ bᵢ`.
8. **Outputs**
   - `rho(θ)` = Pearson r between `A·β` (in-sample-excluded, because of the mask) and `y(t+h)`.
   - `IS(t, θ)` = `Σ β[1+E : 1+2E]` — sum of coefficients on the source lags (∂y(t+h)/∂x).
   - `AR(t, :, θ)` = `β[1 : 1+E]` — coefficients on y(t), y(t−1), … (the target's own block).
9. **λ** = modulus of the dominant eigenvalue of the companion matrix
   `C = [a₁ … a_E ; I_{E−1} 0]`; **τ = −1/log λ** in bins (× bin for seconds), **NaN where λ ≥ 1**
   (do not clamp). Requires E ≥ 3 to be meaningful.

Defaults (MMV, 1 Hz input): `bin=5 s, E=3, h=1, theiler=8 bins (40 s),
theta=(0, .25, .5, 1, 2, 4, 8, 16, 32), ridge=1e-4, nSurr=99, maxIter=100`.
Set `bin` and `theiler` **per dataset** (method handoff §3); set Theiler in seconds and convert.

**Direction.** `(src, tgt)` = "src → tgt" in the **forecasting** sense. This is the *opposite* of
the CCM reconstruction convention, though the arrow is written the same way. Two earlier ports in
this project had reversed direction docs (`ccm-direction-and-expweight-traps`); state the
convention in every output file's metadata.

---

## 3. API (`smap.py`)

```python
from smap import fill_gaps, bin_zscore, smap_pair, smap_pairs, smap_lambda, \
                 smap_timelocal_pair, miaaft, run_session

Xb  = bin_zscore(fill_gaps(X), bin=5)
r   = smap_pair(Xb[:, 0], Xb[:, 1], E=3, h=1, theiler=8, keep_ar=True)   # x -> y
r['rho'], r['IS'], r['AR'], r['neff'], r['t']
lam, tau_s = smap_lambda(r['AR'], bin_s=5)          # (nT, nTheta), tau in seconds

tl  = smap_timelocal_pair(Xb[:, 0], Xb[:, 1])       # matched time-local control
S   = run_session(X, pairs=[(0,1),(1,0)], bin=5, nSurr=99, seed=subj*100+sess+7, keep_ar=True)
S['rho'], S['rhoNull'], S['sdIS'], S['sdISNull']    # observed + multivariate-IAAFT null
```

`run_session` reproduces `mmv_heartBrainSmap` end to end. Surrogate *values* differ from MATLAB
(different RNG); null *distributions* are equivalent. Record the seed.

---

## 4. Running it on cloud workers

**Unit of work = one session.** Sessions are independent; parallelise across them (one process
per session, `multiprocessing`/job array/batch queue). Do not parallelise inside a session first —
numpy's BLAS already uses threads; set `OMP_NUM_THREADS=1` per worker when packing many workers
on one machine.

**Cost.** Memory is O(nT²) per pair: D, W and the `(nT, p, p)` Gram stack. Measured: one pair ×
9 θ at nT ≈ 1000 bins takes ~0.5 s. A session with 12 pairs is ~6 s observed; with 99
surrogates ~10 min. Scaling with nT is quadratic:

| nT (bins) | D/W float64 | Gram stack (E=3) | guidance |
|---|---|---|---|
| 1 000 | 8 MB | 0.4 MB | trivial |
| 5 000 | 200 MB each | 2 MB | fine on 2–4 GB workers |
| 20 000 | 3.2 GB each | 8 MB | chunk queries (see below) |

Note the einsum that builds the Gram stack is O(nT²·p²) compute; it dominates at large nT.
For long recordings, **chunk over query rows**: compute D, W and G for a block of queries against
the full library, solve, concatenate. Results are identical because each query's fit is
independent. d̄ must still be computed over the full library.

**Checkpointing.** Write one output file per session (and, for long surrogate runs, one per
block of surrogates) so a pre-empted worker resumes rather than restarts. Name files
`sub-XX_ses-Y_smap.npz`; include `opts`, `pairs`, pair names, the direction convention string,
the seed, and the git hash/version of `smap.py`.

**Determinism.** Use `np.random.default_rng(seed)` per session, never the global RNG — otherwise
results depend on worker scheduling.

---

## 5. Traps specific to the port (each was checked)

| trap | what goes wrong | how it's handled |
|---|---|---|
| 0- vs 1-based indexing | queries shifted one bin; lags misaligned | `t = arange(E−1, n−h)`; `tBin = t+1`; verified to 0 diff |
| `zscore` ddof | 0.03% scale error everywhere | `ddof=1` |
| Theiler without self | a query's own future leaks into its coefficients | mask `<=`, diagonal included |
| d̄ over the whole row | θ silently means something else | d̄ excludes the masked library |
| ridge on intercept / unscaled ridge | coefficients shrink differently from MATLAB | slopes only, scaled by mean slope diagonal |
| IAAFT Nyquist / DC bins | spectra not preserved; marginal broken | phases randomised on bins 1…ceil(n/2)−1 only, mirrored conjugate; DC and (even-n) Nyquist untouched |
| IAAFT phase per column | destroys cross-spectra (univariate surrogates) | one common random phase across columns, then per-iteration phase fit to the original cross-spectrum |
| binning before surrogates | surrogates of block means, wrong null | surrogate raw `X`, then `bin_zscore` |
| τ near λ = 1 | huge, unstable τ | NaN for λ ≥ 1; report medians, not means |
| θ chosen by surrogate excess | picks an overfitting θ | choose by observed local-over-global gain (method §3) |
| float32 storage | AR/IS differences at 1e-7, τ amplified | compute in float64; downcast only for storage |

---

## 6. What to compute, in order (minimum viable pipeline)

1. **Intake**: uniform grid, gaps, column names, units; choose `bin` from the coupling's known
   band and Theiler in seconds.
2. **Observed run**, all θ, `keep_ar=True`. Check `rho(θ=0)` is sensible and that τ (seconds)
   lands in the expected band. Check `neff` — at the chosen θ, how much of the library is used.
3. **θ choice** from observed gain `rho(θ) − rho(0)`; keep θ = 0 as a column in every
   downstream table.
4. **Null**: `run_session(..., nSurr=99)`. Report per pair the gain vs `rhoNull` and the SD(IS)
   ratio vs `sdISNull`; FDR across pairs.
5. **Time-local control** (`smap_timelocal_pair`) before calling anything "geometric".
6. **Downstream models** (behaviour, events) per `HANDOFF_SMAP_GENERAL_METHOD.md` §4–§7:
   anchor rule (anchor ≥ 2 × bin), levels + squared levels as covariates, circular-shift p for
   autocorrelated outcomes, per-participant refits. Those steps are not in `smap.py` — they are
   dataset-specific and were MATLAB-only (`mmv_heartBrainLambdaModeration.m`,
   `mmv_lambdaOngoingState.m`, `mmv_gateParticipantLevel.m`). Port them against this module's
   output layout.

---

## 7. Not yet ported / open

- The embedding-sensitivity sweep (`mmv_smapEmbeddingSensitivity.m`) and the moderation /
  shift-null / participant-level scripts — the analyses, not the S-map.
- Chunked-query mode is described (§4) but not implemented; implement it as a loop over query
  blocks inside `smap_pair` and re-run the validation.
- Open science questions are listed in the method handoff §11 (no surrogate test of the
  moderation itself; no λ null; hysteresis untested).
