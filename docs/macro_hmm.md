# The `macro_analysis` Node: Complete ML Walkthrough

A reference for defending the machine-learning component of ARGUS's macro agent
in an interview or viva. It covers the model, the data path, the features, the
fit protocol, the labeling scheme, the validation gate, inference, and the
downstream consumers — followed by an anticipated question bank with answers.

Code entry points referenced throughout:

| Concern | File |
|---|---|
| Model, persistence, inference | `argus/agents/macro_hmm.py` |
| Orchestration, caching, `MacroContext` assembly | `argus/agents/macro.py` |
| Rule-based fallback and label derivations | `argus/agents/macro_context.py` |
| Feature construction (shared by fit + predict) | `argus/data/macro_features.py` |
| Free numeric parameters + provenance | `argus/params.py` (`MacroParams`) |
| Data access seam (live vs fixture) | `argus/seams.py` |
| Raw fetchers (FRED, yfinance) | `argus/data/fetchers.py` |
| Training CLI + validation gate | `scripts/train_macro_hmm.py` |
| Output contract | `argus/schemas/signals.py` (`MacroContext`, `Regime`) |
| Graph node | `argus/orchestration/nodes.py` (`node_macro_analysis`, `GraphDeps`) |
| Graph wiring | `argus/orchestration/graph.py` (`build_graph`) |
| Tests | `tests/test_macro.py` |
| Persisted artifact | `argus/models/macro_hmm.joblib` |

---

## 1. What the node is for

`macro_analysis` runs after `fetch_price_history`, in parallel with technical,
fundamental and sentiment analysis. Its output informs aggregation, risk and
portfolio synthesis; those three specialists do not consume it. It answers one
question: **which latent macroeconomic regime are we in right now, and how
confident is the model?**

It emits a `MacroContext` carrying:

- `macro_regime ∈ {EXPANSION, CONTRACTION, TRANSITIONAL}` — the ML output
- `regime_confidence ∈ [0, 1]` — the HMM posterior for that regime
- `model_healthy: bool` — whether that confidence is a real posterior or the
  fixed-confidence rule-based fallback
- Derived descriptive fields (`yield_curve_shape`, `vix_regime`,
  `inflation_trajectory`, `sector_rotation_signal`, …) — deterministic
  thresholding, *not* ML
- `agent_multipliers` — conviction scalars applied to the fundamental,
  technical, and sentiment agents

When `macro_context is None` after a total bundle outage, the specialists still
run. Aggregation uses neutral macro weighting, risk tries a direct VIX lookup,
and portfolio synthesis reports the missing context. A missing direct VIX reading
blocks risk approval instead of fabricating a volatility level.

---

## 2. Model choice: Gaussian Hidden Markov Model

### 2.1 Formal specification

Let $x_t \in \mathbb{R}^5$ be the standardized macro feature vector for month
$t$, and $z_t \in \{1,2,3\}$ the unobserved regime. The model is:

$$
z_1 \sim \pi, \qquad
z_t \mid z_{t-1} \sim \text{Cat}(A_{z_{t-1}, \cdot}), \qquad
x_t \mid z_t = k \sim \mathcal{N}(\mu_k, \Sigma_k)
$$

Parameters $\theta = (\pi, A, \{\mu_k, \Sigma_k\})$:

- $\pi \in \Delta^2$ — initial state distribution (3 states, 2 free params)
- $A \in [0,1]^{3\times3}$, rows summing to 1 — transition matrix (6 free)
- $\mu_k \in \mathbb{R}^5$ — per-state emission mean (15 free)
- $\Sigma_k \in \mathbb{S}^{5}_{++}$ — **full** per-state covariance
  ($3 \times 15 = 45$ free)

**Total: 68 free parameters** against 427 monthly observations (1990→present),
i.e. roughly 6 observations per parameter. That ratio is the single most
important number to have memorized — it is the honest answer to "is this model
over-parameterized?" (see Q&A 5.4).

`RegimeClassifier.__init__` in `argus/agents/macro_hmm.py` builds only a placeholder, through the one `_new_hmm()` builder
that reads `MACRO.hmm_covariance_type` (`"full"`) and `MACRO.hmm_n_iter` (300):

```python
GaussianHMM(n_components=3, covariance_type="full", n_iter=300)
```

The artifact loader compares the loaded model's `covariance_type` and `n_iter` against those
two parameters, so changing either forces a retrain instead of silently serving a model fit
under different settings.

`fit()` immediately discards it — it fits 20 candidates with `random_state = 0
… 19` and keeps the best non-degenerate one (§5.3). There is no fixed default
seed; the model that ships is whichever restart won, not seed 42 (that was the
old single-seed configuration, and it was one of the seeds that collapsed).

### 2.2 Why an HMM rather than the alternatives

| Alternative | Why rejected |
|---|---|
| **k-means / GMM clustering** | Treats months as i.i.d. Regimes are *persistent* — the probability of being in a recession this month depends overwhelmingly on last month. An HMM encodes that in $A$; a GMM throws it away and produces a regime label that flickers month to month. |
| **Hand-coded rules** (the fallback) | Requires me to pre-specify thresholds for regimes. The point of the latent-variable formulation is that the regime definition is *learned* from the joint behavior of five indicators rather than asserted. The rules survive only as a degradation path. |
| **Supervised classifier on NBER labels** | NBER recession dating is published with a 6–18 month lag and is revised. Training a supervised model on it means training on labels that don't exist in real time. Here NBER (`USREC`) is used only *after* fitting, to attach semantic names to states the model found on its own — an evaluation signal, not a training target. |
| **Markov-switching regression (Hamilton 1989)** | A closer relative, and arguably the correct econometric citation for this design. It models a switching *conditional mean of a target series*; here there is no single target, only a joint density over five indicators, so a plain HMM is the more direct formulation. Worth naming explicitly — it is the canonical literature this sits in. |
| **Change-point detection** | Detects breaks, doesn't produce a persistent labeled state you can condition trading behavior on, and offers no posterior probability. |

### 2.3 Why 3 states

Fixed at `n_components=3` because the downstream contract is a 3-valued `Regime`
enum consumed by the aggregator, the cultural-memory tagger, and the portfolio
prompt. **BIC does not favour k=3** on the shipped training run: k=2 → 4533.6,
k=3 → 4795.5, k=4 → 3839.6, k=5 → 3903.0 — it is *worse* than k=2, drops
sharply at k=4, then ticks back up slightly at k=5. It is not the "decreases
monotonically, so a larger model wins" story a first pass might expect; a
purely statistical criterion would prefer k=4 over both its neighbours, not
simply "larger." (This diagnostic fits once per k at a fixed `random_state=0`,
not the fit's best-of-20 protocol, so some of that non-monotonicity is plausibly
single-seed noise rather than a real preference — worth naming if pressed.)
k>3 was rejected regardless: it forces an arbitrary many-to-3 collapse onto the
enum for no validated gain, while k=3 already achieves 3.8× NBER enrichment.
**BIC for k ∈ {2..5} is logged as a diagnostic so the choice stays visible
rather than silently defensible.** Be ready to say "I chose interpretability
over BIC, and BIC doesn't even cleanly agree with itself here" — that is a much
stronger answer than pretending BIC agreed.

### 2.4 Why full covariance

Macro indicators are strongly correlated (rate changes and curve slope; VIX
percentile and unemployment change). `covariance_type="diag"` would force each
state's ellipsoid axis-aligned and push the model to spend extra states
representing correlation structure. The cost is 45 of the 68 parameters. If
pressed on the sample-size ratio, the honest mitigation is that a `"tied"` or
`"diag"` covariance is the first knob to try under a shrinkage argument — and
that the degeneracy gate (§6.2) is what actually protects against the
over-parameterized failure mode.

---

## 3. Data loading

### 3.1 Sources

All access flows through the `MarketDataProvider` protocol (`argus/seams.py`) —
never direct fetcher calls — so training, live inference, and fixture-backed
tests traverse the same code path.

| Series | ID | Source | Native frequency |
|---|---|---|---|
| Effective Federal Funds Rate | `FEDFUNDS` | FRED | Monthly |
| Unemployment rate | `UNRATE` | FRED | Monthly |
| CPI, all urban consumers | `CPIAUCSL` | FRED | Monthly |
| 10Y–2Y Treasury spread | `T10Y2Y` | FRED | Daily |
| CBOE Volatility Index | `^VIX` | yfinance (`period="max"`) | Daily |
| NBER recession indicator | `USREC` | FRED | Monthly (labeling/eval only) |

`LiveMarketDataProvider` delegates to `argus/data/fetchers.py`;
`FixtureMarketDataProvider` serves pre-captured JSON from
`tests/fixtures/market_data/`. FRED access requires `FRED_API_KEY`; without it
`fetch_fred_series` degrades and the agent ends up on the rule-based path.

### 3.2 Why `period="max"` for VIX is hardcoded

`build_macro_feature_frame` fetches VIX with `period="max"` rather than a
caller-supplied period. The 5-year rolling percentile needs ~1260 trading days
of warm-up; if training used `"max"` and inference used `"2y"`, the
`vix_pctile` feature would be computed against different reference
distributions in fit and serve — a textbook train/serve skew. Hardcoding it
makes the two paths bit-identical for the same provider.

### 3.3 Caching

- `fetch_fred_series` caches responses (see `argus/data/cache.py`).
- `MacroStatisticalAgent` caches the whole `MacroContext` for a **6-hour TTL**.
  Justified because the regime only moves monthly; the daily-responsive fields
  (VIX level/percentile/regime and therefore the multipliers) are the parts that
  actually decay within a day, which is the known cost of that TTL.

---

## 4. Preprocessing and feature engineering

All of this lives in one function — `build_macro_feature_frame(provider, start)`
in `argus/data/macro_features.py` — called by **both** `fit_on_history()` and
`analyze()`. That single-code-path property is the design's most important
structural guarantee: features cannot drift between training and serving,
because there is only one construction.

### 4.1 Point-in-time alignment (look-ahead bias)

FRED indexes monthly series by their **reference** date, not their
**publication** date. The April 2020 unemployment print is stamped 2020-04-01
but wasn't public until early May. Training on the raw index means the model
sees, on that row, information no real-time observer had.

Each monthly series' index is therefore shifted forward by its published
release lag (`MacroParams`, tagged `Provenance.LITERATURE` — these come from
BLS/Fed release calendars):

```python
fed_funds.index   += pd.Timedelta(days=32)   # Fed H.15
unemployment.index += pd.Timedelta(days=35)  # BLS Employment Situation
cpi.index          += pd.Timedelta(days=45)  # BLS CPI
```

`T10Y2Y` and `^VIX` are market data, known same-day, and are not shifted.

Magnitude of the bias this removes (measured against the unlagged series):
`cpi_yoy` differs on **99.5%** of rows, `fed_funds` on 79.7%, `unemployment` on
74.0%, with a maximum error of **11.3 percentage points** — on 2020-04-15 the
unlagged construction sees UNRATE 14.8 when a real-time observer knew 4.4.

### 4.2 Monthly sampling

```python
frame = frame.resample("ME").last()
return frame.ffill(limit=2).dropna()
```

The previous implementation forward-filled monthly series to **calendar-daily**
resolution, producing 13,010 rows from ~439 real observations. Consequences,
all measured:

- `fed_funds` carried new information on 2.6% of rows, `unemployment` 2.4%
- 29.9% of rows were exact duplicates of their predecessor
- lag-1 autocorrelation 0.998–0.9999

That flatly violates the HMM's conditional-independence assumption
($x_t \perp x_{t-1} \mid z_t$). EM absorbed the artificial persistence into the
transition matrix, producing a **degenerate 2-cycle**: two states with
effectively identical emissions ($|\Delta\mu|_{\max} = 0.001$) alternating every
single day (dwell time 1.0). Re-fitting the same model at different sampling
frequencies isolates the cause:

| Sampling | n | closest-pair $|\Delta\mu|_{\max}$ | dwell (obs) |
|---|---|---|---|
| daily ffill | 13,010 | 0.0010 — degenerate | 1.0, 1.0, 1436 |
| business-day | 9,294 | 0.0014 — degenerate | 1.0, 1.0, 1030 |
| weekly | 1,859 | 0.0039 — degenerate | 1.0, 1.0, 205 |
| **monthly** | **428** | **1.63 — healthy** | 48.7, 8.7, 60.9 |

**This is the single best "what did you learn" story in the project**: more rows
made the model strictly worse, because the extra rows carried no information but
did carry violated assumptions.

`ffill(limit=2)` is a bounded fill — it tolerates one late release without
silently carrying a stale print forward for a year. `.dropna()` then removes
the warm-up head (the first 12 months lack `cpi_yoy`/`d_unemp_12m`, and
`vix_pctile` needs ≥252 trading days).

### 4.3 The five features

`FEATURE_COLUMNS` in `argus/agents/macro_hmm.py`, in the fixed order used by both
the scaler and the HMM:

| Feature | Construction | Why this transform |
|---|---|---|
| `d_fed_funds_6m` | `FEDFUNDS.diff(6)` | 6-month change in policy rate. Differenced because the *level* is non-stationary. Captures tightening/easing cycles. |
| `d_unemp_12m` | `UNRATE.diff(12)` | Year-over-year change in unemployment. The Sahm-rule intuition: it is the *rise* in unemployment, not its level, that marks recessions. |
| `cpi_yoy` | `CPIAUCSL.pct_change(12) * 100` | Inflation rate. CPI itself is a near-monotonic index; the YoY rate is the economically meaningful, roughly stationary quantity. |
| `t10y2y` | Level | Already a spread, hence mean-reverting; differencing it would destroy the inversion signal, which is a *level* phenomenon. |
| `vix_pctile` | `rolling(1260, min_periods=252).rank(pct=True) * 100` | 5-year rolling percentile rank of VIX. Rank-transformed so the feature is scale-free and robust to VIX's fat right tail; rolling so it doesn't leak the full-sample distribution into early rows. |

**The stationarity argument is the crux.** The earlier version fed raw *levels*
(`fed_funds`, `cpi_yoy`, `t10y2y`, `unemployment`, `vix`). A Gaussian HMM
assumes each state's emission density is time-invariant, so with non-stationary
inputs the states segment **rate eras rather than regimes** — which is exactly
how the shipped model labeled Oct-2008 → May-2017 (104 months spanning the
longest bull market in US history) as CONTRACTION: it had simply discovered the
ZIRP period.

Every current feature is a difference, a ratio-of-levels, or a rolling rank —
each stationary by construction.

### 4.4 Scaling

`StandardScaler` is fit on the monthly training frame and persisted alongside
the HMM. Two points worth being precise about:

1. **Why scale at all** when a full-covariance Gaussian is affine-equivariant in
   theory? Numerics and initialisation. `vix_pctile` spans 0–100 while
   `d_fed_funds_6m` spans roughly ±5; k-means initialisation inside hmmlearn is
   Euclidean and would be dominated by the largest-variance column, and the EM
   covariance updates are better conditioned on comparable scales.
2. **The scaler is a fitted parameter, not a preprocessing detail.** It is
   pickled into the artifact and applied at inference via `.transform()` (never
   `.fit_transform()`), so serving uses the *training* mean and scale. Fitting a
   fresh scaler on the inference window would silently re-centre the data and
   invalidate every learned mean.

A subtle bug the monthly resampling also fixed: under daily forward-fill the
scaler's mean/std were weighted by *calendar days*, so a Friday print counted
three times (Fri/Sat/Sun). The standardisation itself was biased.

---

## 5. Training

### 5.1 Entry point

```bash
.venv/bin/python -m scripts.train_macro_hmm --start-date 1990-01-01
```

Offline and manual — there is no scheduled retrain. The script fits, runs the
validation gate, saves the artifact only on pass, and prints the per-state
feature-mean table as a human sanity check. Training start is 1990-01-01 so the
sample spans five NBER recessions (1990-91, 2001, 2007-09, 2020, plus the
2011 Euro-crisis stress episode) — a 2010 start would contain essentially one
regime and could not teach the model what a contraction looks like.

### 5.2 The estimator: Baum-Welch (EM)

`GaussianHMM.fit(X)` runs the Baum-Welch algorithm, the EM specialisation for
HMMs, on the single standardized sequence:

- **E-step** — forward-backward computes, for the current $\theta$, the
  posterior state marginals $\gamma_t(k) = P(z_t = k \mid x_{1:T}, \theta)$ and
  pairwise posteriors $\xi_t(j,k) = P(z_t = j, z_{t+1} = k \mid x_{1:T},\theta)$.
- **M-step** — closed-form weighted MLE:
  $\mu_k = \frac{\sum_t \gamma_t(k) x_t}{\sum_t \gamma_t(k)}$,
  $\Sigma_k$ the analogous weighted scatter, and
  $A_{jk} \propto \sum_t \xi_t(j,k)$.
- Iterate to `n_iter=300` or convergence in log-likelihood.

EM monotonically increases the observed-data log-likelihood but converges only
to a **local** optimum, which motivates the next subsection.

*(hmmlearn prints "Model is not converging" on sub-1e-4 monotonicity wobble near
the optimum. That is benign floating-point noise, not a failure.)*

### 5.3 Multi-restart selection

A single fixed seed is not a fit protocol. 3 of 5 seeds collapsed to a
degenerate solution on the original configuration — and the shipped `seed=42`
was one of them. The fit protocol is therefore:

1. Run `n_init = 20` restarts with `random_state = 0 … 19`.
2. Score each by `hmm.score(X)` (total log-likelihood).
3. Keep the highest-scoring model **that also passes the degeneracy checks**.
4. If none passes, **raise** — `fit` must fail loudly rather than leave
   `is_fitted = False` for a caller to notice later.

### 5.4 Degeneracy rejection

Two hard constraints on any accepted solution:

- **Separation.** The minimum pairwise Mahalanobis distance between state means
  under the pooled covariance must exceed **1.0**. The shipped fit scores
  **2.75**; the old degenerate artifact it replaced scored ~**0.001**.

  $$d_M(\mu_i, \mu_j) = \sqrt{(\mu_i - \mu_j)^\top \Sigma_{\text{pooled}}^{-1} (\mu_i - \mu_j)}$$

  Mahalanobis rather than Euclidean because separation must be measured in units
  of the within-state spread — two means 1.0 apart are distinguishable if states
  are tight and meaningless if they are diffuse.
- **Occupancy.** Every state must hold ≥5% of observations, so the model cannot
  "satisfy" k=3 with a state that fires twice in 36 years.

Dwell time (expected duration in state $k$ is $1/(1 - A_{kk})$) is also checked
in the gate; a 3–120 month band rejects both the 1-month 2-cycle and an
absorbing state that never leaves.

---

## 6. From hidden states to named regimes

An HMM is unsupervised: it returns states 0, 1, 2 with no meaning, and their
indices are arbitrary across runs (**label switching** — the likelihood is
invariant to permuting states). Something must attach semantics.

### 6.1 The labeling rule

1. Fetch `USREC` (NBER recession indicator) over the training window.
2. Compute each state's **recession frequency** — the fraction of months in that
   state during which `USREC = 1`.
3. **CONTRACTION** = the state with the highest recession frequency, *provided*
   it clears a minimum **enrichment ratio of 2.0×** over the sample base rate.
   If no state clears it, the run **fails**.
4. **EXPANSION** = of the remaining states, the one with the lowest
   `d_unemp_12m` (falling unemployment) — an economic criterion.
5. Everything else is **TRANSITIONAL**.
6. `state_to_regime` is cleared at the start of each fit (it was previously
   never reset, so a re-fit could inherit stale labels).

### 6.2 Why the previous rule was structurally broken

The old rule was `CONTRACTION = means["vix"].idxmax()`: whichever state had the
highest average VIX. That guarantees **exactly one state is always
CONTRACTION**, whether or not a contractionary state exists. The model cannot
express "there is no contraction in this fit" — which is precisely how the ZIRP
era acquired the label. Any argmax-over-states labeling rule has this defect;
the enrichment threshold is what turns labeling into a falsifiable claim.

### 6.3 What the shipped fit actually looks like

Read from the persisted artifact's `metadata.validation_metrics` and a Viterbi
decode (`hmm.predict`) of the full training frame:

| State | Regime | months | recession % | dwell (mo) | d_fed_funds_6m | d_unemp_12m | cpi_yoy | t10y2y | vix_pctile |
|---|---|---|---|---|---|---|---|---|---|
| 0 | TRANSITIONAL | 189 | 1.6 | 47.25 | 0.223 | −0.243 | 2.738 | 0.244 | 51.536 |
| 1 | CONTRACTION | 101 | 26.7 | 25.25 | −0.652 | 0.993 | 3.349 | 1.389 | 64.798 |
| 2 | EXPANSION | 137 | 0.0 | 45.67 | −0.017 | −0.508 | 1.981 | 1.793 | 27.176 |

State 1 → CONTRACTION: rising unemployment, easing policy (the Fed cutting
*into* the stress), elevated VIX. It is economically coherent, which matters —
a state that scored well on NBER but had incoherent feature means would suggest
overfitting to the label.

(The issue's design prototype — run before the fit/labeling/gate code in §5–7
existed — found a differently-shaped split of the same 427 months, roughly
114/122/191 across the three states. The shipped artifact's multi-restart
protocol converges on the split above instead; both are non-degenerate and
NBER-enriched, which is the reproducible property, not the exact state
boundaries.)

---

## 7. Evaluation and the validation gate

The hardest question a professor will ask is **"how do you evaluate an
unsupervised model?"** The answer here has three layers.

### 7.1 External-validity metrics (against NBER `USREC`)

| Metric | Value | Meaning |
|---|---|---|
| NBER base rate over the sample | 7.0% | Fraction of months that were recessions |
| CONTRACTION state's recession frequency | 26.7% | |
| **Enrichment** | **3.8×** | 26.7 / 7.0 — the state is 3.8× more recession-dense than chance |
| **Recall** | **90%** | Of all recession months, 90% fall inside CONTRACTION |
| Min pairwise Mahalanobis separation | 2.75 | vs 0.001 for the degenerate artifact |

Note precision is deliberately *not* high, and that is correct: CONTRACTION is a
broader "stressed macro environment" concept than NBER's official recession
dating, and the intended use is de-risking, where early and slightly wide is the
desired asymmetry.

### 7.2 Face validity — the historical trace

CONTRACTION spans from the shipped fit: 1991-02→1992-06, 2001-05→2002-10,
**2007-11→2010-05** (one month *before* the NBER-dated start of the GFC), and
2020-05→2023-03. Being asked to walk through this trace is likely; the 2007-11
entry is the strongest single data point. (The design prototype in the issue
also flagged a 2011-07→2011-12 Euro-crisis span; the shipped restart didn't
carve out a separate span for it — that period reads as EXPANSION/TRANSITIONAL
instead, which is a real difference between the two fits, not a documentation
error.)

### 7.3 The gate — the part that makes this engineering rather than a notebook

`scripts/train_macro_hmm.py` hard-fails with a non-zero exit if any of the
following is true, so a broken model cannot be committed:

1. Min state separation below threshold
2. Any state under the 5% occupancy floor
3. Dwell time outside 3–120 months
4. CONTRACTION enrichment under 2.0×
5. **Discrimination test** — a set of canned scenarios (GFC crash, COVID, 2023
   inversion, Goldilocks boom, stagflation) must yield at least two distinct
   labels

All validation metrics (per-state recession frequency, enrichment, recall,
separation, dwell times) are written into the artifact's `metadata`, so a
shipped model carries its own evidence.

The pre-fix pipeline checked only `is_fitted` — which a degenerate model passes
— and the tests fit on synthetic well-separated blobs, validating the *code
path* rather than the *model*. Check (5) alone would have caught the shipped
artifact.

### 7.4 What is honestly *not* evaluated

Be first to say this, before the examiner is:

- **No held-out or walk-forward evaluation of regime accuracy.** The model is
  fit on the full history and validated in-sample against NBER. A proper
  protocol is expanding-window walk-forward: fit on 1990→T, score month T+1,
  roll. Not done.
- **No downstream causal claim.** There is no measurement showing that
  regime-conditioned multipliers improve portfolio outcomes. `MacroParams`
  values are tagged `ARBITRARY` in `argus/params.py` precisely so this is not
  quietly forgotten.
- **`inference_window_months = 36` is not backtested** against alternatives.
- **NBER is itself a revised, judgment-based label**, so even the enrichment
  metric has a soft ground truth.

---

## 8. Persistence: the artifact

`RegimeClassifier.save()` writes a joblib payload:

```python
{"hmm": ..., "scaler": ..., "state_to_regime": ..., "is_fitted": ...,
 "metadata": {"hmmlearn_version", "sklearn_version", "feature_columns",
              "n_train_observations", "start_date", "trained_at", ...}}
```

`RegimeClassifier.load()` is defensive by contract — **it never raises to the
caller**. It validates three things and, on any failure, logs `ERROR` and
returns an *unfitted* classifier so `predict()` takes the rule-based path:

1. **Feature-column equality** against the code's `FEATURE_COLUMNS`. This is the
   check that makes a feature-set change safe: change the columns and the old
   artifact is *rejected*, rather than silently scored with misaligned columns —
   which would produce plausible-looking garbage.
2. **hmmlearn version** at major.minor.
3. **scikit-learn version** at major.minor.

Major.minor rather than exact string equality because a patch-level
`pip install -U scikit-learn` should not silently drop production into the
rule-based path; a major/minor bump can genuinely change unpickling semantics.

### 8.1 The `startprob_` repair

```python
@staticmethod
def _stationary_startprob(transmat: np.ndarray) -> np.ndarray:
    eigenvalues, eigenvectors = scipy.linalg.eig(transmat.T)
    stationary = eigenvectors[:, np.argmin(np.abs(eigenvalues - 1.0))].real
    return stationary / stationary.sum()
```

This is a favourite exam target, so know it cold. Fitting on **one** long
sequence gives EM nothing to average over for $\pi$ — the E-step's estimate of
$\pi$ is just $\gamma_1$, so $\pi$ collapses to a near one-hot pointing at
whichever state the sequence's first month happened to occupy. That is an
artifact of where the history starts, not a prior belief.

The consequence was concrete: `startprob_[CONTRACTION]` was exactly 0.0, making
CONTRACTION *unreachable* from the single-observation fallback path — that path
returned a constant TRANSITIONAL forever.

The repair replaces $\pi$ with the **stationary distribution** of the chain, the
left eigenvector of $A$ for eigenvalue 1 (i.e. $\pi A = \pi$, obtained as the
right eigenvector of $A^\top$), normalized to sum to 1. Interpretation: "absent
any other information, assume we start in a state with probability equal to the
long-run fraction of time the chain spends there." For an irreducible aperiodic
chain that distribution is unique and is the correct uninformative prior.

**It is applied in both `fit()` and `load()`, not load-only.** The first cut of
this fix only patched `startprob_` in `load()`, on the theory that only the
deserialized, production-facing classifier needed it. That missed a real
consumer: `validation_failures()`'s discrimination check calls `predict()` on
the classifier immediately after `fit()`, before any save/load round trip, and
each discrimination scenario is a *single* observation — exactly the case
`startprob_` dominates. With the correction load-only, the freshly-fit model's
raw one-hot $\pi$ swamped every scenario's emission likelihood regardless of
its features, and training runs with otherwise-passing metrics (separation
2.75, enrichment 3.8×, 90% recall) failed the gate with "discrimination
scenarios collapsed to a single label: {'CONTRACTION'}". `fit()` now calls
`self._stationary_startprob(self.hmm.transmat_)` on the winning restart right
after selecting it, so the gate evaluates the model under the same $\pi$ it
will actually ship with. `load()` keeps calling the same helper — harmless and
idempotent, since it depends only on `transmat_` — so an artifact saved before
this fix still gets a reachable CONTRACTION state.

---

## 9. Inference

### 9.1 Windowed prediction

```python
history_start = now − 365 * MACRO.inference_history_years        # 10 years
frame  = build_macro_feature_frame(self.market_data, history_start)
window = frame.tail(MACRO.inference_window_months)               # 36 months
regime, confidence = self.classifier.predict(window)
```

`predict()` then:

1. Selects `FEATURE_COLUMNS` in fixed order.
2. **Rejects NaN/inf with a `ValueError`** — otherwise a NaN propagates to
   `argmax` and silently returns state 0.
3. `scaler.transform()` (training statistics).
4. `hmm.predict_proba(X)` → the $T \times 3$ posterior matrix.
5. Takes the **final row**: `state = argmax(γ_T)`, `confidence = max(γ_T)`.

**Why a window rather than a single observation.** Scoring one row ignores $A$
entirely and reduces the HMM to a GMM — the classification becomes a nearest-
Gaussian lookup with no persistence. With a 36-month window, forward-backward
propagates evidence from 35 surrounding months through the transition matrix, so
one anomalous print cannot flip the regime. `tests/test_macro.py` encodes
exactly this as
`test_windowed_predict_resists_a_single_point_spike`: an ambiguous point decoded
alone is *not* EXPANSION, but the same point as the final row of a stable
expansion window *is*.

The 10-year history fetch is deliberately longer than the 36-month window: the
`vix_pctile` feature needs its own 5-year rolling warm-up before the first
usable row, plus margin.

### 9.2 Filtering vs smoothing — a likely trap

`predict_proba` returns **smoothed** posteriors $\gamma_t = P(z_t \mid x_{1:T})$,
which for $t < T$ use future data relative to $t$. That would be look-ahead if
those rows were used. Only the **final** row is read, and at $t = T$ smoothing
and filtering coincide ($\gamma_T \propto \alpha_T$, since the backward message
$\beta_T = \mathbf{1}$). So the reported regime uses no future information.
Being able to state that cleanly is the difference between "I called an API" and
"I understand the algorithm."

### 9.3 Viterbi vs forward-backward

`predict()` uses posteriors, not `hmm.predict()` (Viterbi). Viterbi returns the
single most likely *path*, which is what you want for a coherent historical
segmentation (and is what `_map_states` uses at fit time to assign each training
month to a state). For live classification we want the marginal at the current
month *and a probability*, which is the forward-backward output. The two can
disagree: the most likely path need not pass through the per-timestep most
likely state.

### 9.4 What `regime_confidence` is and is not

It is $\max_k P(z_T = k \mid x_{1:T}, \theta)$ — the model's posterior mass on
its own chosen state, **conditional on the model being correct**. It is *not* a
calibrated probability of being right, and it is not comparable to a supervised
classifier's confidence. It also tends to sit near 1.0 whenever states are
well-separated and the chain is persistent — high confidence is mostly a
statement about state separation, not about macroeconomic certainty. Say this
before you are asked.

`model_healthy` exists so consumers can distinguish a real posterior from the
fallback's hardcoded 0.5/0.6 — previously indistinguishable downstream.

### 9.5 Degradation ladder

| Failure | Behavior |
|---|---|
| Artifact missing / corrupt / version-mismatched | `load` logs ERROR, returns unfitted classifier → rule-based fallback, `model_healthy = False` |
| Feature-window assembly fails (FRED outage mid-run) | Logs WARNING, falls back to single-observation `predict(macro_bundle())` |
| All of `vix`, `fed_funds`, `t10y2y` None | `analyze()` returns `None`; independent specialists continue and the graph reports degradation |
| VIX missing | `vix_level` and `vix_regime` remain null; percentile computation is skipped and its neutral placeholder is logged |
| Yield spread unavailable from both window and bundle | `t10y2y` and `yield_curve_shape` remain null |
| Other descriptive macro fields unavailable | Current field resolution uses logged/neutral defaults; these remain a modeling limitation |

`macro_context.rule_based` uses a static heuristic: VIX > 30 or 10Y2Y < −0.5 →
CONTRACTION (0.6); VIX < 18 and 10Y2Y > 0.3 → EXPANSION (0.6); else TRANSITIONAL
(0.5). Missing inputs are unknown: observed stress can trigger CONTRACTION,
but EXPANSION requires both a measured low VIX and a measured positive spread.
Fixed confidences signal lower certainty by construction.

### 9.6 Which `MacroContext` fields come from the window

Post-fix, `fed_funds`, `t10y2y`, `cpi_yoy`, and `unemployment` are read from the
window's **final row** (`window_final`), not from an independently-fetched
`macro_bundle()`. Two separate fetches can disagree, and the reported context
must describe the same data the classifier actually scored.

The VIX path stays **daily**: `vix_level`, `vix_percentile`, `vix_regime`, and
therefore `agent_multipliers` are computed from a fresh daily VIX pull. That is
deliberate — the regime moves monthly, but volatility responsiveness belongs on
a daily clock. Note this means the node carries *two* VIX statistics with
different windows: the model feature `vix_pctile` (5-year rolling rank, monthly)
and the reported `vix_percentile` (2-year daily distribution). Do not conflate
them if asked.

---

## 10. Downstream consumption

1. **`agent_multipliers`** — scalars applied to specialist convictions in
   `orchestration/aggregator.py`. Currently derived from the regime *and the VIX
   percentile*, e.g. `fundamental = 1.3` under EXPANSION with VIX percentile
   < 40, else 0.9. All `ARBITRARY`-tagged.
2. **Contraction override** — under `Regime.CONTRACTION` the aggregator
   suppresses low-conviction BULLISH signals (below the 0.70 threshold). This is
   the sharpest edge: a mislabeled regime silently changes every allocation.
3. **Cultural memory** — every stored decision is tagged with its regime, and
   retrieval filters on it. When all 207 stored rows read EXPANSION, the field
   carried literally zero information, and regime-scoped retrieval degenerated
   to unscoped retrieval. Rows written under the degenerate model were
   remediated to `unknown`, with the retrieval filter skipping the regime clause
   for that value.
4. **Portfolio prompt** — the regime and its confidence appear in the LLM
   context for allocation reasoning.
5. **API** — surfaced at `/analyze` and, as `macro_classifier_fitted`, at
   `/pipeline/status`.

---

## 11. Testing strategy

`tests/test_macro.py` covers both the code path and the model properties:

- `test_rule_based_fallback` — unfitted classifier takes the heuristic
- `test_save_load_round_trip` — labels and predictions survive serialisation
- `test_load_missing_file` / `test_load_corrupt_file` / `test_load_version_mismatch`
  — every failure mode returns unfitted rather than raising
- `test_windowed_predict_classifies_a_stable_window_consistently`
- `test_windowed_predict_resists_a_single_point_spike` — the property windowed
  inference exists to provide
- `test_analyze_falls_back_to_single_observation_when_window_assembly_fails`
- **Window-length invariance** — the same constant window at lengths $n$ and
  $n+1$ must return the same regime (the direct regression test for the parity
  bug, where `tail(89)` and `tail(90)` on the same data returned TRANSITIONAL
  and EXPANSION)
- **Discrimination** — crash-like and boom-like windows must not share a label
- **CONTRACTION reachable from a single observation** — guards the `startprob_`
  fix
- **Degeneracy rejection** — duplicated-cluster data must make `fit` raise
- **Point-in-time** — no feature row may contain information published after
  that row's own timestamp
- **Train/serve parity** — `build_macro_feature_frame` must produce identical
  frames for the training and inference call sites against the same
  `FixtureMarketDataProvider`

Synthetic fits use well-separated Gaussian blocks (not shuffled, so the HMM also
learns transition structure) with an explicit `random_state=2`, because EM's
local-optimum sensitivity means not every seed separates small synthetic data.
Note the honest limitation: synthetic-blob tests validate the *code*, not the
*model* — which is why the training gate exists as a separate mechanism.

---

## 12. Anticipated questions and answers

### A. Conceptual — model theory

**Q1. What is a hidden Markov model, in one paragraph?**
A generative model for sequences with an unobserved discrete state. The state
evolves as a first-order Markov chain — the future is conditionally independent
of the past given the present state — and each observation is drawn from a
state-specific emission distribution, here a multivariate Gaussian. Given
observations we infer a distribution over the hidden states rather than
observing them, which is what makes "regime" a learned latent construct rather
than a hand-drawn threshold.

**Q2. What are the three canonical HMM problems and which do you solve?**
Evaluation ($P(x_{1:T}\mid\theta)$, via forward), decoding (most likely states,
via Viterbi or per-timestep forward-backward), and learning ($\theta$ from data,
via Baum-Welch/EM). I solve learning offline in training and decoding online in
inference; evaluation is used implicitly as the restart-selection score.

**Q3. Walk me through the E and M steps.**
E-step: with $\theta$ fixed, forward-backward gives $\gamma_t(k)$ and
$\xi_t(j,k)$. Forward recursion $\alpha_t(k) = b_k(x_t)\sum_j \alpha_{t-1}(j)A_{jk}$;
backward $\beta_t(j) = \sum_k A_{jk} b_k(x_{t+1})\beta_{t+1}(k)$;
$\gamma_t \propto \alpha_t\beta_t$. M-step: weighted MLE —
means and covariances are $\gamma$-weighted sample moments, and transition rows
are normalized $\xi$ sums. Repeat; log-likelihood is non-decreasing each
iteration.

**Q4. Why does EM only find a local optimum, and what did you do about it?**
The complete-data likelihood is non-convex in $\theta$; EM ascends a lower bound
and converges to a stationary point that depends on initialisation. Concretely,
3 of 5 seeds collapsed to a degenerate solution — the shipped `seed=42` among
them. Mitigation: 20 restarts, select by log-likelihood *subject to* passing the
degeneracy checks. Note the ordering matters — highest likelihood alone would
have kept a degenerate model, because a 2-cycle over near-identical states can
*out-score* an honest fit on autocorrelated data.

**Q5. What assumptions does the model make, and which are violated?**
(i) First-order Markov states — recessions have duration dependence, so a
geometric dwell distribution is an approximation; a hidden semi-Markov model
would relax it. (ii) Conditional independence of observations given the state —
violated badly under daily forward-fill (lag-1 autocorrelation 0.999), which is
what produced the degenerate 2-cycle; monthly sampling with differenced features
substantially repairs it, though monthly macro data is still somewhat
autocorrelated within regime. (iii) Gaussian emissions — macro features have
fat tails and skew, so this is a working approximation, and full covariance is
doing real work absorbing correlation. (iv) Time-invariant parameters — the
economy's structure changed between 1990 and 2026 (ZIRP, QE), so parameter
stationarity is an assumption I cannot defend strongly.

**Q6. Why not just cluster with a GMM?**
A GMM is exactly this model with $A$ removed and every row equal to the
stationary distribution. It discards persistence, so labels flicker month to
month, and it gives no mechanism by which surrounding months damp a noisy print
— which is the property `test_windowed_predict_resists_a_single_point_spike`
verifies.

**Q7. How did you choose $k = 3$?**
Product constraint: the downstream `Regime` enum is 3-valued and consumed by the
aggregator, memory tagger, and portfolio prompt. BIC doesn't even cleanly agree
with itself on this sample — k=2 → 4533.6, k=3 → 4795.5, k=4 → 3839.6, k=5 →
3903.0, so k=3 actually scores *worse* than k=2 before dropping sharply at k=4
— so I logged BIC for k ∈ {2..5} as a visible diagnostic and chose
interpretability knowingly rather than chasing a criterion that isn't even
monotonic here. k=3 already delivers 3.8× NBER enrichment; k>3 would force an
arbitrary many-to-3 collapse for no validated gain.

**Q8. Label switching — how do you know state 1 is "the contraction state"?**
I don't, a priori: the likelihood is permutation-invariant, so indices are
meaningless across runs. Labels are assigned *after* fitting by an external
criterion — recession frequency against NBER `USREC` — with a hard 2.0×
enrichment requirement. If no state clears it the training run fails rather than
shipping an arbitrary assignment.

**Q9. Isn't using NBER labels supervision by the back door?**
It is supervision at the *labeling and validation* stage, not the *fitting*
stage — the parameters are estimated with no reference to `USREC`. That
distinction matters practically: NBER dating is published 6–18 months late and
is revised, so it cannot appear in a real-time feature, but it is legitimate for
offline naming and evaluation of an already-fitted model. The stronger critique
is that I select among 20 restarts partly on a criterion informed by NBER, which
does leak a little; a cleaner protocol would hold out the last two recessions
from the labeling decision.

**Q10. Why is stationarity so important here?**
A Gaussian HMM assumes each state's emission density is fixed in time. Feed it
non-stationary levels and the states will partition the *time axis* by level
rather than by economic condition. That is not hypothetical: with level features
the model labeled Oct-2008→May-2017 as CONTRACTION — it had discovered the ZIRP
rate era, and the label happened to span the longest bull market in US history.

**Q11. Why Mahalanobis and not Euclidean distance for the separation check?**
Because separation is only meaningful relative to within-state spread. Two means
1.0 apart are cleanly separated if the states are tight and completely
overlapping if they are diffuse. Mahalanobis normalizes by the pooled
covariance, so the threshold (1.0) means the same thing regardless of feature
scaling — which is also why the check survives a change of units.

**Q12. What does a dwell time of 1.0 tell you?**
$1/(1 - A_{kk}) = 1$ implies $A_{kk} = 0$ — the chain leaves the state every
single step. Combined with two states having near-identical means, that is the
signature of a 2-cycle: EM used two states to model one distribution and spent
the transition matrix alternating between them. The observable symptom was that
the reported regime depended on window-length *parity*: `tail(89)` and
`tail(90)` on identical data returned TRANSITIONAL and EXPANSION.

### B. Conceptual — data and evaluation

**Q13. How do you validate an unsupervised model?**
Three layers, none sufficient alone. (1) *Internal* — is the solution
non-degenerate? Separation ≥ 1.0 Mahalanobis, ≥5% occupancy per state, dwell in
3–120 months. (2) *External* — does it align with an independent label it never
saw? 3.8× enrichment over the 7.0% NBER base rate, 90% recall. (3) *Face
validity* — do the state means read as economically coherent, and do the
contraction spans line up with known history? They do, including a 2007-11
entry one month ahead of the NBER-dated GFC start. Plus a discrimination test:
canned crash/boom/stagflation scenarios must not all get the same label.

**Q14. Why is your precision so low — 22% recession frequency inside
CONTRACTION?**
Because the state is intentionally broader than NBER's definition. It also
captures the 2011 Euro crisis and the 2022 tightening, which are not US
recessions but are macro environments where de-risking is the correct action.
The asymmetry matches the use: recall on genuine downturns matters far more than
precision, since the downstream cost of a false CONTRACTION is reduced exposure
while the cost of a miss is full exposure into a drawdown.

**Q15. Where is your held-out test set?**
There isn't one, and that is the largest honest gap. The model is fit on the
full 1990→present history and evaluated in-sample against NBER. The correct
protocol is expanding-window walk-forward — fit on 1990→T, score T+1, roll —
which would also let me measure whether regime *transitions* are detected
promptly rather than only whether states are enriched. That is the first thing
I would add.

**Q16. Explain the look-ahead bias you found and fixed.**
FRED stamps monthly series by reference date, not release date. So training saw,
for example, the April 2020 unemployment print on a row dated 2020-04-15, when a
real-time observer knew only the March figure. The gap was material:
`cpi_yoy` differed on 99.5% of rows, `unemployment` on 74.0%, with a maximum
error of 11.3pp (UNRATE 14.8 vs the 4.4 actually known). The fix shifts each
monthly index forward by its published BLS/Fed release lag — 32/35/45 days,
recorded in `params.py` with `Provenance.LITERATURE`. Market series (T10Y2Y,
VIX) are same-day and are not shifted.

**Q17. Why did *reducing* your dataset from 13,010 rows to 428 improve the
model?**
Because 12,500 of those rows were duplicated information, not evidence. Three of
five features are natively monthly; forward-filling to daily produced 29.9%
exactly-duplicate rows and lag-1 autocorrelation of 0.998–0.9999, violating
conditional independence. EM absorbed that artificial persistence into a
degenerate 2-cycle. Re-fitting the identical model at daily, business-day,
weekly, and monthly frequency isolates it cleanly: only monthly produces
separated states (1.63 vs ~0.001). The lesson is that sample size is about
*independent information*, not row count.

**Q18. Is your training sample large enough for 68 parameters?**
It is thin — roughly 6 observations per parameter, and the observations are not
independent. Three things keep it defensible: the model is heavily structured
(not a free 428×5 fit), the degeneracy gate rejects the specific failure mode
over-parameterisation produces here, and the fitted result generalises in the
sense of aligning with recessions it was never shown. If I needed more headroom
the first move is `covariance_type="tied"`, which drops 30 parameters, and I'd
compare BIC and separation before and after.

**Q19. Are your features leaking into each other?**
They're correlated by construction — a Fed easing cycle co-occurs with rising
unemployment and an elevated VIX percentile — which is exactly why the model
uses full rather than diagonal covariance. No feature is a deterministic
function of another. The one design smell is that `t10y2y` appears both as a
model feature and as the driver of the deterministic `yield_curve_shape` field,
so those two outputs are not independent evidence downstream.

**Q20. What would you do differently with more time?**
In order: walk-forward validation; a hidden semi-Markov variant to relax the
geometric dwell assumption; `covariance_type="tied"` compared under BIC; a
Student-t emission for fat tails; and — most importantly — an actual measurement
of whether regime-conditioned multipliers improve realised outcomes, since
today they are `ARBITRARY`-tagged with no evidence behind them.

### C. Implementation-specific

**Q21. Why does feature construction live in its own module?**
Because training and inference previously built features through two independent
code paths with nothing asserting they agreed, which is the classic train/serve
skew setup. `build_macro_feature_frame` is now the sole construction, called by
both `fit_on_history()` and `analyze()`, through the same
`MarketDataProvider` seam — and there is a parity test asserting both call sites
produce identical frames from the same fixture provider.

**Q22. Why is the VIX fetch period hardcoded to `"max"`?**
So the 5-year rolling percentile is computed against the same reference
distribution in fit and serve. A caller-supplied period would let training use
full history and inference use two years, silently changing what `vix_pctile`
means between the two.

**Q23. Why does `load()` never raise?**
Because the macro node is a hard dependency of the whole DAG, and an unloadable
artifact should degrade the *quality* of the analysis, not take the service
down. `load` returns an unfitted classifier, `predict` takes the documented
rule-based path, and `model_healthy = False` propagates that fact into every
consumer so nobody mistakes a 0.6 heuristic for a posterior.

**Q24. Why check feature columns in the artifact metadata?**
It is the guard against the worst class of ML bug: silent misalignment. If I
change `FEATURE_COLUMNS` and an old artifact is loaded, the scaler and Gaussians
would be applied to a different set of columns and return confident nonsense. An
explicit equality check converts that into a loud rejection — and it is precisely
what correctly rejects the old level-feature artifact after the feature rewrite.

**Q25. Why major.minor for the library version gate rather than exact match?**
Exact matching meant a patch bump from `pip install -U scikit-learn` silently
dropped production to the rule-based path — a false positive with real cost.
Major.minor still catches the changes that plausibly break unpickling or alter
estimator internals. It's a deliberate sensitivity/specificity trade on a
gate whose false-negative cost is bounded by the other checks.

**Q26. Explain the stationary-distribution fix in one breath.**
Fitting one sequence makes EM's estimate of $\pi$ equal to $\gamma_1$, so $\pi$
becomes a one-hot on whatever state month one occupied — an artifact of the
history's start date. That set $\pi[\text{CONTRACTION}] = 0$ exactly, making
CONTRACTION unreachable from single-observation inference. I replace $\pi$ with
the chain's stationary distribution, the left eigenvector of $A$ for eigenvalue
1, which is the correct uninformative prior for "we joined the process at an
arbitrary time" — applied right after `fit()` selects its winning restart, and
again (idempotently) in `load()`. It has to happen at fit time too: the
training gate's discrimination check calls `predict()` on the in-memory,
not-yet-saved classifier, and a load-only fix left that check exposed to the
same one-hot artifact it exists to remove — which is exactly the failure mode
that first surfaced it, on an otherwise-passing retrain.

**Q27. Why 36 months for the inference window?**
Long enough that the transition matrix has real evidence to work with and one
print cannot flip the call; short enough that a genuine pivot still registers.
It is tagged `ARBITRARY` in `params.py` and has *not* been backtested against
alternatives — which is exactly the disclosure the provenance system exists to
force. The prior implementation had a worse problem than an unvalidated value:
because the model was degenerate, changing the window by one observation flipped
every EXPANSION in the system.

**Q28. Does `predict_proba` leak future information?**
It returns smoothed posteriors, which for interior timesteps do condition on
later observations. Only the final row is used, and at $t = T$ the backward
message is all-ones so smoothing and filtering coincide — $\gamma_T \propto
\alpha_T$. So no future information enters the reported regime. If I ever
started reading interior rows — for a historical trace, say — I would need
filtered probabilities instead.

**Q29. Why forward-backward posteriors instead of Viterbi at inference?**
I need a probability, and I need the marginal at *now*, not a globally coherent
path. Viterbi returns the single best path with no per-timestep uncertainty, and
the two can genuinely disagree — the most likely path need not pass through each
step's most likely state. Viterbi is the right tool at fit time, where
`_map_states` assigns each training month to a state for the labeling
statistics, and it is what I'd use to render a historical regime trace.

**Q30. Why does `predict` reject NaN/inf explicitly?**
Because `numpy.argmax` on an array containing NaN returns a position rather than
erroring, so a missing feature would have silently produced a confident regime
call. Failing loudly routes the caller into the documented single-observation
fallback instead of fabricating an answer.

**Q31. Why is the model a committed artifact rather than fit at startup?**
Three entry points construct the agent (API, unattended collector, replay). If
each fit at boot they would produce different models from different data
vintages, and a boot would take minutes and require network plus a FRED key.
Committing one artifact makes the model a reviewable, versioned dependency and
makes every process's regime call identical.

**Q32. Why joblib rather than a JSON of parameters?**
Convenience — joblib handles the numpy-heavy sklearn/hmmlearn objects directly.
The cost is a pickle: it is version-fragile (hence the version gate) and
executes on load, so it is only safe because the file is repo-committed and
trusted. A parameter-only serialisation (means, covariances, transmat, scaler
stats as plain arrays) would be strictly more robust, and is the right move if
this ever loaded artifacts from an untrusted or external source.

**Q33. Why a 6-hour cache TTL on `MacroContext`?**
The regime updates monthly, so caching the model output for hours costs
essentially nothing and avoids re-fetching a decade of FRED history on every
request. The real cost is that the daily-responsive VIX fields go stale within
the window, which is a known trade rather than an oversight.

**Q34. What breaks downstream if the regime is wrong?**
Directly: the aggregator suppresses low-conviction BULLISH signals under
CONTRACTION, so a false CONTRACTION cuts exposure across the board and a missed
one leaves it on. Indirectly: `agent_multipliers` reweight all three specialists,
and every stored decision is tagged with the regime, so a stuck regime poisons
cultural memory. That last one was observed — 207/207 stored rows read
EXPANSION, making the field pure noise and regime-scoped retrieval equivalent to
unscoped retrieval. Those rows were remediated to `unknown` and the retrieval
filter skips the clause for that value.

**Q35. What single check would have caught the original broken model?**
The discrimination test in the training gate: run a Goldilocks boom and a GFC
crash through the classifier and require at least two distinct labels. The
shipped artifact called a VIX-11, 1% Fed funds, 3.8% unemployment economy
CONTRACTION with confidence 1.000 and the deep 2023 curve inversion EXPANSION.
Five lines of assertion would have blocked it. The general lesson: the pipeline
checked `is_fitted`, which is a statement about the *code* having run, not about
the *model* being usable — and every ML pipeline needs at least one gate that
tests the model's behavior rather than its existence.

---

## 13. Numbers worth memorizing

| Quantity | Value |
|---|---|
| States / features / free parameters | 3 / 5 / 68 |
| Training sample (monthly, 1990→present) | 427 observations |
| Inference window / history fetch | 36 months / 10 years |
| Publication lags (FEDFUNDS / UNRATE / CPIAUCSL) | 32 / 35 / 45 days |
| VIX percentile window | 1260 trading days (~5y), min 252 |
| Restarts / separation floor / occupancy floor | 20 / 1.0 Mahalanobis / 5% |
| Dwell-time band | 3–120 months |
| NBER base rate / CONTRACTION frequency / enrichment / recall | 7.0% / 26.7% / 3.8× / 90% |
| Achieved min separation (vs degenerate artifact) | 2.75 vs ~0.001 |
| BIC, k = 2 / 3 / 4 / 5 (single-seed diagnostic, non-monotonic) | 4534 / 4796 / 3840 / 3903 |
| Daily-ffill row count that caused the failure | 13,010 (29.9% duplicates) |

## 14. Glossary

- **Baum-Welch** — the EM algorithm specialised to HMMs.
- **Forward-backward** — the E-step; yields smoothed state posteriors $\gamma_t$.
- **Viterbi** — dynamic program for the single most likely state *path*.
- **Filtering vs smoothing** — $P(z_t \mid x_{1:t})$ vs $P(z_t \mid x_{1:T})$;
  equal at $t = T$.
- **Dwell time** — expected duration in state $k$: $1/(1 - A_{kk})$.
- **Stationary distribution** — $\pi$ with $\pi A = \pi$; the chain's long-run
  state occupancy.
- **Label switching** — the likelihood's invariance to permuting state indices;
  why external labeling is required.
- **Enrichment** — a state's positive-label rate divided by the sample base rate.
- **Point-in-time data** — data indexed so each row contains only information
  publicly available as of that row's timestamp.
- **Train/serve skew** — features computed differently at training and inference
  time.

---

## Appendix — implementation status

All six stages of issue #10 are landed on `main`: Stage 1 (shared point-in-time
feature construction, `argus/data/macro_features.py`), Stage 2 (windowed
inference, stationary `startprob_`, `model_healthy`, major.minor version gate,
NaN rejection, window-final `MacroContext` population), Stages 3–5 (multi-restart
fit protocol with degeneracy rejection, the NBER-based labeling rule, the
`scripts/train_macro_hmm.py` validation gate, and the corresponding tests), and
Stage 6 (the retrained artifact and the cultural-memory remediation). §5–§7
describe shipped code and the actual persisted `argus/models/macro_hmm.joblib`,
not a design spec — the numbers in §6.3, §7.1, §7.2, and §13 are read from that
artifact's `metadata.validation_metrics`, not the issue's pre-implementation
design prototype. The fit protocol's multi-restart search is stochastic in the
sense that it explores 20 seeds and keeps the best; a re-run of
`scripts.train_macro_hmm` is not guaranteed to reproduce the exact state
boundaries in §6.3 (though it must still pass the same gate), so treat those
figures as illustrative of *this* artifact's `trained_at` rather than eternal
constants — reconfirm against the current artifact's metadata before an
interview if it has been retrained since.
