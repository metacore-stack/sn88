# SN88 US Equity Portfolio Management System — Development Guide

**Revision 4 · 2026-09-01.** A build specification for an automated portfolio asset management
system trading **US stocks** on SN88. Written for the US-equity class only: every constraint,
clock, cost and target below is the stock-class one. The alpha/staking class is out of scope.

**What you are accepting by choosing US equities.** The stock half is the crowded half — **151
miners compete for it against 75 on the other side, for the same 50% of emission.** In exchange you
get a 555-name universe with real fundamental data, mature tooling, no AMM price-impact modelling,
and a game you can actually reason about. It is a defensible choice; just go in knowing the field
is twice as deep and plan for a top-20 finish (§1.3).

### What changed in r4, and why it matters

Revision 3 said *"maximise Sharpe, then run at the maximum feasible scale."* That instruction was
derived from a synthetic experiment that does not correspond to any decision a long-only miner can
actually make, and it is **withdrawn**. Four corrections:

1. **The scale identity is narrower than r3 claimed.** `lsr` and `odds` are exactly invariant under
   rescaling; `mar` is **not** — `gain` is compounded while `risk` is an arithmetic cumsum, so the
   ratio drifts (§2.2). More importantly, **you cannot rescale a fixed return path.** Raising
   portfolio volatility means holding different assets, which changes Sharpe, skew, correlation,
   turnover and crash exposure simultaneously. The elasticity is not a sizing rule.
2. **The live "winners run 21.8% vol" observation is confounded**, not causal (§1.3). Winners have
   positive returns by construction; the score mechanically ranks them up. `R² = 0.36` leaves most
   cross-sectional variation unexplained.
3. **The Monte Carlo behind r3 was i.i.d. Gaussian**, so it understates exactly the tail risk that
   the inception gate punishes: gaps, volatility clustering, regime shifts, correlations going to
   one (§11.2).
4. **Retrospective candidate scoring was a look-ahead bug.** Replaying today's candidate weights
   over the past 50 sessions evaluates a portfolio on returns that were already known when it was
   generated. §13 now specifies three separate evaluation modes.

The replacement policy is **maximise expected future incentive subject to survival**, not maximise
scale. The scoring analysis in Part I is unchanged and remains the core asset; what changed is the
line between *knowing the objective* and *gaming it*.

> **Evidence base.** `mobiusfund/investing` at `main`, read 2026-09-01: `README.md`,
> `Investing/doc/Subnet88.md`, `Investing/core/{const,etc,simst,api}.py`,
> `Investing/strat/README.md`, `neurons/{miner,validator}.py`, `min_compute.yml`.
> Live owner API `http://api.investing88.ai/{assets,ratio,days,dist,pnl,score}` — 151 US-stock
> UIDs, full P&L history, complete 555-ticker universe. Chain state by JSON-RPC at block ~8,974,700.
> Simulation figures are Monte Carlo over a re-implementation of `simst.py score()` / `pl2sc()`,
> 500–3,000 paths per point. The replica reproduces the live `/score` table to <1e-6 relative on
> all rows and the full pipeline reproduces on-chain incentive to 0.01pp.

---

## 0. Orientation

### 0.1 What you are building

A closed-loop system that emits **one Python dict of US equity weights** on a schedule, and
maximises a published, fully-knowable objective. Unusually for quantitative finance:

- **The objective function is open source, but the test set is not.** You can compute your own
  score, and everyone else's, before the validators do — the *formula* is public. The **market path
  it will be evaluated on is unknown**, which makes this an online forecasting competition, not a
  static benchmark. Optimising the formula against known history is the failure mode this whole
  document is trying to prevent.
- **The deliverable is ~200 bytes.** No model to serve, no latency budget, no GPU.
- **The entire field is observable.** Every rival's full daily equity curve, cash level, rebalance
  recency and pairwise portfolio distance is a public HTTP GET.
- **There is no real market impact.** Your book is simulated against $10M of notional at session
  prices. Liquidity modelling matters far less here than in a real fund; costs are a flat 0.2%.
- **Feedback is daily and permanent.** The day you first submit starts clocks you cannot restart.

The highest-value component is therefore **not** the alpha model. It is an exact replica of the
scoring pipeline used as the candidate ranker. Build that first (§24).

### 0.2 Evidence tags

| Tag | Means |
|---|---|
| `[CODE]` | Read directly out of the repository at `main`. |
| `[LIVE]` | Pulled from the owner API or the chain on 2026-09-01. **Will drift.** |
| `[MEASURED]` | Monte Carlo over the real scoring code. Reproducible — Appendix B. |
| `[INFERRED]` | Reasoning from code plus observation. Stated as inference. |
| `[DOC]` | The subnet's own README or Subnet88.md. |

### 0.3 Before you use any number

1. **Every price, rank and headcount is a snapshot.** The class ratio, the miner pool, the scoring
   constants and the similarity threshold are owner-controlled and unversioned.
2. **Validators auto-update hourly.** `etc.update()` runs `git pull && pip install -e .` and exits
   for pm2 to restart. Constants can change under you within the hour, unannounced. `[CODE]`
3. **`simst` is not the validator.** It applies the window, the clip and the ramp — nothing else.
   Four multipliers and one hard gate live only in `etc.py`. Build against §3. `[CODE]`
4. **This is a tournament objective, not a mandate.** It truncates your right tail, taxes cash
   linearly, and gates you on an absolute floor. The score-optimal portfolio is not the portfolio
   you would run with your own money. Read §2.8 before committing real capital to the same strategy.

### Contents

**Part I — the target**
[1. The prize and the field](#1-the-prize-and-the-field) ·
[2. The objective function](#2-the-objective-function) ·
[3. Every multiplier](#3-every-multiplier-in-order) ·
[4. Nine ways to score zero](#4-nine-ways-to-score-zero) ·
[5. The submission contract](#5-the-submission-contract) ·
[6. The universe](#6-the-universe)

**Part II — the system**
[7. Overview and layout](#7-system-overview) ·
[8. Data plane](#8-module-1--data-plane) ·
[9. Feature store](#9-module-2--feature-store) ·
[10. Forecast ensemble](#10-module-3--forecast-ensemble) ·
[11. Risk engine](#11-module-4--risk-engine) ·
[12. Candidate generator](#12-module-5--candidate-generator) ·
[13. Score replica](#13-module-6--the-score-replica) ·
[14. Selector](#14-module-7--selector) ·
[15. Risk governor](#15-module-8--risk-governor) ·
[16. Publisher](#16-module-9--publisher) ·
[17. Field intelligence](#17-module-10--field-intelligence) ·
[18. Observability](#18-module-11--observability)

**Part III — correctness**
[19. Research protocol](#19-research-protocol) ·
[20. Test plan](#20-test-plan) ·
[21. Failure matrix](#21-failure-matrix)

**Part IV — operating it**
[22. The daily cycle](#22-the-daily-cycle) ·
[23. Runbook](#23-runbook) ·
[24. Build order](#24-build-order) ·
[25. What not to build](#25-what-not-to-build) ·
[26. Risks](#26-risks-you-are-actually-taking)

**Part V — governance and survival**
[27. Survival & deregistration](#27-survival-and-deregistration-model) ·
[28. Economic viability](#28-economic-viability) ·
[29. Security](#29-security-architecture) ·
[30. Data contract](#30-the-data-contract) ·
[31. Model lifecycle](#31-model-lifecycle-governance) ·
[32. Attribution](#32-performance-attribution) ·
[33. API integrity](#33-api-integrity-and-adversarial-inputs)

**Appendices** — [A constants](#appendix-a--constants) · [B the replica](#appendix-b--the-score-replica-reference-implementation) · [C config](#appendix-c--configuration-template) · [D reading list](#appendix-d--reading-list) · [E hardware](#appendix-e--hardware-and-infrastructure)

---

# Part I — The target

## 1. The prize and the field

### 1.1 How you are paid

SN88 mints exactly **1.000 alpha per block** to participants — 7,200/day — split 18% owner /
41% miners / 41% validators-and-stakers. The miner pool is **2,952 alpha/day**, structural rather
than a snapshot. Tempo is **360 blocks ≈ 72 minutes**, so 20 settlements per day. `[LIVE]`

```
your alpha/day ≈ 2952 × 0.50 (the US-stock share) × (your share of the stock class)
```

Emission arrives as **alpha stake on your hotkey** — there is no liquid-versus-staked choice.
Realising TAO means unstaking through the pool and paying swap fee plus slippage.

Entry is 256 alpha (192 staked + 64 paid, **per coldkey**) ≈ 1.0 TAO ≈ $222, plus a `0.0100 TAO`
registration burn.

### 1.2 The stock class today `[LIVE]`

| | US stocks |
|---|---|
| Emission share | 50% of the miner pool = **1,476 alpha/day ≈ $1,280/day** |
| Registered in class | 151 |
| Scoring above zero | **92** |
| Top 10 take | **68.8%** of the class |
| Median cash | 1.0% |
| Median days since rebalance | 4 |

### 1.3 What it takes — the empirical target `[LIVE]`

This is the most useful table in the document. Live US-stock miners, grouped by rank, with
volatility and Sharpe computed from their own published 50-day equity curves:

| Cohort | n | Median ann. vol | Median ann. Sharpe | Median cash | Median days idle |
|---|---:|---:|---:|---:|---:|
| **Top 10** | 10 | **21.8%** | **5.95** | 3.3% | 2 |
| Rank 11–25 | 15 | 15.0% | 5.23 | 0.9% | 2 |
| Rank 26–50 | 24 | 13.3% | 3.38 | 1.8% | 5 |
| Rank 51+ | 36 | 14.9% | 2.48 | 1.0% | 4 |
| **Scoring zero** | 44 | 13.4% | **−2.38** | 1.9% | 5 |

Read it as **descriptive, not causal.** Three observations and one warning.

- **Sharpe is the separator.** It falls monotonically with rank — 5.95 → 5.23 → 3.38 → 2.48 →
  negative. The zero-score cohort is not badly calibrated; it is simply losing money.
- **The refresh cadence is a tell.** Top cohorts sit at 2 days idle, the tail at 4–5. §3.4.
- **Cash discipline is near-universal** among earners: 1–3% across every cohort.

> ⚠️ **The volatility column does not license raising your own volatility.** Top-10 vol is 21.8%
> against 13–15% elsewhere, and r3 read that as proof that scale is rewarded. It is not proof. The
> sample is confounded in at least four ways: winners have positive returns *by construction* and
> the score mechanically ranks high-return books up; 50 sessions is a very small sample; the
> leaderboard is survivor-selected; and rivals may share common market exposures, so a single lucky
> beta regime lifts the whole high-vol cohort together. The cross-sectional regression in §2.3 has
> `R² = 0.36` — **most of the variation in score is unexplained by volatility and Sharpe together.**
> Use this table to calibrate what a competitive book looks like, not to choose your risk level.
> §12 sets risk from a survival-constrained objective instead.

Distribution across the 92 earning stock miners: annualised volatility p10 9.8%, median 15.0%,
p90 32.1%. Annualised Sharpe p10 1.63, median 3.48, p90 6.05. (These Sharpes are inflated by the
50-day window and by simulation without market impact — treat them as *relative* targets only.)

**The top of the board, for calibration** `[LIVE]`:

| Rank | 50-day return | Max DD | mar | lsr | odds | Ann. vol | Ann. Sharpe | Raw score |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 16.64% | 2.01% | 8.28 | 0.615 | 75.6 | 36.2% | 5.99 | 198.1 |
| 2 | 11.45% | 1.11% | 10.31 | 0.514 | 70.0 | 10.9% | 7.29 | 111.9 |
| 3 | 13.26% | 2.05% | 6.48 | 0.551 | 73.8 | 27.0% | 5.51 | 109.5 |
| 5 | 13.92% | 2.00% | 6.96 | 0.474 | 71.1 | 21.8% | 6.13 | 98.9 |
| 10 | 19.74% | 6.55% | 3.02 | 0.322 | 66.0 | 63.9% | 4.37 | 38.7 |

**The shape to aim for: roughly 12–17% over 50 sessions with a 1–2% maximum drawdown.** Note rank 2
achieves it at 10.9% volatility and rank 10 at 63.9% — there are multiple routes, and Sharpe is
what they have in common.

## 2. The objective function

### 2.1 The formula, exactly `[CODE]`

`Investing/core/simst.py`, `score()`:

```python
gain  = max(swap - init, -init) / init * 100       # % over the window (compounded equity)
risk  = drawdown(dd['pnl%'])                       # max DD of the cumsum of daily pnl%
daily = ((1 + gain / 100) ** (1 / days) - 1) * 100
mar   = gain / max(risk, risk_init / days ** 0.5)  # risk_init = 5
lsr   = dd['pnl'].sum() / (dd['pnl'].abs().sum() or 1e18)
odds  = 50 + kelly(prob, pavg / lavg) / 2 * 100    # kelly(p,b) = (p(b+1)-1)/b
if odds <= 0: odds = 0
if np.isnan(odds): odds = prob * 100               # fires when there are no losing days
score = mar * lsr * odds * daily
if score <= 0: score = 0
```

**The window is 50 TRADING SESSIONS, not 50 calendar days.** `[LIVE]` The `/pnl` feed emits one row
per session for the stock class — the longest-lived stock UID has 107 rows over 155 calendar days
with zero weekend rows. So:

| Clock | Counted in | Real elapsed time |
|---|---|---|
| Scoring window (50) | trading sessions | **~10 calendar weeks** |
| New-miner ramp (30) | trading sessions | **~6 calendar weeks** |
| **Inactivity (`last`)** | **calendar days** | — |

**That asymmetry is a trap.** Your score accrues on a trading-day clock while your inactivity
penalty ticks on a calendar clock, including weekends and holidays. A Friday submission is already
2 days idle by Sunday night having earned no new P&L. Plan the refresh schedule around calendar
days (§3.4), not around sessions.

### 2.2 The scale identity — what is exact, what is not, and what it does not license `[MEASURED]`

Multiply every daily return in a **fixed** path by a constant `k`, and run it through a real
compounded equity curve (as `pl2sc` does — this matters, see below):

| k | 0.25 | 0.5 | **1.0** | 2.0 | 4.0 | 8.0 |
|---|---:|---:|---:|---:|---:|---:|
| `risk / k` | 4.0224 | 4.0224 | **4.0224** | 4.0224 | 4.0224 | 4.0224 |
| `odds` | 61.9323 | 61.9323 | **61.9323** | 61.9323 | 61.9323 | 61.9323 |
| `gain / k` | 9.938 | 9.999 | **10.116** | 10.334 | 10.696 | 11.035 |
| `mar` | 2.4707 | 2.4857 | **2.5148** | 2.5691 | 2.6592 | 2.7433 |
| `lsr` | 0.236788 | 0.235246 | **0.232147** | 0.225892 | 0.213180 | 0.187128 |
| **score ratio** | 0.2550 | 0.5068 | **1.0000** | 1.9399 | 3.5986 | 5.8052 |
| **score / k** | 1.020 | 1.014 | **1.000** | 0.970 | 0.900 | **0.726** |

**Exactly invariant: `odds` only.** It is built from `pnl%` ratios, so scale cancels completely —
61.9323 across a 32× range. `risk` is exactly *linear* in `k`, because `drawdown()` runs on the
arithmetic cumsum of `pnl%`.

**`mar` rises** (2.471 → 2.743) through a **compounding wedge**: `gain` comes from the *compounded*
equity curve `∏(1+rᵢ)` while `risk` is an *arithmetic* cumsum, so the numerator grows superlinearly
and the denominator linearly.

**`lsr` falls** (0.2368 → 0.1871, a 21% decline) because **`lsr` is computed from *dollar* P&L**, not
from `pnl%`. In a compounding account the later, larger sessions dominate `Σ|pnl|` faster than they
dominate `Σpnl`. r3 tabulated `lsr` from `pnl%` and reported it as invariant; that was wrong, and
the discrepancy was caught by the property test in `tests/test_properties.py`.

**The net result — and it reverses r3's conclusion.** `mar` rising is more than offset by `lsr`
falling and `daily` being sublinear. **`score / k` falls monotonically from 1.020 to 0.726**:
the score function has **decreasing returns to scale.** Doubling scale gives 1.94×, and 8× gives
5.81×, not 8×.

> ⚠️ **What this experiment still does not license.** It rescales a *fixed return path*. **No
> long-only miner can do that.** To raise portfolio volatility you must hold different assets, and
> that changes Sharpe, skewness, crash exposure, sector concentration, correlation structure,
> turnover and path dependence *at the same time*. The clean `k` here is a device for understanding
> which terms carry scale — it is **not** a portfolio decision you can execute, and the elasticities
> in §2.3 are **not** a sizing rule. §12 sets scale from a survival-constrained search over real
> candidate portfolios, which is the only version of this question that is actually decidable.

> ⚠️ **What this experiment does not license.** It rescales a *fixed return path*. **No long-only
> miner can do that.** To raise portfolio volatility you must hold different assets, and that
> changes Sharpe, skewness, crash exposure, sector concentration, correlation structure, turnover
> and path dependence *at the same time*. The clean `k` in this table is a mathematical device for
> understanding which terms carry scale — it is **not** a portfolio decision you can execute, and
> the elasticities in §2.3 are **not** a sizing rule. §12 sets scale from a survival-constrained
> search over real candidate portfolios, which is the only version of this question that is
> actually decidable.

### 2.3 The two elasticities `[MEASURED]`

500 fixed standard-normal 50-session paths, rescaled; median score; log-log regression.

**At fixed volatility (1.0%/session), score versus daily Sharpe:**

| Daily Sharpe | 0.04 | 0.06 | 0.08 | 0.10 | 0.13 | 0.16 | 0.20 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Annualised | 0.63 | 0.95 | 1.27 | 1.59 | 2.06 | 2.54 | 3.17 |
| **Score** | 0.009 | 0.059 | 0.182 | 0.444 | 1.193 | 2.690 | 6.228 |

→ **score ∝ Sharpe^4.8**

**At fixed daily Sharpe (0.08), score versus volatility:**

| Vol %/session | 0.25 | 0.40 | 0.60 | 0.80 | 1.00 | 1.40 | 2.00 | 3.00 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Annualised | 4.0% | 6.3% | 9.5% | 12.7% | 15.9% | 22.2% | 31.7% | 47.6% |
| **Score** | 0.049 | 0.078 | 0.114 | 0.149 | 0.182 | 0.240 | 0.311 | 0.390 |

→ **score ∝ volatility^0.85**

Together **`score ∝ Sharpe^4.8 × vol^0.85`**. Holding *return* fixed instead gives
`score ∝ return^3.9 / vol^2.9` — the same surface, and the reason "lower vol is better" is so
tempting and so wrong. You do not have a fixed return to defend; you have a fixed skill level and a
free choice of scale.

Live cross-section `[LIVE]`: Spearman(score, `mar`) = 0.971, Spearman(score, `lsr`) = 0.952,
Spearman(score, 50-day gain) = 0.918. A log-log regression on 116 scored miners gives
`log(score) = −2.19 + 2.42·log(Sharpe_ann) + 0.97·log(daily_vol)`, **R² = 0.36**.

> ⚠️ **Scope of these elasticities.** They were measured on **i.i.d. Gaussian** paths, over a narrow
> parameter range, on 50-session windows, holding one variable fixed while moving the other. That
> generating process has no fat tails, no volatility clustering, no serial correlation, no gaps, no
> skew, no regime changes, no transaction costs and no leveraged-ETF path dependence. Real equity
> returns have all nine. And `R² = 0.36` on the live cross-section means **most of the observed
> variation in score is explained by neither term.**
>
> Treat §2.3 as an explanation of *why* the score behaves as it does — Sharpe dominates, scale is
> not penalised the way a naive reading of "risk-adjusted return" suggests — and **not** as a
> function to be maximised directly. The stress suite in §11.2 exists precisely because this model
> is too clean.

### 2.4 Does the inception gate create a volatility optimum? No. `[MEASURED]`

The obvious objection: more volatility means more chance of tripping the lifetime high-water gate
(§3.2) and scoring zero. Measured over 3,000 250-session lifetimes at daily Sharpe 0.08:

| Vol (ann.) | 4.0% | 9.5% | 15.9% | 22.2% | 31.7% | 44.4% | 63.5% |
|---|---:|---:|---:|---:|---:|---:|---:|
| P(below inception) | 10.2% | 10.7% | 11.3% | 11.8% | 13.0% | 14.3% | 16.5% |
| **Mean gated score** | 1.39 | 3.89 | 6.60 | 9.43 | 13.89 | 20.26 | **30.88** |

Breach probability rises ~6pp across a 16× volatility range while score rises ~22×. Under **this
generating process** there is no interior optimum.

> ⚠️ **This is the weakest experiment in the document, and it points in the most dangerous
> direction, so read the caveat before acting on it.** The paths are i.i.d. Gaussian. The inception
> gate is a *first-passage* problem, and first-passage probabilities are exactly what a Gaussian
> model gets most wrong: no overnight gaps, no volatility clustering, no crash regimes, no
> correlation convergence, no leveraged-ETF decay, no multi-session loss clusters. A fat-tailed or
> regime-switching generator produces materially higher breach probabilities at high volatility,
> and the breach is **absorbing in score terms** — you earn zero until you climb back over the line.
>
> The honest conclusion is narrower: **the gate does not by itself impose a low volatility target**,
> and low volatility is not the way to survive it. It does not follow that maximum volatility is
> safe. Re-run this experiment on your own block-bootstrapped and crisis-replay paths (§11.2) before
> letting it influence any sizing decision.

What genuinely protects you from the gate is **Sharpe**, not timidity: at daily Sharpe 0.04 the
breach probability is 27–34%; at 0.08 it is 10–17%. That result is robust to the generator, because
it is drift versus noise rather than a tail question.

### 2.5 What each term actually rewards `[CODE]` `[MEASURED]` `[LIVE]`

**`odds` is not a hit rate, and it is the weakest term.** Algebraically
`kelly(p, pavg/lavg) = mean / pavg` exactly, so:

```
odds% = 50 · (1 + mean_daily_return / average_up_day)
```

Verified to 14 decimal places. **Win rate does not enter independently.** A 60%-win-rate book and
an 80%-win-rate book with the same mean and same average up-day score identically. What `odds`
rewards is a *small average winning day relative to the mean* — consistency of the up-days, not
their frequency. It is bounded in (0, 100], at most a 2× lever, versus `daily` which is unbounded.
**Never trade expected return for hit rate.**

**`lsr` is a Sharpe proxy and the best single predictor of score.** `lsr = Σpnl / Σ|pnl|`. For
i.i.d. normal returns `lsr = 1.2533 × daily Sharpe = Sharpe_ann / 12.66`. The README claims ">0.99
correlated to Sharpe" and "≈ Sharpe/10" — **neither holds on the live population**: measured
correlation with annualised Sharpe is **0.80**, and the median `Sharpe_ann / lsr` ratio is **15.2**,
not 10. The gap is mostly the outlier clip cutting `lsr`'s numerator. Calibrating a target `lsr`
from "Sharpe/10" overestimates by ~50%. Live top-10 `lsr` runs 0.32–0.62.

`lsr` uses **dollar** P&L post-clip, not `pnl%`, so in a compounding account later sessions carry
more weight; and unlike Sharpe it never penalises a large up-day.

**`mar` grows with time.** `gain` accumulates roughly linearly in sessions while max drawdown grows
roughly as √sessions, so `mar` grows ~√days. The same strategy earns materially more at session 50
than at session 30, even after the ramp completes. "The window resets everything" is the wrong
mental model.

### 2.6 The drawdown floor `[CODE]` `[LIVE]`

`mar = gain / max(risk, 5/√days)`. At a full 50-session window the floor is **0.707pp**.

**It almost never binds for a mature strategy.** 145 of 146 live miners with ≥10 sessions sit above
it; the median 50-session max drawdown is 3.03pp, and the live top-10 sit at 1.1–6.6pp. Reaching
the floor would need ~1.8% annualised volatility — a T-bill book, which the cash penalty then
multiplies by 0.01. **So for a mature miner, drawdown reduction is always fully rewarded.**

**Its real bite is on new miners.** At session 3 the floor is 2.89pp and at session 5 it is 2.24pp —
above a typical realised drawdown, so `mar` is capped for roughly your first 5–8 sessions. And
below the floor score becomes **quadratic** in scale rather than linear (measured: k = 0.25 gave
0.14× where linear predicts 0.25×). **A young miner running low volatility is penalised twice.**

### 2.7 The clip halves the median miner's score `[CODE]` `[LIVE]` `[MEASURED]`

```python
ii = dd[dd['pnl%'] > 0].sort_values('pnl%')[::-1][:self.clip_outliers + 1].index
if   len(ii) == 0:                      clip = 0
elif len(ii) == self.clip_outliers + 1: clip = dd['pnl%'][ii[-1]]                          # 3rd-best
else:                                   clip = min(self.clip_default, dd['pnl%'][ii[-1]])  # 1%
dd.loc[ii, 'pnl%'] = clip
for i in ii:                                     # and the equity curve is rebased
    open = dd['swap_close'][i-1] if i else init
    dd.loc[i:, 'swap_close'] -= dd['swap_close'][i] - open * (1 + clip / 100)
```

- Your top **three** positive sessions are all set to the third-best session's return, and the
  entire subsequent equity curve is rebased down. **Total gain is cut, not just the daily stats.**
- **Marginal profit on your best two sessions is worth exactly zero.**
- If the window holds ≤ 2 positive sessions, *all* of them are set to `min(1%, the smallest)`.
  A lone +5.98% session becomes +1.00%. 33 of 226 miners are in that branch. `[LIVE]`

Measured on live data: the median miner loses **1.00 percentage point of a 50-session gain (26% of
gain)** and the median post-clip / pre-clip **score** ratio is **0.518** — the clip halves the median
miner's score. The 10th percentile ratio is 0.005.

**The design consequence, and it is a big one for equities.** The score is **right-tail truncated
and fully left-tail exposed.** At identical mean and standard deviation over 50 sessions, a
positive-skew profile scores **0.68×** a negative-skew one. Returns that arrive in small consistent
increments are worth more than the same total arriving in a few large sessions.

> **Do not turn this into "avoid momentum".** r3 did, and that conflates two different strategy
> families with opposite skew. *Time-series trend following* does tend to be positive-skew, and the
> clip does penalise it. *Cross-sectional equity momentum* is the opposite — it has a documented
> severe left tail, crashing during rebounds out of panic states (Daniel & Moskowitz, *Momentum
> Crashes*, NBER w20439), which the drawdown term and the inception gate punish even harder than
> the clip punishes positive skew.
>
> So the label tells you nothing. **Measure the realised, post-cost properties of each candidate
> signal** — skewness, worst-session concentration, downside beta, rebound sensitivity, and its
> marginal contribution *after* clipping — and let §13 rank it. A signal earns its place by
> surviving the replica, not by its family name. The 0.68× figure is a skewness experiment, not a
> verdict on any strategy class.

### 2.8 What the objective wants — and the warning attached

```
score ≈ [ N·m / max(dd, 5/√N) ] · [ m / E|r| ] · [ 50(1 + m/pavg) ] · [ m ]
```

In priority order:

1. **Sharpe.** Exponent ~4.8, robust across generators, and the thing that keeps you above the
   inception gate. This is the lever. Everything else is second order.
2. **Full invested exposure.** `sum(|w|) = 1.00` with cash ≤ 1%. Unambiguous, costless, and
   independent of any risk view — cash is a linear score tax (§3.5). This is *not* the same claim
   as "maximum volatility"; it is about not leaving the budget unused.
3. **A thin right tail is free; a fat left tail is fatal.** The clip confiscates your best two
   sessions; the drawdown term and the gate punish your worst. Asymmetry, not smoothness.
4. **Low realised turnover.** The fee elasticity is ~4–5, not 1 (§12).
5. **Time in the window.** `mar` grows ~√days, the ramp completes at 30 sessions, and nothing
   accrues offline.
6. **Scale — decided last, and by search, not by rule.** Higher volatility is not penalised the way
   a naive risk-adjusted reading suggests, but §2.2's caveat means you cannot pick a volatility
   number from an elasticity. Choose it in §12 by maximising expected incentive subject to survival
   constraints, over real candidate portfolios, under the §11.2 stress suite.

> ⚠️ **The structural warning.** Optimising this score hard biases you toward the return profile of
> a **short-volatility / carry** strategy: high hit rate, small consistent gains, thin right tail —
> because the clip confiscates right-tail payoffs while only the drawdown term pushes back on the
> left tail. That is precisely the profile with hidden ruin risk, and a 50-session window cannot see
> it. **Build the absolute-floor risk governor in §15 before you build the alpha model**, and treat
> any strategy whose edge comes from selling tails as one you have not yet understood.

## 3. Every multiplier, in order

| # | Multiplier | Formula | Source |
|---|---|---|---|
| 1 | Outlier clip | top-3 positive sessions → 3rd-best; curve rebased | `pl2sc()` |
| 2 | New-miner ramp | `× (days/30) ** 1` while **lifetime** sessions < 30 | `pl2sc()` l.449 |
| 3 | **Inception gate** | `× 0 or 1` — all-time equity vs all-time start | `etc.py score()` |
| 4 | Similarity | `× min(gap_days / 30, 1)` for the **later** submission | `etc.py dedupe()` |
| 5 | Inactivity | `× 1 − min((last/20)², 1)` once **age** > 5 sessions | `etc.py score()` |
| 6 | Cash & short | `× clamp(1 − cash, 0.01, 1)` — linear from zero | `etc.py score()` |
| 7 | Class normalisation | stock class rescaled to 50% of the pool | `etc.py score()` |

### 3.1 The ramp, quantified `[MEASURED]`

`DAYS_DELAY = 1` today so the ramp is linear — but it is a **tunable constant**, one commit from
`(days/30)²`. Combined with `mar`'s √days growth and the short-window MAR floor, the same strategy
shape earns:

| Session | 10 | 20 | 30 | 50 |
|---|---:|---:|---:|---:|
| Share of its session-50 score | **0.3%** | 7.7% | 19% | 100% |

At ~21 sessions per calendar month, that is roughly six calendar weeks to 19% and ten to 100%.
**Nothing accrues offline.** This arithmetic decides §24.

### 3.2 The inception gate `[CODE]` `[LIVE]`

```python
for uid, dd in pl.groupby(pl.columns[0]):
    oc.loc[len(oc)] = uid, int(dd['swap_close'].iat[-1] >= dd['swap_open'].iat[0])
sc['score'] *= sc['c']                                    # hard 0/1
```

It groups over the **full** P&L frame; the 50-session truncation happens later, on a copy, inside
`pl2sc()`. `/pnl` returns up to 144 rows per UID. The test is *"is your book above its all-time
starting value of $10M?"* — which directly contradicts the reassurance that old performance leaves
the window.

`[LIVE]` **65 of 226 UIDs (29%) are currently zeroed by this gate**, and the 25th percentile of
since-inception return is only **−0.36%** — a quarter of the field is one bad session from losing
all emission. This term dominates every other risk consideration and changes what the risk engine is
for: **defend an absolute floor anchored at inception capital, not a rolling drawdown budget.**

It is not independently binding today only because every underwater UID has ≤ 50 sessions of
history, so window and lifetime coincide. It goes live the first time a long-lived miner draws down
and partially recovers.

### 3.3 The similarity rule — and the grief vector `[CODE]` `[LIVE]`

`dd_trigger = 0.01` — Euclidean distance between **L1-normalised** weight vectors; recovery is
linear over 30 days of submission gap.

- **It is an anti-plagiarism tripwire, not a diversity incentive, and it binds on nobody.** Zero
  live pairs are below 0.01. The **closest US-stock pair is 0.0561 — 5.6× the threshold.** Stock
  distance quantiles: p1 = 0.146, p5 = 0.184, p50 = 0.283. Two equal-weight 25-name books sharing
  24 of 25 names are already at 0.057. **Do not build a similarity-avoidance subsystem.**
- **The penalty lands on whoever holds the *later* submission.** `du[du >= 0]` keeps only positive
  gaps; the first submitter is never penalised. So a copier who clones your book and then never
  resubmits flips the penalty onto **you** the moment you next rebalance into anything still within
  0.01. Check your distance to the field before every rebalance (§17).
- Threshold sensitivity if the owner raises it: at 0.10, 8 stock pairs would trigger. `[LIVE]`

### 3.4 Inactivity — no grace period, on a calendar clock `[CODE]` `[LIVE]`

**The `> 5` guard is on `days` — your track age — not on idle time.** Once older than five sessions,
the penalty applies from the first idle day.

| Days idle | 1 | 2 | 3 | 4 | 5 | 10 | 15 | 20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Score kept | 99.75% | 99% | 97.75% | 96% | 93.75% | 75% | 43.75% | **0%** |

`last` is **calendar days**, so a stock book's clock runs over weekends and market holidays when no
P&L accrues at all. A Friday-only cadence is a permanent ~1% tax for nothing; a long weekend is 3%.

Live: top-cohort miners run `last = 2`; the tail runs 4–5. **Refresh daily, including weekends.**
A timestamp touch with unchanged weights is a full resubmission — it rebalances to target and pays
fees only on the drift, roughly 0.5bp per refresh on a 25-name book.

### 3.5 Cash and shorts — linear from the first basis point `[CODE]` `[LIVE]` `[MEASURED]`

```python
sc['score'] *= ((1 - sc['cash']) ** CASH_DECAY).clip(CASH_RESIDUE, 1)   # 1, 0.01
```

Nothing is reserved for "excessive" cash, and the true cost compounds because cash costs the
multiplier *and* the return it would have earned:

| Cash weight | 10% | 20% | 35% | 50% |
|---|---:|---:|---:|---:|
| Score retained | **0.809×** | 0.638× | 0.420× | 0.248× |

Live stock miners know this: median `cash` is 1.0%, and 51% hold ≤ 1%. **Your invested band is
99–100%, not 90–100%.**

**Shorts are counted in the same penalty.** Shorting *is* permitted in the stock class (negative
weights), unlike the alpha class — but a 70/30 long-short book is multiplied by **0.70** before any
performance is measured, *and* consumes 30 points of the gross budget. That is a punishing price for
a hedge. **Do not short in v1.**

### 3.6 Owner-side levers `[LIVE]`

| Lever | Effect | Live value |
|---|---|---|
| `/dist` tail `[0]` | Blacklist — dedupe multiplier forced to `0`, score zeroed | `[]` |
| `/dist` tail `[1]` | Whitelist — forced to `1`, exempt from similarity | `[]` |
| `/dist` tail `[2]` | Overrides `DD_TRIGGER` itself | `0.0` → falls back to 0.01 |
| `/ratio` | Class split **and** a burn valve — shortfall below 1.0 goes to UID 0 | `[0.5, 0.5]` |
| `/days` | Supplies `cash` and `last` — two multipliers — as owner-computed numbers | 226 rows |

Publishing `/ratio = [0.25, 0.25]` would halve every miner's income overnight; publishing
`[0.3, 0.7]` would raise the stock class by 40%. **Poll it.**

Also: **validators run no simulation.** `validator.py` fetches four payloads over plain `http` and
`etc.score()` does arithmetic on them; `pldaily1()`, the market simulation, is never called in the
validator path. Nothing is signed. `[CODE]`

## 4. Nine ways to score zero

Only three are documented.

1. **Gross exposure above 1.0.** `if asum > 1: continue` — the whole file is **silently discarded**
   and your *previous* allocation stays live. No error, no partial fill, and the inactivity clock
   keeps running while you believe you rebalanced. `[CODE]`
2. **Below your all-time starting value.** §3.2. Currently zeroing 29% of the field. `[CODE]`
3. **Negative return over the window.** `if score <= 0: score = 0`. No partial credit — this is what
   the 44 zero-score stock miners are hitting. `[CODE]`
4. **20 days without a rebalance.** §3.4. `[CODE]`
5. **`odds` collapses.** An all-loss window sends `pavg` to NaN and `odds` to `prob × 100`. `[CODE]`
6. **Blacklisted by the owner.** §3.6. `[CODE]`
7. **Wrong asset class.** `an = 2`, so `'_':2` drops the whole strategy — and omitting `'_'`
   entirely defaults to `0`, the *other* class. **`'_': 1` is mandatory in every file you write.**
   The class lock is keyed on **UID, not hotkey**, so a recycled UID may inherit its predecessor's
   class and silently ignore everything you file. **Verify your `a` column reads 1 before building
   anything.** `[CODE]` `[INFERRED]`
8. **Cash to the floor.** A 99%-cash book retains 1% of score. §3.5. `[CODE]`
9. **Deregistered before the epoch settles.** A neuron pruned mid-tempo gets nothing for the partial
   tempo. `[DOC]`

## 5. The submission contract

### 5.1 The file `[CODE]` `[DOC]`

One Python dict, in a file named for your hotkey SS58, under `Investing/strat/`.

```python
{'_': 1,                                   # MANDATORY — selects US stocks
 'AAPL': 0.15, 'AMZN': 0.10, 'GOOG': 0.15,
 'MSFT': 0.20, 'NVDA': 0.25, 'SPY': 0.14}  # sums to 0.99 → 1% auto-cash → ×0.99
```

Tickers are **case-sensitive**. The repo's own sample sums to 0.95 and is already leaking 5% of
score; target `sum(|w|)` in `[0.99, 1.00]`.

### 5.2 What the parser does to your file `[CODE]`

```python
st['strat'] = st['strat'].str.replace(r'''[^{'\w":.,}\[\]\*=+-]''', '', regex=True)
asum = sum([abs(strat[k]) for k in strat if k not in ['_', '*', '=', '-+']])
if asum > 1: continue
strat = {k: strat[k] for k in strat if k and k in bn[bn['date'] == date]['netuid'].values}
strat[''] = 1 - sum([abs(v) for v in strat.values()])
```

Five consequences the publisher must encode:

- **A regex sanitiser strips every character outside `{ ' \w " : . , } [ ] * = + -`.** Never put a
  comment, a parenthesis or an arithmetic expression in a strategy file — it is silently mangled,
  then `eval`'d, then dropped by `except: continue`.
- **No leverage.** `asum > 1` discards the file entirely and silently.
- **Cash is the leftover, and your explicit `''` value is ignored.** Falsy keys are dropped, then
  cash is *recomputed*. `{'_':1, 'AAPL':0.5, '':0.2}` yields **50% cash**, not 20%.
- **An unsupported, delisted or misspelled ticker is silently reclassified to cash** and feeds the
  cash penalty. **Validate every symbol against today's `/assets` before writing.** This is the
  single most likely silent failure in a live equity system, because tickers change.
- Reserved keys `'*'`, `'='`, `'-+'` are excluded from the budget, then dropped. Inert today.

### 5.3 The miner loop `[CODE]`

```python
while True:
    if etc.isnew(ss58): api.rev(ss58), time.sleep(10)
    time.sleep(1)
```

`isnew()` is `getsize(strat) and getmtime(strat) > getmtime(last)`, where `last` is the sentinel
`Investing/strat/.last-update`. `api.rev()` POSTs `{'ss58':…, 'strat': <raw text>}` to `/rev` over
plain **http** and only touches `.last-update` on `status_code <= 201`.

- A zero-byte strategy file is never submitted.
- **You do not need a retry loop** — a failed submission is retried every ~11 seconds, forever.
- `forward()` in `miner.py` is unmodified template code returning `input * 2`. No validator queries
  your axon. **Do not build serving logic.**

### 5.4 Execution — MOO and MOC only `[CODE]` `[DOC]`

| | US stocks |
|---|---|
| Order types | Market on Open, Market on Close |
| Cutoffs | `STK_MOO = -120s` → **09:28 ET**; `STK_MOC = -600s` → **15:50 ET** |
| Collapsing | **Only the last file submitted inside each window executes** |
| Late submission | Waits for the next session's MOO |
| Cost | `STK_FEE = 0.002` — **$0.002 per SHARE traded**, not 0.2% of notional (§12) |
| Marked at | Session close |
| Scored at | **06:00 UTC** |
| Benchmark | `STK_BENCH = SPY` |

Two operational consequences. **Intraday resubmissions are free but pointless** — everything since
the previous cutoff is discarded and only the final pre-cutoff file fills, so your cycle should
produce exactly one decision per window. And **you have two decision points per session**, 09:28 and
15:50 ET; a system that only runs once per day is leaving an execution option unused.

### 5.5 MOO and MOC need separate feature snapshots and separate labels

This is a leakage vector that an ordinary daily-bar backtest will not catch, and it silently
inflates every result.

- An **MOO** decision may use only information available before **09:28 ET** — i.e. through the
  prior session's close, plus overnight news and pre-market data.
- An **MOC** decision may use only information available before **15:50 ET**. It **may not use the
  session's closing price, closing volume, or anything published after 15:50** — but a daily-bar
  backtest hands it exactly those, because the daily bar *is* the close.

If you build one daily feature set and use it for both windows, your MOC model is trading on the
close it is supposed to be predicting.

Required:

```
features_moo(t)   : available_time <= t 09:28 ET
features_moc(t)   : available_time <= t 15:50 ET     # distinct snapshot, not a copy
label_moo(t)      : fill at open(t)  -> close(t) or open(t+1)
label_moc(t)      : fill at close(t) -> close(t+1)
execution prices  : opening auction / closing auction, not the daily bar
calendar          : early closes (13:00 ET), DST transitions, half-days
```

And a decision rule: **analyse twice per session, but only publish at MOC when new information
justifies the extra turnover.** At 0.2% per side (§12) a reflexive twice-daily rebalance is a
straightforward way to lose more to fees than the second decision point is worth.

Validators recompute scores at minutes `:25` and `:55`.

## 6. The universe

**555 supported tickers** `[LIVE]`, ranked by market cap, roughly $98T of aggregate cap. About 88
are ETFs and funds; the rest are single names, heavily weighted to US large cap with some ADRs
(TSM, ASML, SAP, NVS, HSBC, BABA, TM, MUFG, RY, TD…).

### 6.1 Sixteen tickers are scored as cash

```
SGOV, BIL, JPST, BSV, VCSH, VGIT, GOVT, BND, IUSB, AGG, IEF, BNDX, MBB, VCIT, MUB, VTEB
```

They are all tradable and fill normally, but they count toward `cash` and multiply your score
linearly (§3.5). **It is a curated list, not a rule about bonds**, and that asymmetry is worth real
score:

| Free — full weight, zero penalty | Taxed as cash |
|---|---|
| **TLT** (20y+ Treasuries, 15–18% vol), **LQD** (credit) | SGOV, BIL, JPST — T-bills |
| **GLD, IAU, SLV** — precious metals | AGG, BND, IUSB — aggregate |
| **JEPI, JEPQ** — covered-call income | IEF, VGIT, GOVT — intermediate Treasuries |
| **VNQ** (REITs), **XLV**, **SCHD** — defensive equity | MUB, VTEB, MBB, VCIT, VCSH, BSV, BNDX |

**What this buys you is a defensive sleeve that costs no cash penalty** — a real option most of the
field is not using, because the list is invisible from the documentation.

**What it does not buy you is a universal hedge.** r3 called TLT and GLD "the correct crisis hedge"
and that is too absolute. The equity–bond correlation is **regime-dependent**: duration hedges a
growth or recession shock, and fails in an inflation or rate shock, where stocks and bonds fall
together (BIS, *The correlation of equity and bond returns*, Quarterly Review, Dec 2023 — and 2022
is the obvious lived counterexample). Gold helps in some geopolitical and currency scenarios and is
uncorrelated-to-unhelpful in many ordinary equity drawdowns.

**Select the hedge from the diagnosed shock, not from a fixed rule:**

| Diagnosed shock | Candidate hedges |
|---|---|
| Growth / recession | TLT, LQD, defensive equity (XLV, SCHD) |
| Inflation / rate | GLD, energy, inflation-sensitive equity — **not** duration |
| Credit | High-quality equity; Treasuries only if inflation permits |
| Geopolitical / energy supply | Energy plus GLD |
| Technology-specific | Broader sectors, value and quality tilts |
| USD | Adjust international-revenue and exporter/importer exposure |

The regime model in §11 decides which row applies; the governor never hard-codes one.

### 6.2 TQQQ: a capped candidate, not a scale lever

**TQQQ is the sole leveraged ETF in the universe** — no SOXL, UPRO, SPXL, TNA or their inverses.
Since `sum(|w|) ≤ 1` forbids leverage outright, it is the only instrument that can push portfolio
volatility past what a long-only cash-equity book reaches.

r3 called it "the scale lever". That was too strong. TQQQ pursues a **daily** 3× objective, so its
multi-day return can diverge substantially from 3× the index — the SEC's leveraged-ETF investor
bulletin exists precisely because that compounding behaviour produces sudden, large losses in
volatile markets. The decay shows up directly in `mar` and the drawdown term, and its left tail is
what the inception gate is waiting for.

Treat it as one candidate among many, with hard rules:

```yaml
tqqq:
  baseline_weight: 0.00          # start here
  max_weight: 0.10               # challenger sleeve only
  requires:
    - path_dependent_simulation  # not a 3x linear proxy
    - gap_and_decay_stress
    - nasdaq_factor_concentration_check
  forbidden_when:
    - headroom_below_floor_warn  # §15
    - vol_regime_elevated
  stress_periods: [2000-2002, 2008, 2020, 2022]
```

Rank 10 on the live board runs 63.9% annualised volatility, so *someone* is taking high-beta
exposure. That is an observation about the field, not a recommendation — and note rank 10 scores
38.7 against rank 2's 111.9 at 10.9% volatility.

### 6.3 Liquidity

Deepest names by price × volume: SNDK, SPY, NVDA, QQQ, TSLA, MU, MSFT, AAPL, AMZN, GOOGL. Because
fills are simulated at session prices on a $10M book with a flat fee, **market impact does not enter
the score** — liquidity matters only as a proxy for data quality and for the realism of any strategy
you might later trade with real money. Do not spend modelling effort on impact.

---

# Part II — The system

## 7. System overview

```
                    ┌─────────────────────────────────────────────┐
                    │  §17 field intelligence  (/pnl /score /days │
                    │       /dist /assets /ratio — every rival)   │
                    └───────────────┬─────────────────────────────┘
                                    │ rivals' scores, your projected rank
   §8 data plane ──▶ §9 features ──▶ §10 forecasts ──▶ §12 candidates
        │                                 │                  │
        │                            §11 risk engine ────────┤
        │                                                    ▼
        │                                    §13 SCORE REPLICA  ◀── §3 rules
        │                                                    │
        │                                              §14 selector
        │                                                    │
        └──────────── §18 observability ◀───────────  §15 risk governor
                                                             │
                                                       §16 publisher ──▶ /rev
```

**Design rule: the score replica is the only component allowed to rank candidates.** Everything else
proposes; §13 decides. Sharpe, CVaR and tracking error are diagnostics, never objectives.

```
sn88/
├── config/{rules.yaml, system.yaml}        # versioned mirror of const.py + API values
├── data/{ingest,store,quality}/            # point-in-time parquet + duckdb
├── features/
├── models/{forecast,risk}/
├── portfolio/{candidates,governor,publisher}.py
├── scoring/{replica.py, golden/}           # §13 + captured /score fixtures
├── field/                                  # §17 rival modelling
├── research/{ledger.db, walkforward/}
├── ops/                                    # scheduler, alerts, runbook checks
└── tests/
```

## 8. Module 1 — Data plane

Store every observation with both `event_time` (when it happened) and `available_time` (when you
could have known it). This one discipline prevents most leakage.

```
prices(ticker, session_date, available_time, open, high, low, close, adj_close, volume, source)
actions(ticker, event_time, available_time, kind, ratio, cash_amount)   -- splits, dividends, delistings
symbol_map(old_ticker, new_ticker, effective_date)                      -- ticker changes
universe(ticker, date, is_tradable, is_cash_classified, sector, market_cap)   -- daily /assets snapshot
fundamentals(ticker, event_time, available_time, field, value)          -- original publication dates
regime(date, field, value)                                              -- rates, curve, credit, USD, commodities, vol
field_state(date, uid, score, lsr, mar, risk, odds, daily, gain, days, last_active, cash, rank)
field_dist(date, uid_a, uid_b, l2_distance, gap_days)
rules(date, key, value)                                                 -- const.py + /ratio + /assets header
sessions(date, is_trading_day, moo_cutoff_utc, moc_cutoff_utc)          -- NYSE calendar
```

The `sessions` table matters more here than in a normal equity stack, because §2.1's two clocks both
depend on it: the scoring window counts sessions while the inactivity penalty counts calendar days.

**Quality gates — all must pass before the cycle proceeds (§21):** no stale price beyond N sessions;
corporate actions reconciled; **every held ticker present in today's `/assets`**; no `available_time`
in the future; the `rules` snapshot matches upstream.

**Acceptance test: reconstruct an arbitrary past session without reading anything published after
it.**

## 9. Module 2 — Feature store

Four families, fitted on training data only, versioned with the code that produced them.

- **Cross-sectional value / quality** — earnings yield, profitability, leverage, accruals, revisions
  (point-in-time only). Returns arrive in small increments, which §2.7 rewards.
- **Mean reversion** — short-horizon reversal, distance from moving averages, pairs and
  sector-relative dislocation. Favoured over breakout momentum because of the clip.
- **Risk** — realised and downside volatility, beta, correlation and its rate of change, historical
  CVaR, drawdown and recovery duration, gap risk, volatility-of-volatility.
- **Regime** — market trend, cross-sectional dispersion, volatility level, correlation
  concentration, breadth, and cross-market confirmation (rates, curve, credit, USD, commodities,
  broad vol).

Include longer-horizon momentum, but do not let it dominate v1 — §2.7 measured a 0.68× penalty on
positive-skew profiles.

## 10. Module 3 — Forecast ensemble

### 10.0 Does AI earn its place here? Yes — but third. `[MEASURED]`

The question is not "ML or no ML", it is **where the score actually comes from.** Using
`score ∝ Sharpe^4.8 × vol^0.85` and the exact multiplier formulas, here is what each lever is worth
in score multiples:

| Lever | Needs a model? | Score × |
|---|---|---:|
| Cut cash 20% → 1% | no | 1.24× |
| Refresh cadence 10d → 1d idle | no | 1.33× |
| Drop a 30% short sleeve | no | 1.43× |
| Turnover 20%/session → 3% | no | 1.42× |
| Raise vol 13.4% → 21.8% at equal Sharpe | no | 1.51× |
| **All of the above, stacked** | **no** | **≈ 2.5×** |
| Sharpe: rank-51+ → median (2.48 → 3.48) | yes | 5.1× |
| Sharpe: median → rank 11–25 (3.48 → 5.23) | yes | 7.1× |
| Sharpe: median → top-10 (3.48 → 5.95) | yes | **13.1×** |
| Sharpe +10% at the top (5.95 → 6.55) | yes | 1.6× |

Read it in two passes.

**Model quality dominates the ceiling.** Because the Sharpe exponent is ~4.8, a 40% Sharpe
improvement — roughly what Gu, Kelly & Xiu report for trees and shallow nets over a linear benchmark
on US equities — is worth **5.0×** on score. Getting from the field median to the top ten is 13×.
Nothing mechanical comes close. **So yes, build the models.**

**But the mechanics are free and come first.** ~2.5× sits in cash discipline, refresh cadence,
turnover control and scale — none of which needs a single fitted parameter, all of which can be
live within a week, and all of which multiply whatever the model later earns. A brilliant model
running at 20% cash and a 5-day refresh throws away half its own output.

The build order in §24 follows directly: mechanics and the replica first, models at stage 7.

### 10.1 Which models actually work here

The score's shape (§2.7 — right-tail truncated, left-tail exposed) and its data scale (§8 — a
555-name daily panel) both point the same way, and it is not toward the largest architecture.

| Tier | Use | Verdict for v1 |
|---|---|---|
| **Gradient-boosted trees** (LightGBM / XGBoost / CatBoost) on a cross-sectional panel | Rank forward residual returns; classify `P(down session)` | **Yes — the workhorse.** Best evidence-to-effort ratio on tabular equity panels; trains on CPU in minutes. |
| **Regularised linear** (elastic net, ridge on orthogonalised factors) | The baseline every other model must beat | **Yes.** Cheap, stable, and surprisingly hard to beat after costs. |
| **Shallow feed-forward nets** (2–4 layers) | Non-linear interactions the trees miss | **Yes, as an ensemble member.** Gu/Kelly/Xiu find gains flatten past ~3 layers. |
| **Conformal / quantile models** | Uncertainty for the shrinkage step and the governor's floor budget | **Yes.** Directly feeds §11 and §15. |
| **Sequence models** (LSTM, Temporal Fusion Transformer) | Multi-horizon forecasting with variable selection | **Challenger only.** Must beat the GBDT ensemble through the replica at 2× costs. |
| **Deep RL** (portfolio agents) | End-to-end allocation | **No for v1.** The 2026 review of 156 studies finds reported gains confounded by data, infrastructure, tuning and regime. |
| **LLMs** | Structured event extraction for the macro overlay; research tooling | **Narrow use only.** Never let one size a position. An API call, not a hosted model. |

**Two SN88-specific model requirements** that a generic equity stack will not have:

1. **A joint predictive distribution, not just a point forecast.** `lsr` and `odds` saturate when
   there are no losing sessions, and the drawdown term and the inception gate punish the left tail —
   so the left tail is what you must model. That means quantiles and a dependence structure, from
   which portfolio down-probability is *derived* after weights are proposed (§10.2), not a
   standalone "will today be down" classifier.
2. **The replica as the model-selection criterion.** Selecting on Sharpe or IC will systematically
   favour signals whose payoff the clip then confiscates, and will not see the inception gate at
   all. Score every candidate model through §13 in **mode B** — never by replaying it over known
   history (§13.1).

### 10.2 The targets

Predict tractable quantities, never price levels:

| Target | Why |
|---|---|
| Cross-sectional rank of forward residual return | Drives `mar` and `daily` |
| Return **quantiles** per name, not just the mean | Feeds the stress suite and the governor |
| Forward volatility | Sets achievable scale (§2.2) |
| **Covariance / dependence structure** | Without it, per-name downside probabilities cannot be combined |
| Gap probability | Feeds the jump component of §11.2 |

**On `P(down session)` — r3 overstated it and left it underdefined.** "Down" is ambiguous across at
least four different targets: an individual stock falling, *the selected portfolio* falling, the
market falling, or *the equity crossing the inception floor*. Only the last two are well-posed as
standalone forecasts. **Portfolio down-probability is endogenous to the weights you have not chosen
yet**, so it cannot be a model input to weight selection without circularity; and per-name down
probabilities cannot be aggregated without the dependence structure, which is why the covariance
row above is not optional.

The correct output is a **joint predictive distribution** — expected residual return, quantiles,
downside probability, expected shortfall, covariance, gap probability — from which the portfolio's
down-probability is *derived* once weights are proposed, inside §13. Keep a market-level
`P(down session)` and a floor-crossing probability as diagnostics, and promote either to a
first-class objective only after an ablation study shows it improves projected incentive through the
replica.

A sound v1: elastic net + gradient-boosted trees + a reversion model + a defensive risk model + an
equal-weight or inverse-vol prior. Shrink toward the prior in proportion to model disagreement or
conformal interval width:

```
μ̃ᵢ = cᵢ·μ̂ᵢ + (1 − cᵢ)·μ_prior,ᵢ        cᵢ low when disagreement or interval width is high
```

Every model must beat equal-weight and inverse-volatility **through the score replica**, not through
Sharpe.

## 11. Module 4 — Risk engine

### 11.1 Estimators

- **Two covariance estimators minimum** — Ledoit–Wolf shrinkage and an EWMA. Never invert a raw
  sample covariance for production weights.
- **Regime detection** — diagnoses *which shock* is live, which selects the hedge row in §6.1.
- **The floor budget** — distance from current equity to the $10M inception value, and the
  probability of touching it before recovery. This is the number the governor consumes (§15), and
  it matters more than any rolling-drawdown metric because of §3.2.

### 11.2 The stress suite — this replaces the Gaussian model

Everything in §2.3 and §2.4 was measured on i.i.d. Gaussian paths. That generator has no gaps, no
clustering, no regimes and no crashes, so it is systematically blind to the events that trip the
inception gate. **The Gaussian simulation is retained as a unit test of the replica, and is not
permitted to set scale.** Candidate evaluation in mode B (§13.1) must draw forward paths from:

| Method | Captures |
|---|---|
| **Stationary / circular block bootstrap** | Volatility clustering and serial correlation, non-parametrically |
| **Historical crisis replay** | 1987, 2000–02, 2008, 2020, 2022 — actual joint dynamics |
| **Regime-switching simulation** | Transitions between calm and stressed states |
| **Jump-diffusion** | Overnight gaps and single-session shocks |
| **GARCH-family residual simulation** | Clustered volatility with fat-tailed innovations |
| **Correlation shock** | Cross-sectional correlation → 1 |
| **Factor shocks** | Market, size, value, momentum, quality dislocations |
| **Leveraged-ETF path simulation** | Multi-day compounding decay for any TQQQ sleeve (§6.2) |

From those paths compute, per candidate:

- Multi-session **expected shortfall**, not just single-session VaR.
- **Probability of crossing the inception floor**, and the expected time to recover if crossed.
- Worst-session and worst-5-session loss concentration.
- Realised skewness after costs and after the clip.

These are the constraints §12 optimises subject to.

## 12. Module 5 — Candidate generator

Generate 20–50 candidates per cycle rather than trusting one optimiser configuration: equal weight,
inverse volatility, equal risk contribution, hierarchical risk parity, shrinkage minimum variance,
Black–Litterman with your views, CVaR-minimising, a TLT/GLD defensive tilt, a TQQQ scale-up of the
current book, and **the previous portfolio unchanged** — often the winner once turnover is priced.

```yaml
gross_exposure:      [0.99, 1.00]   # §3.5 — every point of cash is a point of score
cash_weight:         [0.00, 0.01]   # incl. the 16 cash-classified tickers (§6.1)
shorts:              false          # §3.5 — counted as cash AND consume gross
max_single_name:     0.10
max_sector:          0.25
holdings:            [15, 40]
turnover_per_day:    [0.00, 0.05]   # normal operating point
turnover_hard_cap:   0.15
no_trade_band:       true
```

**Scale is not a constraint here — it is the search variable.** Do not set a volatility target and
do not set `volatility: maximise` (r3's error). Generate candidates across **several volatility
bands** and let the objective choose:

```
maximise   E[ projected incentive ]                        # mode B, over §11.2 paths
subject to P(inception breach within H sessions) ≤ p_max
           expected shortfall (multi-session)   ≤ es_max
           gap loss                             ≤ gap_max
           regime stress loss                   ≤ stress_max
           sector / single-name concentration   ≤ caps
           realised turnover                    ≤ turnover_max
```

The optimiser must be allowed to return a *lower*-volatility book when the stress constraints bind.
An architecture that can only ever choose the highest admissible volatility has assumed its
conclusion.

**Turnover is nearly free here, and r3 said the opposite.** `STK_FEE = 0.002` reads like 0.2% but
the code applies it as `|Δshares| × STK_FEE / price` **shares**, whose dollar value is
`|Δshares| × 0.002` — i.e. **$0.002 per share traded, a per-share commission**. And market impact
never enters the score at all (stock `swap` is identically `value`). Recomputed for a $10M book:

| Price | Turnover | Traded | Per-share fee | r3's "0.2% of notional" | Overstatement |
|---:|---:|---:|---:|---:|---:|
| $50 | 15% | $1.5M | $60 (0.06bp) | $3,000 (3.0bp) | 50× |
| $150 | 15% | $1.5M | $20 (0.02bp) | $3,000 (3.0bp) | **150×** |
| $400 | 30% | $3.0M | $15 (0.01bp) | $6,000 (6.0bp) | 400× |

So r3's claim that "a 20% cap is fatal as a normal operating point" was wrong by two orders of
magnitude, and the score table it rested on is withdrawn.

**Keep a turnover constraint anyway — but for a different reason.** It is a *risk* control, not a
cost control: high-turnover signals are the ones least likely to survive out of sample once real
costs exist (Novy-Marx & Velikov), and a strategy tuned to a simulator where trading is free is a
strategy you cannot ever run with real money. Size the band on signal decay, not on the fee.

## 13. Module 6 — The score replica

**The heart of the system, and the first thing you build.** A byte-faithful port of `pl2sc()` +
`score()` plus the four `etc.py` multipliers and the gate. Roughly 200 lines (Appendix B).

### 13.1 Three evaluation modes — do not collapse them

r3 said "replay the trailing 50-session window with the candidate's weights." **That is a
look-ahead bug**, and it is the most dangerous one available here: a candidate generated from
today's features, applied backwards over 50 sessions whose returns are already known, will score
brilliantly and mean nothing. The replica must expose three distinct modes and the type system
should keep them apart.

| Mode | Inputs | Legitimate use |
|---|---|---|
| **A. Reconciliation** | Your *actual* historical weights, actual historical P&L | Reproduce your current on-chain score. Validates the replica (§13.2). **Never** used to rank candidates. |
| **B. Forward projection** | Historical P&L **frozen**; candidate applied only from the next eligible MOO/MOC cutoff onward, over *simulated* future paths (§11.2) | Ranking today's candidates. This is the only mode the selector may call. |
| **C. Policy backtest** | Walk the clock forward; at each session *t* use only data with `available_time ≤ t` | Measuring whether a strategy *policy* works at all. Feeds §19. |

Mode C, stated precisely, because it is easy to get subtly wrong:

```
for t in sessions:
    snapshot = load_features(available_time <= cutoff(t, window))   # 09:28 or 15:50 ET
    model    = model_registry.as_of(t)                             # trained only on data < t
    w_t      = policy(snapshot, model, w_{t-1})
    fill     = execute(w_t, at = next_eligible_cutoff(t))
    pnl_t    = realised_return(fill, t)                            # revealed only after the fill
    advance
```

**Never apply a weight generated at *t* to any return before *t*.** A single violation makes every
downstream number meaningless, and the failure is silent — the backtest simply looks excellent.

### 13.2 What the replica computes

For a candidate in **mode B**:

1. Apply `STK_FEE = 0.002` per side on the traded difference from the current book.
2. Hold the realised trailing window fixed; append simulated forward sessions from §11.2.
3. Apply the clip **including the equity-curve rebase** (§2.7).
4. Compute `mar`, `lsr`, `odds`, `daily` with the correct **two** day counts:
   - **lifetime** sessions → the ramp and the inactivity age guard
   - **windowed** sessions (≤50) → `daily` and the MAR floor
5. Apply ramp, inception gate, similarity, inactivity, cash.
6. Renormalise the stock class to 50% of the pool and compute **projected incentive share**.
7. Report the *distribution* across simulated paths, not a point estimate — the selector needs the
   median, the dispersion and the breach probability (§14).

**Acceptance test (mode A):** reproduces the live `/score` table to <1e-6 relative on every row, and
`Investing/bin/validator`'s adjusted column to displayed precision. Keep captured `/score` responses
in `scoring/golden/` as regression fixtures (§20).

## 14. Module 7 — Selector

Rank on a robust statistic in **incentive units**, not score units.

```
RobustIncentive = median(incentive across scenarios)
                − 0.5 × stdev(incentive across scenarios)
                − failure_penalty
```

Reject any candidate whose advantage depends on a single regime, a handful of sessions, or the two
sessions the clip will confiscate anyway. Prefer the incumbent book unless the challenger's
projected incentive gain exceeds its round-trip cost.

## 15. Module 8 — Risk governor

A separate component with **final authority the model cannot disable**. It vetoes; it never
proposes. Every rejection is logged with a machine-readable reason.

**Hard vetoes**

- `sum(|w|) > 1.00` — would be silently discarded (§4.1)
- missing or wrong `'_': 1`
- cash or cash-classified weight above the configured band
- any short in v1
- a ticker absent from today's `/assets` snapshot
- concentration or sector breach

**The absolute floor — the governor's primary job.** Because of §3.2 the relevant risk is not a
rolling drawdown budget; it is **distance to the $10M inception value**:

```
headroom = (current_equity / 10_000_000) − 1
if headroom < floor_warn:    block scale-ups, forbid TQQQ, tighten concentration
if headroom < floor_crit:    force the highest-Sharpe admissible candidate under stress
if headroom < 0:             enter RECOVERY MODE — see below
```

**Recovery mode is a survival problem, not a score problem.** r3 said that below the floor your
score is zero regardless, so the rational response is *more* risk. **That is wrong and dangerous.**
It optimises today's score in isolation and ignores that more risk can push you further below the
line, lengthen expected recovery, burn fees, and get you pruned before you ever recover. The gate
is absorbing in score terms but *not* absorbing in capital terms — you keep trading.

The objective below the floor is closer to:

```
maximise  E[ Σ_t δ^t · incentive_t ]
        − λ · P(pruned before recovery)
        − γ · E[ time to recover ]
```

Whether that implies more risk, less risk, or a differently *structured* risk depends on current
headroom, expected alpha, pruning pressure (§27) and recovery probability — all of which change
daily. **Build a recovery-policy simulator** that solves this over the §11.2 paths, and record its
recommendation. Do not hard-code a direction in either direction.

```yaml
recovery_mode:
  objective: expected_discounted_incentive_minus_pruning_and_delay
  horizon_sessions: 60
  solver: simulate_over_stress_paths      # §11.2
  hard_limits_still_apply: true           # gross, cash, concentration, no shorts
```

## 16. Module 9 — Publisher

```
validate → write atomically → confirm mtime advanced → confirm POST ≤ 201
        → confirm .last-update advanced → else alert
```

Validation is not optional; §5.2 fails silently in four different ways. Never write a file you have
not validated, and always validate tickers against today's `/assets`. Do not implement retries — the
miner already retries every ~11 seconds.

## 17. Module 10 — Field intelligence

**The whole field is readable, and this is a bigger edge than the alpha model.**

| Endpoint | What it gives you about *every* UID |
|---|---|
| `/pnl` | Complete daily equity curve — OHLC in mark and swap terms, up to 144 rows |
| `/score` | Authoritative raw score table: score, lsr, mar, risk, odds, daily, return |
| `/days` | Rank, class, lifetime sessions, days since last rebalance, cash allocation |
| `/dist` | Full pairwise distance matrix, submission gaps, blacklist, whitelist, threshold |
| `/assets` | 555 tickers with price/volume/market cap, the cash list, the delisting list |
| `/ratio` | Class emission split |
| `Investing/bin/validator` | Runs `etc.score()` live — the adjusted numbers that become on-chain incentive. **No wallet, stake or registration required.** |

What to build on it:

- **Projected rank**, recomputed every cycle, before validators publish.
- **Marginal incentive per unit of score** — what a given improvement is worth *today*.
- **Distance-to-field monitor** — your minimum L2 to any rival, checked before every rebalance.
- **Rule-change detector** — diff `/ratio`, the `/assets` header and the `const.py` mirror.
- **Rival decomposition** — reconstruct each competitor's Sharpe, volatility, turnover and cash from
  their public equity curve (this is how §1.3 was built). You cannot see their model; you can see
  everything it produced, every session.

The dashboard shows **raw** performance; `bin/validator` shows **adjusted**. They differ.

## 18. Module 11 — Observability

Emit and alert on: projected vs realised score; projected vs realised incentive; predicted vs
realised volatility; realised turnover and fees; `last_active`; cash weight; gross exposure;
headroom to the inception floor; minimum distance to the field; replica-vs-`/score` drift;
submission confirmation latency; and the `rules` diff.

Log every decision with the inputs that produced it. When live and backtest diverge you will need
the inputs, not the outputs.

---

# Part III — Correctness

## 19. Research protocol

**Anchored walk-forward only.** Never random k-fold on overlapping labels. Purge observations whose
label windows overlap a fold and place an embargo between train and validation (López de Prado,
ch. 7).

```
Train 2010–2017 │ Validate 2018 │ Test 2019
Train 2010–2018 │ Validate 2019 │ Test 2020
… through the latest completed period
```

**Every backtest must include:** point-in-time universe membership, **delisted names and ticker
changes**, corporate actions, fundamentals publication lags, fees and turnover, missing-data
behaviour, MOO/MOC order timing, failed submissions, and **the exact rolling-score logic with all of
§3**. Survivorship bias is the classic killer in an equity backtest and the 555-name universe is
itself a survivorship-selected snapshot — reconstruct historical membership, do not assume it.

**The trial ledger.** Every model, feature set, parameter set and constraint set gets an ID *before*
it is tested: `experiment_id · git_commit · data_snapshot · features · params · constraints ·
train_window · validate_window · cost_assumptions · result · accept/reject reason`.

Then correct for the search: Probability of Backtest Overfitting, the Deflated Sharpe Ratio, White's
Reality Check, Hansen's SPA. A model enters production only if it survives most walk-forward
periods, higher-cost assumptions, parameter perturbation, removal of its best sessions, and every
regime — **and beats a shape-matched null through the replica** (§19.1).

### 19.1 Power first, then search

Everything above assumes the data can resolve the effect you are looking for. On SN88's own
replication track it cannot, and the arithmetic says so before any code runs: `score ∝ Sharpe^4.8`,
so a **1.5× Sharpe difference is a 7× score difference**, and 314 sessions hold roughly **six
non-overlapping 50-session windows** — two per half. Ordinary regime variation is amplified past
anything a search on this panel can distinguish.

This is not a warning in the abstract. It was measured, after a searched 10-day sector-neutral
reversal returned 8.859 on a properly held-out block against a 2.021 "null bar" — an apparent 4×
edge that survived a disjoint train/test split, and was false:

```
reversal 10d  (searched from 80 configs, frozen, held out)    8.859
random 100 names, same book shape, seed 1                    14.040   <- beats it
random 100 names, same book shape, seeds 2,3            3.078 / 3.631
shuffled reversal, same book shape, seeds 1-3     3.613 / 4.833 / 4.494
```

Two failures compounded, and both are common enough to name:

1. **The null had a different book shape.** The bar was 20 names equal-weighted; the candidate held
   100 names inverse-vol. The score pays richly for diversification (`vol^0.85` with `Sharpe^4.8`),
   so most of the "edge" was book construction credited to signal.
2. **The null was a point, not a distribution.** A no-signal book of the right shape spans 3.1–14.0
   across seeds. Any single number inside that range is unfalsifiable.

**The rule.** Before searching a family, measure what *no signal at all* scores at the same book
shape — same `top_n`, same weighting, same rebalance cadence — across at least eight seeds. That
spread is the resolution floor. If the effect you are hunting is smaller than the floor, the search
cannot find it and will return noise with a confident face. `sn88_replica.signals` provides
`shape_matched_null()` and `exceeds_null()`; a candidate must clear the **entire** null spread, not
its median, because at two independent windows anything less is not distinguishable from luck.

The same test applied to book shape itself: holding one trivial signal fixed, the best shape is
524-equal on sessions 100–207, 100-inverse-vol on 207–314, and 50-equal on 157–314. It is not
stable, so it is not knowable here either.

**What this implies for the build order.** The binding constraint is sample size, not model
quality, and more search on this panel produces more confident wrong answers rather than better
ones. Effort belongs in the **long-regime track** (§30) and in the operational reliability that
§21–§29 cover — uptime, correct fills, surviving rule changes — all of which pay off with no
statistical power required at all. See `research/README.md` for the full sequence.

**A caveat on the caveat.** None of this says no edge exists. It says *this panel cannot certify
one*. A signal with a strong economic prior may well be worth running in shadow (§24 stage 7) on
exactly the reasoning that the backtest is uninformative in both directions.

## 20. Test plan

| Level | What it covers |
|---|---|
| **Unit** | Each scoring term against hand-computed fixtures; the clip branch with 0, 1, 2, 3 and many positive sessions; the two day counts; every multiplier boundary. |
| **Property** | Scale invariance of `mar`/`lsr`/`odds` (§2.2); `odds == 50·(1+mean/pavg)`; monotonicity of score in gain; `sum(|w|) ≤ 1` and `'_':1` present for every generated candidate under randomised inputs. |
| **Golden** | Replica output vs captured `/score` responses, every row, <1e-6 relative. Re-capture weekly. |
| **Replay** | The full cycle against a frozen historical snapshot; byte-identical output on re-run. |
| **Leakage probes** | **Must fail the build.** Feed the session close into the MOC feature snapshot and assert detection (§5.5). Assert mode B never reads a return dated before the candidate's generation time (§13.1). Assert every feature's `available_time` precedes its use. Assert a model cannot load a feature-set version it was not trained on (§31). |
| **Fault injection** | Every row of §21 — stale data, delisted ticker, infeasible optimiser, API 500, clock skew, market holiday, malformed `/dist` payload (§33), publisher race (§29). |
| **Shadow** | The complete system running against live data, publishing to a log instead of `/rev`, reconciled daily against what the replica predicted in mode B. |

Shadow-test the **pipeline**, not the market exposure — 10 to 20 sessions, see §24.

The leakage row is not optional. Three of r3's errors were of this family, and every one of them
made the numbers look *better*. A backtest that silently improves when you introduce a bug is the
only kind of bug you will not go looking for.

## 21. Failure matrix

| Failure | Required action |
|---|---|
| Missing or stale price | Keep the previous validated allocation; touch the timestamp only. Alert. |
| Corporate action unreconciled | Drop the affected ticker from new decisions until reconciled. |
| **Ticker absent from today's `/assets`** | Remove it — otherwise it is silently reclassified to cash. |
| Ticker renamed | Block the cycle; symbol maps must be applied before weights are written. |
| Model inference failure | Fall back to the inverse-volatility or HRP baseline. |
| Optimiser infeasible | Relax soft constraints only. **Never** relax gross, cash or `'_'`. |
| Extreme model disagreement | Shrink expected returns toward the prior; hold scale. |
| Volatility / correlation shock | Rotate into TLT/GLD at full gross; **do not raise cash**. |
| Headroom below the inception floor | Governor policy from §15 takes over. |
| Market holiday / early close | Skip the decision, touch the timestamp — the calendar clock still runs. |
| `sum(|w|) > 1` at publish time | Hard veto. Never submit — it would be silently discarded. |
| Submission unconfirmed | Do nothing; the miner retries. Alert if unconfirmed after 5 minutes. |
| Constants changed upstream | **Freeze candidate selection** until the replica is updated and golden tests pass. |
| Unexplained live/backtest divergence | Safe mode: hold the last validated allocation, preserve full logs. |

---

# Part IV — Operating it

## 22. The daily cycle

Run it twice per session — once before 09:28 ET for MOO, once before 15:50 ET for MOC.

```python
def cycle(as_of, window):                      # window in {"MOO", "MOC"}
    rules = sync_rules()
    if rules.changed:
        freeze_selection(); alert("rules changed"); return

    if not is_trading_session(as_of):
        touch_timestamp_only()                 # calendar clock runs on weekends too — §3.4
        return

    data = load_point_in_time(as_of)
    if not quality_gates_pass(data):
        touch_timestamp_only()
        alert("data quality gate failed"); return

    field  = refresh_field_state()             # §17
    feats  = build_features(data)
    fc     = forecast_ensemble(feats)          # incl. P(down session)
    risk   = risk_engine(data, fc, field)      # incl. headroom to the $10M floor

    cands  = generate_candidates(fc, risk, previous_weights())
    ranked = [replica.project_incentive(c, field, rules) for c in cands]   # §13
    pick   = selector(ranked)                                              # §14
    pick   = governor(pick, risk)                                          # §15 — may veto all

    if pick.projected_incentive_gain > pick.round_trip_cost or days_since_refresh() >= 1:
        publisher.publish(pick)                                            # §16
        publisher.verify()
```

**Submit when** projected *incentive* gain beats round-trip cost; or a risk limit demands it; or the
regime changed materially; or a constraint is violated; **or it has been a calendar day** — §3.4
charges idleness from the first day and a timestamp-only refresh is nearly free.

## 23. Runbook

**Every calendar day, including weekends** — confirm the timestamp refreshed and `last_active` ≤ 1.

**Every session** — confirm the submission landed before the cutoff; check gross in [0.99, 1.00] and
cash ≤ 1%; check headroom to the $10M floor; check projected rank and marginal incentive; check
minimum distance to the field; confirm every held ticker is still in `/assets`.

**Weekly** — re-capture `/score` golden fixtures and re-run §20; reconcile realised vs projected
score; review turnover and realised fees against §12; review the trial ledger.

**On every alert** — `rules` diff (freeze selection until the replica is updated and golden tests
pass); `/ratio` change (reprice everything — a class-split or burn change alters your income
immediately); a delisting that touches your book; replica-vs-`/score` drift beyond tolerance.

**Never** — let the model disable the governor; submit an unvalidated file; run more than one hotkey
per coldkey without budgeting 256 alpha for each.

## 24. Build order

The conventional "build for three months, shadow-trade for three, then go live" sequence is close to
the **worst** possible sequencing here. §3.1 measured it: the same strategy shape earns 0.3% of its
session-50 score at session 10, 7.7% at 20, 19% at 30. The ramp, `mar`'s √days growth and the
inception anchor accrue **only on-chain**. On a trading-day clock that plan is ~6 months to first
emission plus ~2 more to full score — and the operational risk it retires is close to zero, because
the deliverable is a Python dict and the replica is a byte-exact offline evaluator on day 1.

But r3 over-corrected in the other direction, putting registration at stage 3 — before the data
plane, the governor and any monitoring existed. **The asset-class choice is irreversible and early
losses are permanent through the inception gate (§3.2), so registering into an unsupervised system
is not cheap optionality.** The revised order below keeps the ramp argument while refusing to
register something that cannot be watched or stopped.

| # | Stage | Acceptance test |
|---|---|---|
| **1** | **The replica and rule adapter (§13)** — `pl2sc` + `score` + the four multipliers + the gate, with the three evaluation modes kept separate. Adversarial golden tests. | Mode A reproduces `/score` to <1e-6 on every row. Mode B refuses to touch historical returns. |
| **2** | **Secure publisher and wallet isolation (§16, §29)** — validation, atomic write, submission verification, coldkey offline, job lock. | A deliberately malformed candidate is rejected, never written. |
| **3** | **MOO/MOC replay engine (§5.5, §13.1 mode C)** — separate snapshots, auction prices, no look-ahead. | A leakage probe (feeding the close into the MOC snapshot) is detected and fails the build. |
| **4** | **Data plane, quality gates and a safe long-only baseline (§8, §30).** | Reconstructs an arbitrary past session using nothing published after it. |
| **5** | **The survival-aware governor (§15, §27)** — inception floor, recovery policy, pruning model. | Randomised tests produce zero hard-constraint violations; recovery simulator runs. |
| **6** | **Monitoring, alerting, rollback, failure injection (§18, §21, §31).** | Every row of §21 exercised; rollback is one command. |
| **7** | **10–20 sessions of shadow operation** — the full loop publishing to a log, reconciled daily against the replica's projection. | Projected vs would-be-realised score agrees within tolerance; zero unhandled faults. |
| **8** | **Register** with the validated baseline. Refresh daily including weekends. | `a` column reads **1**; `last_active` ≤ 1; submission confirmed. |
| **9** | **Field intelligence (§17)** — projected rank, distance monitor, rule-change detector, rival decomposition. | Predicts today's `bin/validator` output before it runs. |
| **10** | **Forecasting and macro challengers (§10)**, promoted only through §31. **Run the §19.1 power check before searching, not after.** | Edge survives every walk-forward period at 2× costs in mode B **and clears the entire shape-matched null spread** (§19.1). On the 314-session replication panel nothing has yet cleared it, so expect this stage to gate on data rather than on modelling. |
| **11** | **Adjust risk** — only on statistically defensible live evidence, within the §12 constrained search and the §11.2 stress suite. | Attribution (§32) shows the change came from alpha, not beta. |

**Why a 10–20 session shadow, and not the six months r2 proposed nor the zero r3 implied.** The
ramp arithmetic is real — nothing accrues offline, and 30 sessions is six calendar weeks — so a long
shadow is genuinely expensive. But the operational risks that shadow retires are also real and
cheap to retire: a silent `asum > 1` discard, an MOC leak, a publisher race, an unvalidated ticker.
Ten to twenty sessions costs you a small fraction of the ramp and catches all four.

Two things not to get wrong in the first 30 sessions once you do register: **early drawdown is
permanent** (§3.2), and the MAR floor is three times harsher at session 5 than at session 50 (§2.6),
so early volatility is amplified in both directions. **Start conservative and let the floor
loosen** — and note this is a genuine exception to §2.8's steady-state reasoning, not an
inconsistency.

## 25. What not to build

- **A signal chosen by its family name.** §2.7 — measure realised post-cost skew and let the replica
  rank it. Do not adopt momentum because it is standard, and do not reject it because it is called
  momentum.
- **A signal compared against a differently-shaped null.** §19.1 — a 100-name inverse-vol book beat
  a 20-name equal-weight "bar" by 4× on held-out data with *no signal in it at all*. Match `top_n`,
  weighting and cadence, or you are measuring book construction and calling it alpha.
- **More search on the 314-session panel.** §19.1 — it holds ~6 non-overlapping windows and
  `score ∝ Sharpe^4.8`. The search does not fail loudly; it returns confident noise. Spend the
  effort on the long-regime track and on uptime instead.
- **A fixed volatility target — in either direction.** §2.2 and §12: scale is a search variable
  under survival constraints, not a number you pick. Neither "target 12%" nor "maximise feasible"
  is a policy.
- **Any candidate ranking that replays weights over known history.** §13.1 — this is the look-ahead
  bug r3 shipped. Put a leakage probe in CI.
- **A cash buffer or a T-bill sleeve.** §3.5, §6.1 — de-risk into TLT or GLD instead.
- **Shorts in v1.** Permitted, but counted as cash *and* they consume gross. A 30% short book costs
  30% of score before it earns anything.
- **A similarity-avoidance subsystem.** §3.3 — the closest live stock pair is 5.6× the threshold.
- **A market-impact model.** §6.3 — fills are simulated at session prices with a flat fee.
- **Bittensor request-serving logic.** `forward()` is dead template code; no validator queries you.
- **A GPU cluster.** `min_compute.yml` requires none.
- **A submission retry loop.** The miner already retries every ~11 seconds (§5.3).
- **Reinforcement learning in v1.** The 2026 systematic review of 156 DRL portfolio studies warns
  that reported improvements are confounded by datasets, infrastructure, tuning and regime — direct
  comparison of published Sharpe ratios is unsafe. A challenger, never a v1.
- **News-driven intraday trading.** MOO/MOC execution forbids it and the clip punishes it.
- **A six-month shadow period before your first submission.** §24.

## 26. Risks you are actually taking

- **The stock class is the crowded half.** 151 miners for the same 50% of emission that 75 compete
  for on the other side. 44 of the 92 non-earning UIDs are stock miners with negative Sharpe.
- **Alpha issuance far outruns the TAO flowing in.** SN88 mints 7,200 alpha/day to participants
  while the protocol currently routes only **~1.60 TAO/day** into the subnet. You are paid in a
  token whose issuance greatly exceeds the TAO backing it. `[LIVE]`
- **A halving is scheduled.** SN88's own alpha halving cuts the miner pool from 2,952 to 1,476 when
  cumulative issuance crosses 10.5M — currently ~5.14M, growing ~7,600/day. Roughly two years. `[LIVE]`
- **Reward concentration.** The top 10 stock miners take 68.8% of the class. Outside the top 20 this
  is not a business.
- **Owner-controlled scoring.** Blacklist, whitelist, similarity threshold, class ratio, burn valve,
  `cash` and `last` — all served from an unsigned plain-`http` API (§3.6).
- **Unannounced rule changes**, auto-pulled hourly by every validator.
- **Public allocations.** Your holdings are visible immediately; only your reasoning is private.
- **The similarity grief vector** (§3.3) targets whoever is winning.
- **Deregistration.** 256 UIDs, full, 3-day immunity, pruned on incentive.
- **Two stacked price risks:** your incentive share, and the alpha/TAO rate.
- **The objective's own bias** (§2.8). A scoring function that truncates your right tail and cannot
  see beyond 50 sessions will happily pay you to accumulate a short-volatility exposure whose left
  tail has not yet arrived.
- **Your own model of the objective.** Every elasticity in §2 came from a simulation, and a
  simulation is a claim about the world. r3 shipped three errors of exactly this kind — an
  overstated invariance, a confounded live inference, and a look-ahead in the ranker. Treat §11.2
  and the §20 leakage probes as the load-bearing defence, not as diligence theatre.

---

# Part V — Governance and survival

Everything in Parts II–IV assumes the miner keeps its UID, can afford to run, cannot be tampered
with, and can explain its own results. None of that is automatic.

## 27. Survival and deregistration model

§26 lists pruning as a risk; this module makes it a number. It is an input to the recovery policy
(§15) and to any decision to raise risk.

Track and forecast:

```
uid_capacity          = 256, currently full
your_immunity_expiry  = registration_block + 21600            # 3 days
your_incentive_pctile = rank(you) / active_uids               # from §17
pruning_threshold     = incentive of the lowest non-immune UID
headroom_to_prune     = your_incentive − pruning_threshold
p_prune(H)            = P(your incentive < threshold within H sessions)
recovery_time         = E[sessions to climb back above the inception floor]
sunk_registration     = 256 alpha + 0.01 TAO burn
competitor_arrival    = new registrations per week
```

Two consequences the rest of the system must respect:

- **A zero score is not merely zero income.** It moves you toward the bottom of the incentive
  distribution, which is the pruning queue. The cost of a breach is the emission you lose *plus*
  `p_prune × sunk_registration` *plus* the entire ramp you would have to serve again.
- **Recovery time is a first-class quantity.** Combined with the 30-session ramp and the ~√days
  growth in `mar` (§2.5), being pruned and re-registering is materially worse than a long drawdown.

## 28. Economic viability

The guide reports emission numbers; it has not until now answered whether running the system is
rational. Build the model before you register, not after.

```
Net = TAO_value(rewards)
    − registration_cost            # 256 alpha + 0.01 TAO burn, per coldkey
    − alpha_to_TAO_slippage        # unstaking through the subnet pool
    − data_costs                   # §30 — the largest controllable line
    − infrastructure               # §E.2
    − research_time                # value it honestly
    − stake_opportunity_cost       # 192 alpha locked
    − p_prune × sunk_registration  # §27
```

Compute the **break-even rank** and re-compute it under a grid of scenarios, because three of the
inputs are outside your control:

| Scenario axis | Range to test |
|---|---|
| Alpha/TAO price | ×0.25 to ×4 of today's 0.0039 |
| TAO/USD | ×0.5 to ×2 of today's ~$222 |
| Class emission ratio (`/ratio`) | 0.3 to 0.7 to the stock class |
| Burn valve (`sum(ra) < 1`) | 0% to 50% diverted |
| Alpha halving | pool 2,952 → 1,476 |

At today's numbers the whole stock class is ~$1,280/day across 92 earning UIDs. If your honest
break-even is rank 25 and your model gets you to rank 40, the correct decision is not to launch.

## 29. Security architecture

This matters more here than in a typical quant stack for one specific reason: **the official miner
executes upstream code on your production host, hourly.** `etc.update()` runs
`git pull && pip install -e .` and exits for pm2 to restart it (§0.3). That is an unsigned,
unreviewed code path into the machine holding your hotkey.

**Wallet**

- **Coldkey never touches the production host.** Keep it offline; it holds the 192 staked alpha.
- Only the required **hotkey** on the VPS, with least privilege.
- Encrypted backups, and a *tested* recovery procedure — untested backups are not backups.

**Host**

- Non-root service account; SSH keys only, no passwords; firewall with minimal exposed ports.
- Pin and lock dependencies; record the commit hash the miner is running.
- **Monitor the pulled commit.** Alert on every upstream change, diff it against your `rules`
  mirror, and freeze candidate selection until you have read it (§21).
- Isolate the miner process (separate user, container, or both) from anything holding secrets.

**Pipeline**

- Checksums on model artefacts; signed deployment manifests; secrets in a manager, not in files.
- **A job-level lock** so two publishers can never race to write the strategy file — a partial write
  or an interleaved write is a silent `asum > 1` discard (§4.1).
- Audit logging of every published weight vector and the inputs that produced it.
- A standby VPS with a documented failover, and staging separated from production.

## 30. The data contract

§E.3 argues data outranks hardware. This is the specification.

| Item | Requirement |
|---|---|
| Primary vendor | Point-in-time, with delisted names and historical index/universe membership |
| Secondary vendor | Independent enough to catch the primary's errors |
| Adjustment method | Documented, and *reproduced* by you rather than trusted |
| Revision handling | Store the original value **and** each revision, with `available_time` on both |
| Reconciliation | Daily cross-vendor diff; thresholds that quarantine a name rather than trading it |
| Disagreement policy | What happens when vendors differ by more than the threshold |
| News/event licensing | Check terms before ingesting anything for the §10 overlay |
| Retention | Raw pulls immutable; everything else derived and reproducible |

The 555-name `/assets` universe is a snapshot of *today* and is survivorship-selected. Reconstruct
historical membership from your vendor; do not assume it.

## 31. Model lifecycle governance

"Ship versioned artefacts" is not a lifecycle. Define:

```
registry entry:
  artifact_hash · feature_set_version · training_window · git_commit
  walk_forward_results · DSR · PBO · trial_ledger_ids
  promotion_criteria_met · approver · promoted_at · rolled_back_at
```

- **Champion / challenger.** A challenger runs in shadow (mode B, §13.1) against the champion for a
  defined number of live sessions before promotion is even considered.
- **Promotion criteria fixed in advance** — not chosen after seeing the result.
- **Feature-version compatibility** enforced: a model may not load a feature set it was not trained
  against.
- **Rollback** is a tested one-command operation, not a redeploy.
- **Drift and calibration monitoring** on inputs and predictions, with alerts.
- **Manual approval required** for any change to the risk policy, the governor, or the scale band.
  Model weights may auto-promote; risk limits may not.

## 32. Performance attribution

When your score moves, you must be able to say why. Without attribution you cannot distinguish
alpha from beta, and you will scale up luck.

Decompose each session's contribution across:

- Individual names, and sectors
- Market beta, and the size / value / momentum / quality factors
- The regime overlay's adjustments, and the macro overlay's (§10)
- Rebalancing decisions versus drift
- Transaction fees
- Each SN88 multiplier separately: clip, ramp, gate, similarity, inactivity, cash
- Validator rule changes

The last two lines matter uniquely here: a score drop caused by the cash multiplier or an upstream
constant change looks identical to a strategy drawdown on the dashboard, and calls for the opposite
response.

## 33. API integrity and adversarial inputs

§3.6 establishes that `cash`, `last`, the distance matrix, the class ratio, the blacklist and the
similarity threshold all arrive over **unsigned plain HTTP** from a single operator. You cannot make
that source trustworthy — you can make your system refuse implausible input.

```
on every fetch:
  validate against a strict JSON schema
  range-check every field           # cash in [0,1], last >= 0, distances in [0, 2], ...
  check timestamps are monotonic
  hash and archive the raw response
  diff against the previous snapshot
  cross-check what is checkable against the chain (incentive, tempo, UID count)
  if change exceeds the plausibility threshold:
      quarantine, alert, freeze selection, keep the last good snapshot
```

And one hard rule: **field intelligence may never, by itself, trigger a large position change.** It
informs projected rank and the similarity monitor. If a corrupted or manipulated `/dist` or `/days`
response could move your book, the design is wrong.

---

## Appendix A — Constants

`Investing/core/const.py` at `main`, 2026-09-01. **Diff this on a schedule.** `[CODE]`

| Constant | Value | Governs |
|---|---:|---|
| `FIRST_DATE` | `2025-03-20` | Simulation epoch |
| `API_ROOT` | `http://api.investing88.ai` | **Plain http, unsigned** |
| `DD_POWER` | `1` | Similarity recovery exponent |
| `DD_TRIGGER` | `0.01` | Similarity distance threshold (API-overridable) |
| `DEC_UID` | `0` | Burn UID |
| `DEC_DECAY` / `DEC_CUTOFF` | `3` / `0.03` | Subnet-level DEC — **dead, both consumers commented out** |
| `DEC1_DECAY` | `2` | Inactivity exponent |
| `DEC1_START` | `5` | Inactivity guard, on **track age** |
| `DEC1_CLIFF` | `20` | Inactivity zero point |
| `CASH_DECAY` | `1` | Cash penalty exponent — linear |
| `CASH_RESIDUE` | `0.01` | Cash penalty floor |
| `DAYS_FINAL` | `30` | Ramp length **and** similarity recovery length |
| `DAYS_DELAY` | `1` | Ramp exponent — **tunable** |
| `CLIP_OUTLIERS` | `2` | Positive sessions clipped |
| `CLIP_DEFAULT` | `1` | Clip value when < 3 positive sessions (percent) |
| `RISK_INIT_STK` | `5` | MAR floor numerator |
| `WIN_SIZE_STK` | `50` | Scoring window, **in trading sessions** |
| `STK_MOO` / `STK_MOC` | `-120` / `-600` | Seconds before open/close → 09:28 / 15:50 ET |
| `STK_FEE` | `0.002` | 0.2% per side |
| `STK_BENCH` | `SPY` | Benchmark |
| `STK_TZ` | `America/New_York` | Session calendar |

Chain-side `[LIVE]`: `Tempo[88] = 360`, `ImmunityPeriod[88] = 21600` (3 days),
`MaxAllowedUids[88] = 256`, `MaxAllowedValidators[88] = 64` (12 filled), `Burn[88] = 0.0100 TAO`.

## Appendix B — The score replica (reference implementation)

Faithful port of §2.1. Verified against the live `/score` table to <1e-6 relative on all rows.

```python
import math

WIN, RISK_INIT, CLIP_N, CLIP_DEFAULT = 50, 5.0, 2, 1.0


def kelly(p, b):
    return (p * (b + 1) - 1) / b


def drawdown(pct):
    peak = down = cum = 0.0
    for x in pct:
        cum += x
        peak = max(peak, cum)
        down = max(down, peak - cum)
    return down


def raw_score(swap_open0, swap_closes):
    """swap_closes: the windowed daily swap_close series (<=50 sessions)."""
    sc, n, init = list(swap_closes), len(swap_closes), swap_open0
    pnl, pct = [0.0] * n, [0.0] * n
    pnl[0] = sc[0] - init
    pct[0] = pnl[0] / init * 100
    for i in range(1, n):
        pnl[i] = sc[i] - sc[i - 1]
        pct[i] = pnl[i] / sc[i - 1] * 100

    # clip: top N+1 positive sessions -> the (N+1)th value, then rebase the equity curve
    pos = sorted([i for i in range(n) if pct[i] > 0], key=lambda i: -pct[i])[:CLIP_N + 1]
    if pos:
        clip = pct[pos[-1]] if len(pos) == CLIP_N + 1 else min(CLIP_DEFAULT, pct[pos[-1]])
        for i in pos:
            pct[i] = clip
        for i in sorted(pos):
            op = sc[i - 1] if i else init
            delta = sc[i] - op * (1 + clip / 100)
            for j in range(i, n):
                sc[j] -= delta
            pnl[i] = op * clip / 100

    ppos = [x for x in pct if x > 0]
    pneg = [x for x in pct if x < 0]
    prob = len(ppos) / n
    gain = max(sc[-1] - init, -init) / init * 100
    risk = drawdown(pct)
    daily = ((1 + gain / 100) ** (1 / n) - 1) * 100
    mar = gain / max(risk, RISK_INIT / n ** 0.5)
    lsr = sum(pnl) / (sum(abs(x) for x in pnl) or 1e18)
    if not ppos or not pneg:
        odds = prob * 100                                  # the NaN branch
    else:
        odds = 50 + kelly(prob, (sum(ppos) / len(ppos)) / (-sum(pneg) / len(pneg))) / 2 * 100
    odds = max(odds, 0.0)
    return max(mar * lsr * odds * daily, 0.0), dict(
        gain=gain, risk=risk, mar=mar, lsr=lsr, odds=odds, daily=daily)


def adjusted_score(raw, *, lifetime_sessions, first_swap_open, last_swap_close,
                   dedupe=None, last_active=0, cash=0.0):
    """Apply the etc.py multipliers, in order."""
    s = raw
    if lifetime_sessions < 30:                              # ramp — LIFETIME sessions
        s *= (lifetime_sessions / 30) ** 1
    s *= int(last_swap_close >= first_swap_open)            # inception gate
    if dedupe is not None:
        s *= dedupe
    if lifetime_sessions > 5:                               # inactivity — CALENDAR days idle
        s *= 1 - min((last_active / 20) ** 2, 1)
    s *= min(max(1 - cash, 0.01), 1)                        # cash & short
    return s

# then: renormalise the stock class's summed score to 50% of the total,
#       and divide by the grand total to obtain projected incentive share.
```

Inputs:

```bash
curl -s http://api.investing88.ai/score   # authoritative raw scores — golden fixtures
curl -s http://api.investing88.ai/pnl     # full equity curves, every UID
curl -s http://api.investing88.ai/days    # rank, class, sessions, last, cash
curl -s http://api.investing88.ai/dist    # distance matrix + blacklist/whitelist/threshold
curl -s http://api.investing88.ai/ratio   # class split
curl -s http://api.investing88.ai/assets  # 555 tickers, cash-classified list, delisting list
```

## Appendix C — Configuration template

```yaml
# config/system.yaml
class: stocks                     # '_': 1 — IRREVERSIBLE once submitted

constraints:
  gross_exposure:   [0.99, 1.00]
  cash_weight:      [0.00, 0.01]  # includes the 16 cash-classified tickers
  allow_shorts:     false
  max_single_name:  0.10
  max_sector:       0.25
  holdings:         [15, 40]
  turnover_normal:  [0.00, 0.05]
  turnover_hard:    0.15

scale:
  policy: maximise_expected_incentive_subject_to_survival   # §12 — NOT maximise_feasible
  search_bands_ann_vol: [0.08, 0.12, 0.16, 0.20, 0.25, 0.32]  # searched, not targeted
  constraints:
    p_inception_breach_max: 0.05      # over the governor horizon
    expected_shortfall_max: 0.06      # multi-session
    gap_loss_max:           0.04
    regime_stress_loss_max: 0.10
  path_generator: stress_suite        # §11.2 — NOT iid gaussian
  levers: [high_beta_names, tqqq_capped]

tqqq:
  baseline_weight: 0.00
  max_weight:      0.10
  forbidden_when:  [headroom_below_floor_warn, vol_regime_elevated]

evaluation:
  modes: [reconciliation, forward_projection, policy_backtest]   # §13.1
  selector_may_use: [forward_projection]                          # ONLY
  leakage_probe_in_ci: true

governor:
  floor_anchor:  10000000         # inception capital, USD
  floor_warn:    0.03             # headroom below which scale-ups are blocked
  floor_crit:    0.01             # headroom below which the safest book is forced
  below_floor_policy: documented_in_advance   # §15

execution:
  windows: [{name: MOO, cutoff_et: "09:28"}, {name: MOC, cutoff_et: "15:50"}]
  fee_per_side: 0.002

cadence:
  refresh: every_calendar_day     # inactivity is charged on a calendar clock — §3.4
  max_idle_days: 1

alerts:
  rules_diff:        freeze_selection
  ratio_change:      page
  replica_drift:     0.000001
  ticker_delisted:   page
  submission_unconfirmed_seconds: 300
  min_field_distance: 0.02        # 2x the similarity trigger
```

## Appendix D — Reading list

All verified to exist with correct attribution. Four widely propagated mis-summaries are flagged.

**Portfolio construction**
- Markowitz (1952), *Portfolio Selection*, JF 7(1) 77 — the E-V efficient frontier. ⚠️ Frequently
  miscredited with the estimation-error critique; that is Michaud (1989), Best & Grauer (1991),
  Chopra & Ziemba (1993).
- DeMiguel, Garlappi & Uppal (2009), RFS 22(5) 1915 — 14 models, none beats 1/N out of sample.
- Ledoit & Wolf (2004), *Honey, I Shrunk the Sample Covariance Matrix*, JPM 30(4).
- Black & Litterman (1992), FAJ 48(5) 28.
- Maillard, Roncalli & Teiletche (2010), equal risk contribution, JPM 36(4) 60.
- López de Prado (2016), Hierarchical Risk Parity, JPM.
- Rockafellar & Uryasev (2000), *Optimization of CVaR*, J. Risk 2(3) 21. ⚠️ The result that matters
  is that CVaR minimisation reduces to a **linear program** — that is why the term is tractable.

**Forecasting and costs**
- Gu, Kelly & Xiu (2020), *Empirical Asset Pricing via Machine Learning*, RFS 33(5) 2223.
- Novy-Marx & Velikov (2016), *A Taxonomy of Anomalies and Their Trading Costs*, RFS 29(1) 104 —
  directly relevant to §12: high turnover destroys apparent alpha.
- Lim et al. (2019), Temporal Fusion Transformers, arXiv:1912.09363.
- Gibbs & Candès (2021), Adaptive Conformal Inference Under Distribution Shift, arXiv:2106.00170.
- Rout, Acharya, Aithal & Sharma (2026), *Systematic review of reinforcement learning for automated
  equity portfolio management*, Discover Computing 29(1) art. 431 — 156 studies, PRISMA.

**Not fooling yourself**
- Bailey, Borwein, López de Prado & Zhu (2013), Probability of Backtest Overfitting.
- Bailey & López de Prado (2014), The Deflated Sharpe Ratio.
- Harvey, Liu & Zhu (2016), *…and the Cross-Section of Expected Returns*, RFS 29(1) 5.
- White (2000), *A Reality Check for Data Snooping*, Econometrica 68(5) 1097.
- Hansen (2005), *A Test for Superior Predictive Ability*, JBES 23(4) 365.
- López de Prado (2018), *Advances in Financial Machine Learning*, ch. 7 — purged CV and embargo.

**Regime and macro**
- Hamilton (1989), Econometrica 57(2) 357 — Markov regime switching. ⚠️ Use this for regime change,
  **not** Engle's ARCH (1982), which models conditional heteroskedasticity inside a stationary
  process and is evidence for volatility clustering only.
- Adams & MacKay (2007), Bayesian Online Changepoint Detection, arXiv:0710.3742.
- Daniel & Moskowitz, *Momentum Crashes*, NBER w20439 — cross-sectional momentum's left tail, and
  why §2.7's skew argument cannot be applied to a strategy by its label.
- BIS, *The correlation of equity and bond returns*, Quarterly Review, Dec 2023 — the equity-bond
  correlation is regime-dependent; the basis for the hedge table in §6.1.
- SEC, *Leveraged and Inverse ETFs* investor bulletin — daily-objective compounding, the basis for
  the TQQQ constraints in §6.2.
- Caldara & Iacoviello (2022), *Measuring Geopolitical Risk*, AER. ⚠️ The published abstract
  concludes the damage is driven by **both** the threat and the realisation — do not cite it to
  justify discounting unconfirmed threats. Free daily index:
  <https://www.matteoiacoviello.com/gpr.htm>
- Baker, Bloom & Davis (2016), *Measuring Economic Policy Uncertainty*, QJE 131(4) 1593. Free daily
  indices: <https://www.policyuncertainty.com/>
- Loughran & McDonald (2011), *When Is a Liability Not a Liability?*, JF 66(1) 35. General-purpose
  sentiment dictionaries perform badly on financial text. Free dictionary:
  <https://sraf.nd.edu/loughranmcdonald-master-dictionary/>

## Appendix E — Hardware and infrastructure

### E.1 The headline: you do not need a GPU

`min_compute.yml` requires **no GPU** for either miner or validator, and asks only for Ubuntu 22.04
and 20–100 GB of storage. `[CODE]` That covers the SN88 client. The real question is what *your
research stack* needs — and at this data scale the answer is **CPU cores and RAM, not accelerators.**

A 555-name US equity panel is small by modern standards:

| Dataset | Rows | Raw size |
|---|---:|---:|
| Daily bars, 20 years | 2.8M | ~0.3 GB |
| 30-minute bars, 5 years | 9.1M | ~0.9 GB |
| 30-minute bars, 15 years | 27.3M | ~2.7 GB |
| 1-minute bars, 5 years | 272.7M | ~27 GB |
| Feature matrix, 300 cols × 20y daily (float32) | 2.8M | **3.4 GB** |
| Feature matrix, 1000 cols × 20y daily (float32) | 2.8M | 11.2 GB |

**The entire feature matrix fits in RAM.** A LightGBM fit on 2.8M × 300 takes roughly 90 seconds on
16 cores. There is nothing here that a GPU meaningfully accelerates — GBDT on tabular data of this
size is memory-bandwidth bound, and the 3–5× a GPU might give you is dwarfed by simply having more
cores for the search.

**Where the compute actually goes is walk-forward validation** (§19), and it is embarrassingly
parallel:

| Search size | Fits | Wall time @ 16 cores |
|---|---:|---:|
| 15 folds × 20 configs × 1 seed | 300 | ~7.5 h |
| 15 folds × 50 configs × 3 seeds | 2,250 | ~56 h |
| 15 folds × 100 configs × 5 seeds | 7,500 | ~188 h |

That table is also an argument for discipline: the third row is a week of compute *and* a
multiple-testing problem that the Deflated Sharpe Ratio will punish (§19). Buy cores to run a
**small, pre-registered** search fast, not a large one slowly.

### E.2 Three tiers

**Tier 0 — the production miner. Required. Cheap.**

| | |
|---|---|
| Spec | 2–4 vCPU, 8 GB RAM, 100 GB SSD, Ubuntu 22.04 |
| Runs | `neurons/miner.py` under pm2, the publisher (§16), the daily cycle (§22), the replica (§13) |
| Cost | ~$20–40/month VPS |
| Critical property | **Uptime.** The inactivity penalty is charged on a calendar clock (§3.4), so an outage over a long weekend costs ~3% of score for nothing. Use a monitored VPS, not a laptop. |

Put it in a US region for low latency to the API and to your data vendor, and keep its clock
disciplined by NTP — the MOO/MOC cutoffs (§5.4) are hard.

**Tier 1 — the research box. Where the actual work happens.**

| | |
|---|---|
| CPU | 16–32 cores. Ryzen 9 9950X (16C/32T) is the value pick; Threadripper 7960X (24C) if you want headroom |
| RAM | **64 GB minimum, 128 GB comfortable** — you want the whole panel plus several feature matrices resident |
| Storage | 2 TB NVMe. Parquet + DuckDB; keep raw vendor pulls immutable and derive everything else |
| Cost | ~$2,000–4,000 as a workstation, or spot instances for bursts |

This is the machine that matters. Cores buy you parallel walk-forward folds; RAM buys you the
ability to hold the feature matrix and avoid a disk round-trip per fit.

**Tier 2 — a GPU. Optional, and only for challengers.**

| | |
|---|---|
| When | You are training sequence models (TFT, LSTM, transformers) or running DRL experiments |
| Spec | One 24–32 GB card — RTX 4090 / 5090, or an A6000 (48 GB) if you want room |
| Not needed for | GBDT, linear models, shallow nets, conformal methods, the replica, the daily cycle |

Per §10.1 these are v2 challengers. Buying the GPU first is the classic way to spend $2,000 on the
tier of model least likely to beat your LightGBM baseline through the replica.

**LLMs.** If you build the macro overlay, use a hosted API for structured event extraction — it is a
few thousand tokens per event and does not justify local hosting. If you insist on running locally,
a 7–8B model quantised onto a 24 GB card is sufficient for extraction into a fixed JSON schema.

### E.3 The budget line that actually matters is data, not silicon

For a US equity system the binding constraint is **point-in-time data quality**, and it costs money
rather than hardware:

- **Delisted names and ticker changes.** The 555-name `/assets` universe is itself a
  survivorship-selected snapshot of *today*. A backtest built on it without historical membership
  will look far better than reality. This is the single most common way an equity backtest lies.
- **Fundamentals with original publication dates**, not restated values stamped to period end.
- **A trustworthy corporate-actions history** — splits, dividends, spin-offs.

Free sources will get you started and will also give you a survivorship-biased backtest that
overstates every Sharpe you measure. Budget for one paid point-in-time vendor before you budget for
a GPU; a $2,000 card cannot fix a leaking dataset, and §19's whole purpose is to stop you believing
one.

### E.4 Separate the boxes

Run the miner and the research stack on **different machines**. The miner needs boring, monitored
uptime and almost no compute; research is bursty, occasionally saturates every core, and is exactly
the kind of workload that will OOM a box at 15:47 ET on a day you needed the MOC window. Ship
models to the miner as versioned artefacts; never train on the production host.

---

*Not investment advice. Past and simulated performance guarantee nothing. Every `[LIVE]` figure was
true on 2026-09-01 and is drifting as you read this.*
