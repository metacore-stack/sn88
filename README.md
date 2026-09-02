# SN88 US Equity Portfolio Management System

Stages 1-7 of the build order, plus the replication track and the baseline bar in [SN88-DEVELOPMENT-GUIDE.md](SN88-DEVELOPMENT-GUIDE.md): an exact
replica of the SN88 scoring pipeline used as the candidate ranker, a secure publisher that refuses
anything upstream would silently discard, a MOO/MOC replay engine that makes look-ahead
structurally impossible, a point-in-time data plane with the baseline portfolios every model has to
beat, a survival-aware risk governor, and the monitoring, alerting and rollback that have to exist
before any of it is allowed near a registered hotkey, and the shadow harness that gates
registration itself.

Pure standard library. No dependencies, no GPU, Python 3.10+.

## Status

| Component | State |
|---|---|
| `sn88_replica.scoring` — `pl2sc` + `score` | **done**, validated |
| `sn88_replica.multipliers` — the five `etc.py` multipliers + gate + burn | **done** |
| `sn88_replica.pipeline` — full-field scoring → projected incentive | **done** |
| `sn88_replica.modes` — three evaluation modes with the leakage guard | **done** |
| `sn88_replica.api` — client, fixture archive, integrity checks | **done** |
| `sn88_replica.universe` — /assets, cash list, delisting queue | **done** |
| `sn88_replica.strategy` — upstream parser prediction + validation | **done** |
| `sn88_replica.publisher` — atomic locked write + confirmation | **done** |
| `sn88_replica.calendar` — NYSE sessions, DST-correct cutoffs | **done** |
| `sn88_replica.execution` — MOO/MOC fills, fees, the first-day trap | **done** |
| `sn88_replica.replay` — mode C engine + the look-ahead proof | **done** |
| `sn88_replica.pit` — point-in-time store, revisions, horizons | **done** |
| `sn88_replica.baselines` — 1/N, inv-vol, ERC, min-var, HRP | **done** |
| `sn88_replica.governor` — floor-anchored vetoes + recovery policy | **done** |
| `sn88_replica.monitor` — metrics, alerts, rule-change detector | **done** |
| `sn88_replica.registry` — promotion criteria, rollback | **done** |
| `sn88_replica.shadow` — shadow log + the registration gate | **done** |
| `sn88_replica.market` — replication track, PIT market loader | **done** |
| **Register (stage 8)** | **not done — the gate says NOT READY** |
| Forecasting, field intelligence, scale-up (stages 9-11) | not started |


## Acceptance

The replica reproduces the live `/score` table published by the subnet's own validators:

```
/score golden: 226 rows  max=6.789e-08 (uid 203)  median=1.275e-12
component terms: mar=6.83e-09 lsr=2.07e-09 odds=6.88e-11 daily=4.82e-09 return=5.95e-10 swap=2.36e-13
```

Tolerance is 1e-6 (guide §13.2); measured worst case is **6.8e-08**. The full pipeline puts the top
miner at 11.792%, against 11.802% from a direct chain query.

## Quick start

```bash
python3 -m unittest discover -s tests        # 222 tests; needs network unless fixtures exist
python3 tools/archive_fixtures.py            # snapshot every endpoint — run this DAILY
python3 tools/rank_field.py --class stocks --top 25
python3 tools/rank_field.py --uid 54         # one miner, every multiplier broken out
SN88_OFFLINE=1 python3 -m unittest discover -s tests   # replay against the archive
```

`tools/rank_field.py` reproduces what `Investing/bin/validator` prints and adds the projected
incentive share it does not compute. It needs no wallet, no stake and no registration.

## Three things this package exists to prevent

**1. Trusting `simst`.** `Investing/bin/simst` applies the window, the clip and the ramp — and
nothing else. The inception gate, dedupe, inactivity, cash and class normalisation live only in
`etc.py`. A `simst` backtest can report a healthy score for a book validators score at zero.

**2. The clip's iteration order.** Upstream iterates the clipped sessions in **descending `pnl%`**
order, not index order, and the equity rebase is order-dependent. Using index order produces up to
**12% error** on live rows. Locked in by `test_clip_order_is_descending_pnl_not_index`.

**3. Look-ahead in candidate ranking.** `modes.py` keeps three evaluation modes structurally
apart. Mode B (forward projection) is the only one the selector may call, and it raises
`LeakageError` if the forward paths would start on or before the candidate was generated. Mode A
takes no candidate argument at all — enforced by a test that inspects the signature.

## Stage 2: what the publisher refuses

Upstream's parser fails silently in four ways, so `strategy.py` reproduces its semantics
and `tools/check_strategy.py` tells you which one you just hit:

| Input | What upstream does |
|---|---|
| `{'_':1,'AAPL':0.99}  # top pick` | sanitiser leaves `...}toppick` → `SyntaxError` → **silently discarded** |
| `{'_':1,'AAPL':(0.1+0.2)}` | parens stripped, `+` survives → weight silently becomes `0.30000000000000004` |
| `sum(|w|) > 1` | **silently discarded**; your previous allocation stays live |
| `'SGOV'` or a typo'd ticker | **silently reclassified to cash** → linear score penalty |
| `'_'` omitted | selects the ALPHA class, permanently |

The publisher writes via temp + `fsync` + atomic `rename` under an advisory `flock`, forces
mtime strictly forward, and confirms `.last-update` advanced — which is the only evidence the
miner actually got a `<= 201` from the API. It never touches the network and never touches a
wallet: you write the file, the miner daemon POSTs it.

## Stage 3: how look-ahead is made impossible

The leak is specific and invisible: an MOC decision is made at 15:50 ET, but a daily bar **is**
the close, so sharing one daily feature set between both execution windows hands the 15:50 model
the price it is supposed to predict. Results improve, which is why nobody investigates.

Three defences, increasing in strength:

1. **Typed snapshots.** A `FeatureSnapshot` is stamped with its session and window and raises
   `LeakageError` if used for the other one.
2. **A visibility horizon.** Every snapshot carries `as_of`; `visible()` filters timestamped
   records against it, so "what did I know" is answered by the snapshot, not by discipline.
3. **Poison replay.** `assert_no_lookahead()` runs the policy twice — once normally, once with
   every post-cutoff observation replaced by NaN — and asserts the decisions are identical.
   A policy that peeks produces different output and fails the build. That is a proof.

The calendar is DST-correct and rule-based with override tables for unscheduled closures.
Two details that catch hand-rolled calendars:

- **Early closes move the MOC cutoff to 12:50 ET**, not 15:50. Hard-coding the wall-clock time
  misses the window entirely on those sessions.
- **The two clocks disagree.** 30 sessions of ramp is 43 calendar days; the 50-session window is
  71. The inactivity penalty runs on the calendar one.

And the fill model reproduces `pldaily1` exactly, including the **first-day trap**: a miner's
*first ever* strategy submitted after the MOC cutoff is force-initialised to **100% cash**.

## Stage 4: point-in-time data and the baselines

Every observation carries `event_time` (when it happened) and `available_time` (when you could have
known it). Reading by the first is the most common way an equity backtest lies. `PointInTimeStore`
makes it awkward: **there is no method that returns "the latest value" without a horizon**, and a
test asserts that. Restatements are first-class — a revised fundamental is a new observation, and
`asof()` returns what you would have seen then, not what the vendor believes today.

Baseline scores through the replica, on synthetic data:

```
equal_risk_contribution      0.3291
equal_weight                 0.3185      <- 1/N is right at the top, as DeMiguel et al. predict
inverse_volatility           0.2583
hierarchical_risk_parity     0.1732
minimum_variance             0.0460      <- LAST, and that follows from §2
```

Minimum variance coming last is not a bug. The score is steeply increasing in return and the MAR
floor stops paying for drawdown reduction below ~3% annualised vol, so a portfolio that buys
variance reduction with return is buying the wrong thing.

## Stage 5: the governor defends a floor, not a variance budget

The binding risk here is not drawdown — it is the inception gate. Below your all-time starting
capital the score is multiplied by zero, and 65 of 226 live UIDs sit under that line. So the
governor's primary quantity is **headroom to inception**, and its states key off that:

| Headroom | State | Effect |
|---|---|---|
| ≥ 3% | normal | hard limits only |
| 1–3% | warn | scale-ups and leveraged sleeves blocked |
| 0–1% | critical | safest admissible candidate forced |
| < 0 | breached | recovery mode; score is already zero |

**Hard limits apply in every state, including breached** — a breach is not a licence to abandon
gross or concentration limits, and there is a test asserting that. The governor never mutates a
candidate and never proposes one; a test greps its source to confirm it cannot build a portfolio.

`RecoveryPolicy` replaces the "below the floor, take more risk" reasoning with a simulation that
maximises expected discounted incentive net of pruning risk and delay. My first implementation
stopped at first recovery, which made higher volatility unconditionally better — its own test
caught that and called it a rule rather than a policy. Running the full horizon (every session
above the line earns, every session below earns nothing, and high volatility relapses as easily as
it recovers) produces a genuine interior optimum: across scenarios it now picks 0.012, 0.020 or
0.030 depending on headroom, edge and pruning pressure.

## Stage 6: the alert is the action, not the notification

Several §21 failures look identical on a dashboard and demand opposite responses, so every alert
carries the action it requires and the router takes the strongest one
(`safe_mode` > `freeze` > `alert`):

| Failure | Action | Why |
|---|---|---|
| gross > 1.0 | **safe mode** | upstream discards the file silently; trading on is worse than stopping |
| below inception | **safe mode** | the score gate is shut; recovery policy takes over |
| replica drift > 1e-6 | **freeze** | every projected score is against a function we no longer reproduce |
| constants / ratio / cash-list changed | **freeze** | the game changed between two cycles |
| a *held* ticker left `/assets` | **freeze** | it is now silently cash |
| inactivity ≥ 1 day | alert | refreshing is the fix — stopping would make it worse |
| unconfirmed submission | alert | **do not rewrite**; the miner retries every ~11s |

`explain_score_change()` exists because a drop caused by the cash multiplier and a drop caused by a
drawdown are indistinguishable on the dashboard and call for opposite responses.

`Registry` enforces the two rules that make a lifecycle real: **promotion criteria are hashed at
registration**, so the bar cannot move to wherever the result landed, and **risk-policy artefacts
never auto-promote** — model weights may, limits may not.

`tools/watch.py` is the §23 runbook in one command; it exits 0/1/2/3 for ok/alert/freeze/safe-mode
so cron can act on it.

## Stage 7: what a shadow run can and cannot prove

Shadow is the full loop with the publisher disarmed. It is deliberately short — 10 to 20 sessions
— because the ramp is expensive and the failures it retires are cheap to catch here and permanent
if you catch them live: a silent `asum > 1` discard, an MOC leak, a publisher race, an unvalidated
ticker.

**The gate distinguishes three outcomes, not two.** You are not registered during shadow, so you
have no `/score` row and no market-data feed — which means a forward projection *cannot* be
reconciled against a realised outcome. Those gates report `UNPROVEN`, and `ready` is true only if
every gate `PASS`ed:

```
[    PASS] no_unhandled_faults / nothing_would_be_discarded / cutoffs_met
[    PASS] governor_never_overruled / rule_detector_ran / leakage_probes
[UNPROVEN] projection_accuracy: no /score row and no market-data feed
[UNPROVEN] strategy_edge: shadow tests the PIPELINE, not the edge
```

`strategy_edge` is permanently unproven by construction — a clean operational run says nothing
about whether the signal works. A readiness report that quietly counted `UNPROVEN` as `PASS` would
be the most expensive green light in the project, so it refuses to.

## The bar: what a null strategy is worth

`tools/baseline_bar.py` runs the baselines over SN88's own 314-session feed, fits weights on the
first half, scores the second through the replica on rolling 50-session windows, and maps the
result onto the live board:

```
baseline                       score  would rank
equal_weight                   2.021      50 of 151
equal_risk_contribution        0.422      73 of 151
minimum_variance               0.099      86 of 151   <- last, exactly as 2 predicts
```

**Equal weight over 20 liquid names — no modelling at all — lands around rank 50.** That is
roughly the free-stack break-even, which means the null strategy about pays for itself and nothing
more. The gap to a business is the whole value of the modelling work:

| Target rank | Score needed | × null | **× Sharpe needed** |
|---:|---:|---:|---:|
| 40 | 3.7 | 1.9× | **1.14×** |
| 20 | 15.8 | 7.8× | **1.54×** |
| 10 | 38.7 | 19.1× | **1.85×** |
| 1 | 198.1 | 98.0× | **2.60×** |

The last column is the useful one. Because `score ∝ Sharpe^4.8`, a **7.8× score gap is only a
1.54× Sharpe gap** — the score's convexity cuts both ways, and it makes the ladder far less
daunting in the units you actually optimise.

Note also `zero% = 28%` for equal weight: in more than a quarter of rolling windows a null book
scores *exactly zero*, because the window closed negative and `score <= 0 -> 0`.

### Read that table as a scale, not a target

**2.021 is one cell of a table that moves by 100× across it, and the panel cannot say which
cell is right.** Same signal, same code, only the block of sessions and the book shape varying:

| region | 20 names, equal | 50 names, equal | 100 names, inv-vol |
|---|---:|---:|---:|
| sessions 100–207 | 0.000 | 0.000 | 0.021 |
| sessions 207–314 | 0.032 | 0.319 | 6.413 |
| sessions 157–314 | **2.021** | 5.492 | 2.720 |

And a book of **randomly chosen** names on that last shape scores between 3.1 and 14.0 depending
only on the seed. `score ∝ Sharpe^4.8` means a 1.5× Sharpe difference is a 7× score difference,
and 314 sessions contain roughly six non-overlapping 50-session windows — so regime variation is
amplified past anything a search on this panel can resolve.

So use the ladder above for its **last column** — the Sharpe multiples, which are stable and are
the units you actually optimise — and do not treat any single score as a measured quantity. A
candidate is only interesting if it clears `shape_matched_null()` (below), and on this panel
nothing yet does. See [research/README.md](research/README.md).

## What the tests encode

`tests/test_properties.py` pins the scoring algebra the rest of the system reasons from. If one
fails, a constant changed upstream and candidate selection must freeze (guide §21). Notably:

- `odds` is the **only** exactly scale-invariant term.
- `mar` **rises** with scale (compounding wedge: compounded `gain`, arithmetic `risk`).
- `lsr` **falls** with scale — it uses *dollar* P&L, not `pnl%`.
- Net: **score has decreasing returns to scale** — `score/k` falls from 1.020 to 0.726 across a
  32× range. This corrects revision 3 of the guide, which reported the opposite.

## Layout

```
sn88_replica/
  constants.py    mirror of const.py + fingerprint() for the rule-change detector
  scoring.py      pl2sc + score, byte-faithful
  multipliers.py  gate, dedupe, inactivity, cash, class normalisation, burn
  pipeline.py     full-field scoring -> projected incentive
  modes.py        reconciliation / forward projection / policy backtest
  api.py          endpoint client, fixture archive, §33 integrity checks
  universe.py     /assets: 555 tickers, 16 cash-classified, delisting queue
  strategy.py     upstream parser prediction, serialisation, validation
  publisher.py    atomic locked write + .last-update confirmation
tests/            golden, properties, leakage, publisher
tools/            archive_fixtures.py, rank_field.py, check_strategy.py
fixtures/         daily API snapshots — you cannot backfill these
```

## Next

**Stage 3 — the MOO/MOC replay engine** (guide §5.5, §13.1 mode C). Separate feature snapshots for
09:28 and 15:50 ET, auction execution prices, the NYSE calendar with early closes and DST. The
acceptance test is a leakage probe: feed the session close into the MOC snapshot and the build must
fail. A single daily bar shared between both windows means the MOC model is trading on the close it
is supposed to predict.

Then stage 4 (data plane), stage 5 (survival-aware governor), stage 6 (monitoring and rollback),
stage 7 (10-20 sessions of shadow). **Do not register before stages 2-7 exist** - the asset class is
irreversible and the inception anchor is permanent.

### Operational notes for when you do run this

- `tools/archive_fixtures.py` daily, ideally just after the 06:00 UTC stock scoring.
- `Publisher.touch()` every calendar day **including weekends** - inactivity is charged on a
  calendar clock from the first idle day once your track is older than five sessions.
- Alert if `PublishResult.confirmed` is false after 300s. Do **not** write again: the miner retries
  every ~11 seconds on its own because `.last-update` never advanced.
- Diff `constants.fingerprint()` against upstream on a schedule; validators auto-pull hourly.
