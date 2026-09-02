# research/ — the search log, including what failed

Scratch scripts. Nothing here is imported by `sn88_replica` or the miner. It is kept
because a negative result you cannot reconstruct gets rediscovered as a positive one.

## The finding: nothing is identifiable on 314 sessions

Three experiments, run in this order, each killing the previous one's conclusion.

### 1. `oos.py` — a searched reversal looked like a 4× edge

80 configs (lookback × sector-neutral × top_n × weighting) searched on sessions
100–207, one frozen choice evaluated on the disjoint block 207–314:

```
FROZEN: reversal lookback=10 sector_neutral=True top_n=100 weighting=inverse_vol
  null bar         median 0.032
  frozen reversal  median 8.859        <- looks like a 276x edge
```

Two things were already wrong with this before any control was run. The holdout beat
the search block by 3× (`search 2.987 -> confirm 8.859`), which is backwards from every
honest overfitting story; and the "null bar" was a **20-name equal-weight** book while
the candidate held **100 names inverse-vol**. That comparison attributes book
construction to signal.

### 2. `controls.py` — the edge was the book shape, and not even that

Holding the book shape fixed and destroying only the signal:

```
reversal 10d  (the frozen choice)         8.859
momentum 10d  (sign-flipped)              3.084
shuffled reversal   seeds 1-3       3.613 / 4.833 / 4.494
random 100 names    seeds 1-3      14.040 / 3.078 / 3.631     <- 14.040
liquidity-ranked 100                      6.413
low-volatility 100                        2.567
```

A **random** book scores 14.040 against the searched candidate's 8.859, and seed-to-seed
noise spans 4.6×. The candidate sits inside its own null distribution. The reversal
result is dead, and the momentum arm cannot be used to test the guide's §2.7 prediction
either — everything is inside the noise band.

### 3. `bookshape.py` — the book shape is not stable either

One fixed trivial signal (liquidity rank), only the shape varying, three regions:

```
                    10      20      50     100     200     300     524
A 100-207 equal  0.055   0.000   0.000   0.005   0.027   0.316   1.578
  inv-vol        0.000   0.000   0.000   0.021   0.094   0.306   1.416
B 207-314 equal  0.099   0.032   0.319   1.106   2.989   4.455   3.628
  inv-vol        0.503   0.151   0.786   6.413   5.866   5.910   6.065
C 157-314 equal  0.639   2.021   5.492   1.528   1.666   2.253   1.595
  inv-vol        0.595   0.490   1.302   2.720   0.847   1.794   0.866
```

Region A prefers 524 names equal-weight, B prefers 100 inverse-vol, C prefers 50
equal-weight — and C **overlaps** A and B, so it is not even an independent third
opinion. The 2.021 bar quoted in the README is cell `C / equal / 20`. Its neighbour
`C / equal / 50` is 5.492 and its counterpart `B / equal / 20` is 0.032.

## Why this happens, and why it is not a bug

`score = MAR × LSR × odds% × daily%` over a trailing 50-session window, and
`score ∝ Sharpe^4.8`. A 1.5× difference in Sharpe is a 7× difference in score. With 314
sessions there are ~6 non-overlapping 50-session windows in the whole panel and ~2 in
any half, so ordinary regime variation is amplified into 100× score swings that no
amount of care in the search can average away.

**The binding constraint is sample size, not model quality.** More search on this panel
produces more confident wrong answers. This is what the two-track design in
`sn88_replica/market.py` anticipated: the replication track is authoritative but is one
regime, and the long-regime track exists precisely because 314 sessions cannot support
selection.

## What was fixed as a result

- **A fill-timing look-ahead** (commit `e2ee806`). `evaluate()` transacted at the close
  the signal had just read. Corrected to `fill_lag=1`; `TestFillTiming` pins it. It was
  *understating* fast signals, not inflating them.
- **`shape_matched_null()` / `exceeds_null()`** in `sn88_replica/signals.py`. A candidate
  is now compared against no-signal books of identical shape, and must clear the entire
  null spread. Running the dead reversal through it returns `False`.

## Older scratch

`lab.py`, `mr_lab.py`, `robust.py`, `sweep.py`, `mr/` are partial output from an earlier
agent sweep, kept only as raw material. **They predate the fill-timing fix and their
numbers are wrong.** Do not quote them.

### 4. `identifiability.py` — the resolution floor, measured

Twelve no-signal draws at a fixed book shape (100 names, inverse-vol), against four candidates
evaluated identically:

```
B  sessions 207-314            C  sessions 157-314
  12 no-signal draws:            12 no-signal draws:
    min 1.87  med 3.07  max 14.04   min 0.00  med 1.30  max 5.41
  reversal 10d   8.859  83rd pct    reversal 10d   5.637  100th pct  CLEARS
  reversal 1d    8.009  83rd pct    reversal 1d    3.287   83rd pct
  liquidity      6.413  75th pct    liquidity      3.254   83rd pct
  low-vol        2.567  25th pct    low-vol        0.044    8th pct
```

**The null spread is wider than the spread across candidates.** In block B the twelve no-signal
draws range over 7.5× while the four candidates range over 3.5×; the entire candidate family fits
inside the noise. That is the definition of an underpowered experiment, and it means the *ranking*
of those candidates carries no information either.

`reversal 10d` does clear all twelve draws in block C — by 4%, in one of two blocks, at the 83rd
percentile in the other, from a family that was searched over 80 configs, on a block that overlaps
the one used to pick it. That is not evidence; it is what the tail of a null looks like.

A caveat worth keeping: **none of this shows there is no edge.** It shows this panel cannot
certify one, in either direction. A signal with a strong economic prior is still a legitimate
candidate for shadow operation on exactly that reasoning.
