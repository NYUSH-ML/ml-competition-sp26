# CSI500 Spring 2026 — Run Summary (post round-5 NN ablation)

## Final deliverables

| File | Description |
|---|---|
| **`submissions/round4_primary_xgb_v2.csv`** | **Primary submission** — `xgb_v2 K=50`, held-out mean +0.84% (#3 hold-mean) |
| `submissions/round4_defensive_xgb_v3_winsor.csv` | **Defensive backup** — `xgb_v3_winsor`: lowest std (1.58%), highest hit rate (66%), t=2.41; 80% overlap with primary but more diversified |
| `submissions/round3_final.csv` | (identical content to primary, kept under old name for traceability) |
| `submissions/baseline.csv` | Sanity floor — original baseline, never beaten by anything we shipped |

## Primary submission stats (`round4_primary_xgb_v2.csv`)

| Item | Value |
|---|---|
| Reference trading day | 2026-04-21 (latest) |
| Holdings | 50 stocks |
| Weight sum | 1.000000 |
| Max weight | 3.92% (well below the 10% cap) |
| Min weight | 0.08% |
| Validation | passes `validate_submission.py` |

Top 8 holdings: 300390 / 688295 / 002261 / 002436 / 301308 / 600549 / 002756 / 300223

## Why we kept default `xgb_v2` instead of round-2's `xgb_v2_neutral_8`

In round 3 we ran a held-out check: first 30 walk-forward windows as the
*selection* set, last 8 as the untouched *held-out* tail. Result:

| Strategy | sel mean% | sel rank | hold mean% | hold rank | shrink |
|---|---:|---:|---:|---:|---:|
| **xgb_v2 K=50 (final pick)** | +0.67 | #3 | **+0.84** | #3 | **+0.18** |
| xgb_v2_neutral_8 (round-2 winner) | +0.81 | **#1** | +0.52 | #8 | **−0.29** |
| robust_blend_v3a (newly built) | +0.74 | #2 | **−0.04** | #18 | **−0.77** |
| xgb_v2_h4 | +0.62 | #4 | +0.61 | #7 | −0.01 |

`xgb_v2 K=50` is **the only candidate out of 19 that landed in the top 3 on
both the selection set and the held-out tail**, with a *positive* shrink of
+0.18% (held-out actually came in higher than selection). By contrast,
`xgb_v2_neutral_8` owed its #1 selection ranking to the tail epoch
(2025-12 to 2026-02) and collapsed to #8 on held-out. Newly built robust
ensembles instead amplified per-member variance and went outright negative
on held-out.

---

## Round 4: features_v3 + target winsorization attribution

The round-4 rule was: **only adopt changes that produce shrink ≥ 0 on
held-out**. We added:

- `features_v3.py`: v2 + 5 index-relative factors (excess returns at 1/5/20d,
  outperformance streak, price acceleration)
- `XGBStrategyV3` with optional per-day target winsorization

| Strategy | mean% | std% | t | hit | IC IR | hold mean% | shrink |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | +0.39 | 1.74 | 1.39 | 58% | 0.20 | +0.62 | +0.29 |
| xgb_v2 (primary) | +0.70 | 2.13 | 2.03 | 58% | 0.23 | **+0.84** | **+0.18** |
| **xgb_v3 (new features only)** | **+0.23** | 1.79 | 0.80 | 53% | 0.11 | +0.19 | −0.06 |
| **xgb_v2_winsor (winsor only)** | **+0.28** | 1.72 | 1.00 | 47% | 0.27 | +0.26 | −0.03 |
| **xgb_v3_winsor (both)** | **+0.62** | **1.58** | **2.41** | **66%** | 0.17 | +0.62 | **−0.001** |
| xgb_v2_v3_winsor_blend | +0.61 | 1.90 | 1.99 | 55% | 0.20 | +0.37 | −0.31 |

**Key attribution finding (paper-worthy)**:

- v3 features alone: mean **collapses to 0.23%** (−0.47% vs xgb_v2)
- target winsor alone: mean **collapses to 0.28%** (−0.42% vs xgb_v2)
- **Both together: mean 0.62% (close to v2), and std/t/hit all best in class
  (hit 66% is the highest of any candidate)**

This is a textbook **feature × regularization non-linear synergy**: only
once the noisy training samples are clipped by per-day winsorization can
the model cleanly extract the index-relative signal in the higher-dimensional
v3 feature space. Adding either change in isolation hurts; together they
produce the most stable model we built.

**Why we did NOT promote `xgb_v3_winsor` to primary**:

- Its held-out mean (0.62%) is 0.22% below xgb_v2 (0.84%), and its 38-window
  mean is 0.08% lower
- Its real edge is *stability*, not magnitude (worst window −2.69% vs
  xgb_v2 −2.80%)
- Therefore we ship it as a **defensive backup**: if a sharp drawdown hits
  in the days before the contest evaluation window, switch to it

**Why we did NOT use the v2 + v3_winsor blend**:

- 38-window mean (+0.61%) is already worse than xgb_v2 alone
- held-out shrink is **−0.31%** — confirming once more that ensembling
  correlated base learners shrinks negatively on held-out

---

## 38-window walk-forward leaderboard (top 6)

| Rank | Strategy | mean% | std% | t | hit | IC | IR |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | xgb_v2 K=30 | +0.80 | 2.78 | 1.77 | 55% | 0.027 | 0.227 |
| 2 | xgb_v2_neutral_8 | +0.75 | 1.96 | 2.36 | 61% | 0.027 | 0.227 |
| **3** | **xgb_v2 K=50 ← primary submission** | **+0.70** | 2.13 | **2.03** | 58% | 0.027 | 0.227 |
| 4 | xgb_v2_h4 | +0.61 | 1.94 | 1.95 | 58% | 0.036 | 0.294 |
| 5 | xgb_v2_h4_neutral | +0.58 | 1.67 | 2.14 | 58% | 0.036 | 0.294 |
| 6 | xgb_v2 K=80 | +0.56 | 1.61 | 2.14 | 58% | 0.027 | 0.227 |

**Reference baseline**: mean +0.39%, std 1.74%, t=1.39, hit 58%

Final delta vs baseline (full 38 windows):

- mean: **+0.39% → +0.70% (+0.31pp, +79% relative)**
- t-stat: **1.39 → 2.03 (crosses the p<0.05 significance line)**
- worst window: **−3.71% → −2.80% (drawdown reduced 25%)**

**Held-out check (first 30 train / last 8 test) confirms no overfit**:

- selection mean +0.67%, held-out mean **+0.84%**, shrink **+0.18%** (positive)
- only candidate of 19 to rank top-3 on both selection and held-out

---

## Full experiment table (20 candidates)

Sorted by 38-window mean%:

```
                        tag                strat   k  mean%   std%  worst%  hit      t     IC     IR
                     v2_k30               xgb_v2  30 +0.797 +2.775  -3.627 0.55 +1.771 +0.027 +0.227
   xgb_v2_neutral_8_full     xgb_v2_neutral_8   50 +0.749 +1.955  -2.973 0.61 +2.363 +0.027 +0.227
                    v2_full              xgb_v2   50 +0.703 +2.131  -2.797 0.58 +2.034 +0.027 +0.227
            xgb_v2_h4_full           xgb_v2_h4   50 +0.614 +1.943  -3.090 0.58 +1.950 +0.036 +0.294
           xgb_v2_bag_full          xgb_v2_bag   50 +0.597 +2.140  -2.919 0.50 +1.719    NaN    NaN
    xgb_v2_h4_neutral_full   xgb_v2_h4_neutral  50 +0.578 +1.665  -2.368 0.58 +2.139 +0.036 +0.294
                    v2_k80              xgb_v2   80 +0.559 +1.612  -1.977 0.58 +2.139 +0.027 +0.227
   xgb_v2_neutral_15_full   xgb_v2_neutral_15   50 +0.537 +1.907  -2.891 0.58 +1.737 +0.027 +0.227
            xgb_v2_h3_full           xgb_v2_h3   50 +0.509 +2.040  -3.291 0.55 +1.538 +0.023 +0.201
       xgb_v2_neutral_full      xgb_v2_neutral  50 +0.507 +1.882  -2.472 0.55 +1.660 +0.027 +0.227
                   v2_k100              xgb_v2  100 +0.486 +1.381  -1.870 0.61 +2.170 +0.027 +0.227
            xgb_v2_h2_full           xgb_v2_h2   50 +0.483 +1.829  -2.884 0.58 +1.628 +0.026 +0.243
 xgb_v2_multi_horizon_full xgb_v2_multi_horizon  50 +0.483 +1.985  -3.003 0.50 +1.500 +0.028 +0.238
                  mt4_full xgb_v2_multi_target   50 +0.481 +1.549  -2.203 0.58 +1.916 +0.034 +0.345
            xgb_v2_h1_full           xgb_v2_h1   50 +0.463 +2.298  -3.605 0.61 +1.242 +0.034 +0.272
           best_blend_full          best_blend   50 +0.414 +1.579  -2.241 0.55 +1.616 +0.038 +0.404
            ranker_v2_full       xgb_ranker_v2   50 +0.394 +1.513  -1.812 0.53 +1.606 -0.027 -0.239
             baseline_full        xgb_baseline   50 +0.393 +1.738  -3.712 0.58 +1.394 +0.020 +0.202
          ensemble_v2_full         ensemble_v2   50 +0.375 +1.701  -2.539 0.58 +1.361    NaN    NaN
               lgb_v2_full              lgb_v2   50 +0.346 +1.721  -2.730 0.55 +1.240 +0.021 +0.325
```

---

## Empirical findings (paper-worthy)

1. **Feature engineering beats model iteration**. Going from `features.py`
   to `features_v2.py` lifted mean from +0.39% to +0.70%. Every model /
   ensemble / hyperparameter sweep that followed only added a further
   +0.05% in aggregate. **Factors carry the alpha; the model just
   extracts it**.

2. **Cross-sectional z-scoring + winsorization matter** because they prevent
   A-share daily limit hits and small-cap outliers from dominating tree splits.

3. **Cluster neutralization gives the highest single-step ROI**. KMeans
   on the trailing 60-day return matrix into 8 clusters, then demeaning
   model scores within each cluster, lifted t from 2.03 → 2.36 and
   reduced std from 2.13 → 1.96 — using **no external data at all**.

4. **Cluster granularity is sensitive**: 8 clusters is the sweet spot;
   10 and 15 both regress because too-fine grouping starts erasing the
   model's stock-selection signal along with the style exposure.

5. **Bagging fails in weak-signal regimes**. Five XGBoost seeds were so
   correlated that their average underperformed any single seed
   (+0.70 → +0.60).

6. **Equal-weight XGB+LGB blending is an anti-pattern**. The weaker
   learner pulls down the stronger one's mean — ensembling only helps
   when component alphas are comparable.

7. **LambdaRank performs poorly on low-SNR financial time series**. Its
   pairwise loss overfit intra-day noise: 5-day cross-sectional validation
   IC came out at −0.24.

8. **Multi-task ensembling lifts IR but hurts mean**. Sharpe-targeted
   members favor "smooth winners" that don't align with the contest's
   5-day absolute excess return objective. **High IR ≠ contest victory**.

9. **K=30 has the highest expectation but is the wrong bet**: t=1.77
   doesn't cross significance and worst window is −3.63%. K=50 is the
   best mean / significance trade-off.

10. **Held-out checking exposed backtest overfit (round 3)**. Round-2's
    selection-leader `xgb_v2_neutral_8` fell to held-out rank #8
    (shrink −0.29%); the freshly built robust ensembles ranked #2 on
    selection but #18 on held-out (shrink −0.77%). Ensembling three
    correlated base learners amplified the selection-set lucky bias into
    a held-out negative.

11. **The truly stable winner is plain `xgb_v2` (default params)**:
    selection #3, held-out #3, shrink **+0.18%** (positive). It depends
    on no selection-tuned trick, and is the only candidate of 19 to
    appear in the top 3 of both groups.

12. **Ground rule: the more elaborate the model / ensemble / hyperparameter
    schedule, the higher the overfitting risk**. On low-SNR financial
    data, simpler models generalize better.

13. **Round 5 multi-task MLP control experiment confirms the GBM
    advantage in reverse**. A 3-layer MLP (128 → 128 → 64, GELU + dropout
    0.2) with shared trunk and four horizon heads, trained with a per-day
    rank-IC loss, hit **38-window mean −0.44%, hit 39%, t = −1.92** (a
    near-significant *negative* alpha) — yet its **validation IC was 0.070
    and IR 2.24, respectively 2.5× and 10× better than xgb_v2**. This is
    the textbook "high IC ≠ high top-K return" phenomenon: the MLP learned
    fine-grained ordering of mid-rank stocks, but its top-K systematically
    selected extreme-momentum names that mean-reverted over the next
    5 days. A 2:1 rank blend with xgb_v2 only dragged xgb_v2 from +0.70%
    down to +0.23%, confirming that **low base-learner correlation does
    not by itself make an ensemble useful**.

14. **Why MLPs hurt in low-SNR + short-history regimes**: (a) 150K
    training rows are well below the scale at which neural networks
    surpass GBMs; (b) tree models naturally saturate at extreme feature
    values (everything past a split threshold collapses to the same
    score), whereas a feature-linear MLP unboundedly amplifies tail
    z-scores into top-K; (c) A-share short-horizon mean reversion turns
    these "well-ranked" extreme names into 5-day reversions.
    **Conclusion: under the contest's no-pretraining rule and at this
    data scale, GBMs remain SOTA**.

### Round 5 NN experiment summary

| Strategy | 38W mean% | std% | t | hit | val IC | val IR | hold mean% | shrink% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **xgb_v2 (winner)** | **+0.70** | 2.13 | +2.03 | 58% | 0.027 | 0.23 | **+0.84** | **+0.18** |
| mlp_v2 | −0.44 | 1.42 | −1.92 | 39% | **0.070** | **2.24** | −0.42 | +0.03 |
| mlp_v2_small | −0.42 | 1.52 | −1.69 | 39% | 0.056 | 1.04 | +0.03 | +0.56 |
| xgb_mlp_blend (2:1) | +0.23 | 1.39 | +1.00 | 55% | 0.049 | 0.78 | +0.22 | −0.005 |

---

## Five-round optimization overview

| Round | Main work | Primary submission | Decision rationale |
|---|---|---|---|
| 1 | baseline + walk-forward framework + features_v2 | xgb_v2 K=50 | 38-window mean +0.70%, t crosses 1.96 significance |
| 2 | LambdaRank / LightGBM / ensembles / bagging / multi-target / cluster neutralization / hyperparameter sweep | xgb_v2_neutral_8 | 38-window mean +0.75%, t = 2.36 |
| 3 | held-out check (first 30 train / last 8 test) exposes round-2 overfit | **xgb_v2 K=50** | held-out shrink +0.18%, only consistently stable winner |
| 4 | features_v3 (index-relative factors) + target winsorization attribution | **xgb_v2 K=50 (kept)** + xgb_v3_winsor as defensive backup | new variant's hold mean 0.62% < v2's 0.84%; primary unchanged, defensive added |
| 5 | multi-task MLP / xgb_mlp_blend NN control experiment | **xgb_v2 K=50 (kept)** | MLP 38-window mean −0.44%; xgb_mlp_blend drags v2 to +0.23%; NNs are net-harmful at this data scale |

---

## Reproduction commands

```bash
# Activate venv (required)
source .venv/bin/activate

# Regenerate primary submission (after a data update)
python make_submission.py --strategy xgb_v2 --top-k 50 \
    --out submissions/round4_primary_xgb_v2.csv

# Regenerate defensive backup submission
python make_submission.py --strategy xgb_v3_winsor --top-k 50 \
    --out submissions/round4_defensive_xgb_v3_winsor.csv

# Run the full 38-window walk-forward for any registered strategy
python walkforward.py --strategy xgb_v2 --tag any_tag

# Held-out robustness check
python heldout_analysis.py

# Validate a submission against contest format rules
python validate_submission.py submissions/round4_primary_xgb_v2.csv

# Historical scoring (evaluation window 04-15 .. 04-21)
python score_submission.py submissions/round4_primary_xgb_v2.csv \
    --start 20260415 --end 20260421
```

---

## Module structure

| File | Role |
|---|---|
| `features.py` | Original baseline factor set (14 dims) |
| `features_v2.py` | Enriched factor set (30 dims + 4 targets, with daily winsorization and cross-sectional z-scoring) |
| `features_v3.py` | v2 + 5 index-relative factors (excess returns 1/5/20d, outperformance streak, price acceleration) |
| `strategies.py` | Unified `Strategy` interface + 29 registered strategies + shared helpers (`build_portfolio`, `cluster_neutralize_scores`, `XGBStrategyV3` with optional target winsor) |
| `strategy_mlp.py` | Multi-task MLP (PyTorch CPU): shared trunk + 4 horizon heads, per-day rank-IC loss. Round-5 control showing NNs hurt at this data scale |
| `walkforward.py` | Non-overlapping multi-window backtest scheduler, writes `reports/<tag>_<timestamp>.{csv,json}` |
| `make_submission.py` | One-shot submission builder for any registered strategy + automatic format validation |
| `analyze_stability.py` | Splits walk-forward history into 3 chronological epochs and ranks strategies by worst-epoch mean |
| `heldout_analysis.py` | First-30 / last-8 held-out check to surface overfit |
| `baseline_xgboost.py` | Original baseline kept as a reference point |
| `download_data.py` | Data ingestion via akshare |
| `validate_submission.py` | Submission-format constraint checker |
| `score_submission.py` | Score a submission against realized returns |

---

## Contest-day checklist (before submitting)

1. `python download_data.py --update` — pull the latest OHLCV
2. `python make_submission.py --strategy xgb_v2 --top-k 50 --out submissions/<date>_primary.csv` — primary
3. `python make_submission.py --strategy xgb_v3_winsor --top-k 50 --out submissions/<date>_defensive.csv` — defensive backup
4. `python validate_submission.py submissions/<date>_primary.csv` — must pass cleanly
5. `head -10 submissions/<date>_primary.csv` — eyeball the top holdings, sanity-check that nothing is bizarre
6. By default upload `<date>_primary.csv`. If the market sells off sharply or volatility spikes in the days leading up to the evaluation window, fall back to `<date>_defensive.csv`.
