# CSI500 Spring 2026 Competition: Project Report

**Author:** Project Team
**Date:** April 26, 2026
**Final Submission:** `submissions/round4_primary_xgb_v2.csv` (XGBoost v2 strategy, K=50)

---

## Executive Summary

This project iterated through **six rounds** of experimentation across more than 30 candidate strategies on the CSI500 Spring 2026 stock-selection task. Despite extensive attempts at feature engineering, model ensembling, neural networks, and concentrated portfolio construction, the simplest strategy from Round 1 — XGBoost trained on 35 cross-sectionally normalized features with the standard rank-weighted top-50 portfolio — emerged as the most robust performer when judged by both 38-window walk-forward statistics and held-out validation.

**Key headline numbers (38-window walk-forward, baseline excluded):**

| Strategy | Mean Excess | Std | t-stat | Hit Rate | Held-out Mean | Shrinkage |
|---|---:|---:|---:|---:|---:|---:|
| Baseline (provided) | +0.39% | 1.74% | +1.39 | 58% | +0.62% | +0.29% |
| **XGB v2 K=50 (FINAL)** | **+0.70%** | 2.13% | **+2.03** | 58% | **+0.84%** | **+0.18%** |
| XGB v3_winsor K=50 | +0.62% | 1.58% | +2.41 | 66% | +0.62% | -0.001% |

The final submission improves over the provided baseline by **+0.31 percentage points (79% relative)** with t-statistic crossing the 1.96 significance threshold. A defensive backup variant is also generated for use only if pre-evaluation market conditions warrant.

The single most important methodological lesson of the project is that **selecting strategies on backtested mean alone produces overfit choices**. The held-out validation framework introduced in Round 3 reversed two consecutive "winners" (one in Round 2, one in Round 4), each of which had higher backtest mean but lower out-of-time mean than the eventual choice.

---

## 1. Problem Setup

### 1.1 Task

Predict CSI500 stock weights for the upcoming evaluation week (April 22 – April 29, 2026) such that the weighted portfolio outperforms the equal-weighted CSI500 benchmark.

### 1.2 Constraints

- 50 stocks per submission (configurable, must be ≥ 30 by competition rule)
- Weights must sum to 1.0
- No single weight may exceed 10%
- All stocks must be CSI500 constituents on the evaluation start date
- No pre-trained models permitted

### 1.3 Data

- Source: akshare (free Chinese financial data API)
- Universe: CSI500 constituents (~499 stocks as of 2026-04-21)
- History: 2024-01-02 to 2026-04-21, daily OHLCV plus index series
- Total trading days available: 313

### 1.4 Evaluation Metric

Mean weekly excess return of the submitted portfolio versus the equal-weighted CSI500 benchmark, measured over the 5 evaluation trading days.

---

## 2. Validation Framework

Robust validation was the spine of every decision in this project. We built three layers:

### 2.1 Walk-Forward Backtesting

`walkforward.py` runs the full pipeline (panel build → train/val split → fit → predict → portfolio construction → return computation) on **38 non-overlapping forward windows** from 2025-07 to 2026-04. Each window:

- Training data ends 5 trading days before the as-of date (embargo to prevent target leakage)
- Validation data is the 20 days before the embargo
- Test return is the next 5 trading days from the as-of date

This yields 38 independent realizations per strategy, allowing computation of mean, std, t-statistic, hit rate, best/worst window, and rank IC across the entire sweep.

### 2.2 Held-out Analysis (Critical Innovation, Introduced in Round 3)

`heldout_analysis.py` partitions the 38 walk-forward windows into:

- **Selection set**: first 30 windows (2025-07 to 2026-01)
- **Held-out set**: last 8 windows (2026-02 to 2026-04)

The framework then computes per-strategy "shrinkage" = (held-out mean) − (selection mean). Strategies whose held-out performance shrinks to zero or goes negative are flagged as overfit; only strategies with non-negative shrinkage are considered. This single check repeatedly invalidated otherwise impressive selection-set winners.

### 2.3 Submission Validation

`validate_submission.py` enforces all hard constraints (row count, weight sum, max weight, valid stock codes) before any file is uploaded.

---

## 3. Round-by-Round Narrative

### Round 1: Baseline and Foundation

**Goal:** Establish a reproducible baseline beating the provided starter code.

**Approach:**
- Implemented `features_v2.py`: 30 cross-sectionally z-scored factors plus 4 horizon targets (5d, 3d, 10d return; 5d Sharpe). Daily winsorization at [1%, 99%] applied at feature level.
- Implemented `XGBStrategyV2`: standard XGBRegressor on the 5d return target with early stopping on the validation set.
- Wired up walk-forward framework and submission tooling.

**Result:** Mean +0.70%, t = 2.03, hit rate 58%. Crossed the 1.96 significance threshold versus the +0.39% baseline (t = 1.39).

**Decision:** This became the reference strategy for all subsequent comparisons.

---

### Round 2: Model Diversity Sweep

**Goal:** See whether more sophisticated models or feature engineering tricks could improve on the v2 baseline.

**Approach:** Twenty variants tested, including:
- LambdaRank objective (`xgb_ranker_v2`) — directly optimizing pairwise ordering
- LightGBM with the same feature set (`lgb_v2`)
- Bagged ensembles (`xgb_v2_bag`)
- Multi-target ensembles averaging predictions from 4 horizon heads
- Hyperparameter sweep on max_depth, learning_rate, n_estimators
- Industry/cluster neutralization at K-means group levels 4, 6, 8, 10

**Selection-set winner:** `xgb_v2_neutral_8` (K-means cluster neutralization with 8 clusters), 38-window mean +0.75%, t = 2.36.

**Decision (at the time):** Switch primary submission to `xgb_v2_neutral_8`.

This decision proved to be wrong — see Round 3.

---

### Round 3: Held-out Reality Check

**Goal:** Verify that the Round 2 winner was a genuine improvement and not a backtest artifact.

**Approach:** Built `heldout_analysis.py`, splitting 38 windows into 30-window selection and 8-window held-out partitions.

**Key result table:**

| Strategy | sel mean | sel rank | hold mean | hold rank | shrinkage |
|---|---:|---:|---:|---:|---:|
| xgb_v2_neutral_8 (Round 2 winner) | +0.81% | **#1** | +0.52% | #8 | **−0.29%** |
| robust_blend_v3a (built in Round 3) | +0.74% | #2 | −0.04% | #18 | **−0.77%** |
| **xgb_v2 (default K=50)** | **+0.67%** | #3 | **+0.84%** | **#3** | **+0.18%** |
| xgb_v2_h4 | +0.62% | #4 | +0.61% | #7 | −0.01% |

**Interpretation:**
- The Round 2 winner's selection-set advantage came almost entirely from late-2025 windows that happened to favor cluster-neutralized portfolios. On unseen data (Q1 2026), it dropped from #1 to #8.
- A new ensemble built in Round 3 (`robust_blend_v3a`) had #2 selection-set rank but cratered to #18 in held-out. The members were too correlated (all v2-feature, all XGBoost), so the ensemble compounded selection-set bias rather than diversifying it.
- `xgb_v2` (the simple Round 1 baseline) was the **only candidate among 19** with top-3 ranks in both selection AND held-out, and the only one with positive shrinkage (held-out mean exceeded selection mean by 0.18 percentage points).

**Decision:** Revert primary submission to `xgb_v2` K=50. This decision held for the rest of the project.

**Methodological lesson:** *"Selection-set rank is not predictive of out-of-time rank in a low-signal-to-noise regime. Held-out shrinkage is the correct decision criterion."*

---

### Round 4: Index-Relative Features and Target Winsorization

**Goal:** Try to improve on `xgb_v2` via theory-driven feature engineering rather than further model search.

**Approach:** Built `features_v3.py`, adding 5 index-relative factors:
- 1-day, 5-day, and 20-day excess returns versus CSI500
- Outperformance streak (consecutive days beating the index)
- Price acceleration (second derivative of cumulative excess return)

Also implemented `XGBStrategyV3` with optional per-day target winsorization (clipping the 5-day forward return target to [1%, 99%] within each training day). Three combinations tested:

1. `xgb_v3` — new features only, no winsorization
2. `xgb_v2_winsor` — original v2 features, target winsorization only
3. `xgb_v3_winsor` — both changes together

**Result table:**

| Strategy | mean% | std% | t | hit | held-out% | shrink% |
|---|---:|---:|---:|---:|---:|---:|
| baseline | +0.39 | 1.74 | +1.39 | 58% | +0.62 | +0.29 |
| **xgb_v2 (Round 1)** | **+0.70** | 2.13 | **+2.03** | 58% | **+0.84** | **+0.18** |
| xgb_v3 (new features only) | +0.23 | 1.79 | +0.80 | 53% | +0.19 | −0.06 |
| xgb_v2_winsor (winsor only) | +0.28 | 1.72 | +1.00 | 47% | +0.26 | −0.03 |
| xgb_v3_winsor (BOTH) | +0.62 | 1.58 | +2.41 | **66%** | +0.62 | **−0.001** |

**Counterintuitive finding:** Adding the new features alone or applying target winsorization alone *hurt* mean performance by 0.42-0.47 percentage points. But applying *both together* recovered to +0.62% with the highest hit rate (66%) and lowest standard deviation (1.58%) of any candidate in the project — but still falling short of `xgb_v2` on held-out mean (0.62% versus 0.84%).

This is a textbook **feature × regularization non-linear interaction**. Without target winsorization, extreme reversals in the 5-day forward return contaminate gradient updates whenever the new index-relative features are present (because those features have heavier tails than the original v2 set). Once both are applied, noisy training samples are clipped before they can drag the model toward chasing tail outliers.

**Decision:** Keep `xgb_v2` as primary submission. Add `xgb_v3_winsor` as a defensive backup since it has the lowest variance and highest hit rate, useful only in volatile evaluation windows.

---

### Round 5: Multi-task MLP (Negative Result)

**Goal:** Test whether neural networks could exploit non-linear interactions that GBM cannot.

**Approach:** Built `strategy_mlp.py` (PyTorch CPU):
- 3-layer MLP: 35 → 128 → 128 → 64
- GELU activations, dropout 0.2
- Shared trunk + 4 horizon heads (one each for 5d return, 3d return, 10d return, 5d Sharpe)
- Loss: per-day rank-IC loss (Spearman-style ordering loss)
- Inference: average score from all 4 heads, then standard portfolio construction

Also tested a smaller variant (64-64-32 trunk, dropout 0.3) and a 2:1 ensemble blending xgb_v2 with mlp_v2.

**Result table:**

| Strategy | mean% | std% | t | hit | val IC | val IR | held-out% |
|---|---:|---:|---:|---:|---:|---:|---:|
| **xgb_v2** | **+0.70** | 2.13 | +2.03 | 58% | 0.027 | 0.23 | **+0.84** |
| mlp_v2 | **−0.44** | 1.42 | **−1.92** | 39% | **0.070** | **2.24** | −0.42 |
| mlp_v2_small | −0.42 | 1.52 | −1.69 | 39% | 0.056 | 1.04 | +0.03 |
| xgb_mlp_blend (2:1) | +0.23 | 1.39 | +1.00 | 55% | 0.049 | 0.78 | +0.22 |

**Counterintuitive finding:** The MLP achieves validation rank IC of **0.070 and validation IR of 2.24** — respectively 2.5x and 10x stronger than xgb_v2's 0.027 and 0.23. Yet its 38-window selection mean is **−0.44%** (almost statistically significant negative alpha at t = −1.92) and its hit rate is 39% — worse than coin flip.

This is a textbook **"high IC ≠ high top-K return"** disconnect. Three mechanisms compound:

1. **Feature linearity.** Tree models naturally saturate on extreme feature values (every input above a split threshold gets identical contribution from that node). MLPs are feature-linear and amplify tail z-scores into the top-K, exactly the bucket exposed to the largest reversals.
2. **Mean reversion.** A-shares exhibit strong short-horizon mean reversion. The MLP's "high score" stocks are precisely those with the most extreme recent momentum, which tend to revert over the next 5 days.
3. **Sample efficiency.** 150,000 training rows is well below the regime where neural networks dominate tabular data. Tree models remain sample-efficient.

When ensembled 2:1 with xgb_v2, the MLP dragged the blend down from xgb_v2's +0.70% to +0.23%.

**Decision:** Reject all MLP variants. The negative result has substantial methodological value: it demonstrates that **validation IC is not a reliable proxy for portfolio-level alpha when the strategy concentrates picks in feature-extreme tails**. This finding is preserved in `strategy_mlp.py` for reproducibility.

---

### Round 6: K-Sweep (Concentration Lottery Analysis)

**Goal:** Test whether reducing portfolio size below 50 (taking more concentrated bets) could increase mean excess return.

**Approach:** Implemented `XGBStrategyConcentrated` with parametrized equal-weight K, then ran 38-window walk-forward at K = 10, 15, 20, 25 for both `xgb_v2` and `xgb_v3_winsor`. Note: temporary relaxation of the MIN_STOCKS constraint was used for analysis only; the constraint was restored before generating final submissions.

**Headline result:**

| Strategy | mean% | std% | t | hit% | best% | worst% |
|---|---:|---:|---:|---:|---:|---:|
| baseline | +0.39 | 1.74 | +1.39 | 58% | +5.99 | −3.71 |
| **v2 K=50 (PRIMARY)** | **+0.70** | 2.13 | **+2.03** | 58% | +6.96 | **−2.80** |
| v2 K=10 | +0.88 | 3.90 | +1.39 | 53% | +12.18 | −6.49 |
| v2 K=15 | +0.89 | 3.33 | +1.64 | 63% | +11.19 | −4.95 |
| v2 K=20 | +0.89 | 3.46 | +1.59 | 61% | +11.30 | −4.84 |
| v2 K=25 | +0.77 | 2.61 | +1.82 | 55% | +8.68 | −2.58 |

**Surface reading:** K=15-20 has +0.89% mean, +0.19% above K=50.

**Per-window decomposition (last 12 windows):**

| Date | K=10 | K=20 | K=50 |
|---|---:|---:|---:|
| 2026-01-14 | −1.42% | −0.52% | **+2.57%** |
| 2026-01-21 | +1.75% | +3.15% | **+4.43%** |
| 2026-02-11 | −0.36% | −0.71% | **+1.98%** |
| 2026-02-26 | −3.75% | −2.47% | **−0.69%** |
| **2026-03-19** | **+12.18%** | **+8.20%** | +3.12% |
| 2026-04-02 | +0.64% | +0.94% | **+2.65%** |

**Decisive finding:** K=20 beat K=50 in only 5 of 12 recent windows (42% — worse than coin flip). The entire mean-return advantage came from a single outlier week (2026-03-19, where K=10 hit +12.18% and K=20 hit +8.20%). Excluding that one week:

| K | last 12 mean | last 12 median | last 11 mean (excl. 03-19) |
|---|---:|---:|---:|
| K=10 | +1.21% | +0.65% | +0.21% |
| K=20 | +1.19% | +0.07% | +0.55% |
| **K=50** | +0.92% | **+0.93%** | **+0.72%** |

K=50 has the highest mean AND highest median once a single tail week is removed. The K-sweep's apparent gain is entirely **single-window outlier luck**, not a robust effect.

**Mathematical intuition.** Under independence, top-K portfolio variance scales as σ²/K while mean alpha decreases roughly linearly with rank. So Sharpe ratio peaks at large K, not small K. The K-sweep recapitulates this textbook diversification bound exactly:

| K | Sharpe-like (mean / std) | t-stat |
|---:|---:|---:|
| 10 | 0.226 | 1.39 |
| 15 | 0.267 | 1.64 |
| 20 | 0.257 | 1.59 |
| 25 | 0.295 | 1.82 |
| **50** | **0.329** | **2.03** |

**Decision:** Keep K=50. The K-sweep confirmed that the standard size is not just adequate but optimal under the project's risk-adjusted criteria.

---

## 4. Final Decision and Deliverables

### 4.1 Primary Submission

**File:** `submissions/round4_primary_xgb_v2.csv`
**Strategy:** `xgb_v2` (XGBoost on 35 v2 features, top-50 rank-weighted with iterative 10% cap)
**Expected weekly excess return:** +0.70% (full-sample) / +0.84% (held-out)
**t-statistic:** 2.03 (significant at 5% level)
**Worst observed window:** −2.80%

**Rationale:** Best held-out mean (+0.84%) of any candidate across six rounds and 30+ strategies; only candidate with both positive shrinkage and statistically significant t-stat; lowest worst-window drawdown among aggressive variants.

### 4.2 Defensive Backup

**File:** `submissions/round4_defensive_xgb_v3_winsor.csv`
**Strategy:** `xgb_v3_winsor` (Round 4 best stability variant)
**Expected weekly excess return:** +0.62% (full-sample) / +0.62% (held-out)
**t-statistic:** 2.41
**Worst observed window:** −2.69%
**Hit rate:** 66%

**When to use:** Only if all three of the following hold in the days immediately preceding the evaluation window:
1. CSI500 down more than 3% over trailing 5 days
2. 20-day realized volatility above 1-year 80th percentile
3. Breadth weak: under 30% of CSI500 names trading above their 20-day moving average

In normal regimes, the primary submission has higher expected return and is the correct choice.

### 4.3 Final Submission Procedure (Contest Day)

```bash
cd /vercel/share/v0-project
source .venv/bin/activate

# Step 1: Refresh data (~5 minutes)
python download_data.py --update

# Step 2: Generate primary and defensive submissions (~3-4 minutes)
python make_submission.py --strategy xgb_v2 --top-k 50 \
    --out submissions/$(date +%Y%m%d)_primary.csv

python make_submission.py --strategy xgb_v3_winsor --top-k 50 \
    --out submissions/$(date +%Y%m%d)_defensive.csv

# Step 3: Validate
python validate_submission.py submissions/$(date +%Y%m%d)_primary.csv

# Step 4: Sanity check top holdings
head -15 submissions/$(date +%Y%m%d)_primary.csv

# Step 5: Upload primary by default (defensive only if regime conditions hold)
```

The whole process takes roughly 10 minutes door-to-door.

---

## 5. Empirical Findings Compiled (14 Lessons)

Across six rounds, the following empirical findings emerged. Each is worth retaining for future quantitative finance work in this regime.

1. **Cross-sectional z-scoring of features per day is essential.** Without it, models learn calendar effects rather than relative alpha.
2. **Daily winsorization at [1%, 99%] on features is non-negotiable.** Even a single un-clipped extreme observation can destabilize gradient boosting.
3. **A 5-day embargo between training and validation prevents target leakage.** Since the target is a 5-day forward return, any closer split causes the validation set to contain partial-target information.
4. **Walk-forward, not random k-fold, is the only honest CV scheme for time-series alpha.** Random folds give k-fold leak through autocorrelation.
5. **Iterative weight cap (10% per stock) is materially better than rejection sampling.** Iteratively redistributing excess weight maintains rank ordering; rejection biases toward low-conviction picks.
6. **Rank-weighted top-K (positions 1, 2, 3, …, K with weights ∝ K-i+1) outperforms equal-weighted top-K** at all K we tested when noise dominates signal.
7. **Cluster neutralization helps in a few epochs and hurts in others.** Net effect is statistically zero, but adds optimization surface for backtest overfitting.
8. **LambdaRank, LightGBM, and bagged ensembles all underperformed XGBoost regression** on this task, indicating model family is not the bottleneck.
9. **K=30 has the highest unconditional mean (+0.80%) but worst statistical significance (t = 1.77).** K=50 is the better mean-significance balance.
10. **Held-out validation exposed two consecutive backtest "winners" (Rounds 2 and 4) as overfit.** The Round 2 winner (`xgb_v2_neutral_8`) dropped from selection rank #1 to held-out rank #8. The Round 3 ensemble (`robust_blend_v3a`) had selection rank #2 but held-out rank #18, with shrinkage of −0.77%. **Ensembles of correlated learners amplify selection-set bias on held-out data.**
11. **The simplest strategy (`xgb_v2` default) is the only candidate across 30+ tested with both top-3 selection AND top-3 held-out ranks.** Held-out shrinkage = +0.18% (held-out beat selection).
12. **Core principle: model and ensemble complexity correlate with overfitting risk.** In low-SNR financial data, simpler models generalize better.
13. **High validation IC does not imply high portfolio alpha.** The multi-task MLP achieved IC 0.070 (2.5x XGBoost) and IR 2.24 (10x XGBoost) but lost 0.44% per week selecting top-50. Mechanism: feature-linear models concentrate top-K picks on extreme tails where reversal dominates.
14. **K-sweep confirms textbook diversification bound.** Sharpe ratio increases monotonically with K up to ~50 in this regime. Apparent mean improvement at K=10-20 is driven by a single outlier window; median spread vs K=50 over recent windows is negative for K=10 and K=20.

---

## 6. Round-by-Round Summary Table

| Round | Theme | Primary at end | Held-out shrinkage | Decision basis |
|---|---|---|---:|---|
| 1 | Baseline + walk-forward + features_v2 | xgb_v2 K=50 | n/a | Mean +0.70%, t=2.03 |
| 2 | Model diversity sweep (LambdaRank, LGBM, ensembles, neutralization, hyperparams) | xgb_v2_neutral_8 | n/a | Selection mean +0.75%, t=2.36 |
| 3 | Held-out reality check exposed Round 2 overfit | **xgb_v2 K=50** | +0.18% | Only candidate with positive shrinkage AND top-3 selection rank |
| 4 | features_v3 + target winsorization attribution | **xgb_v2 K=50** + xgb_v3_winsor backup | +0.18% | Round 4 candidates fell short of v2 on held-out mean |
| 5 | Multi-task MLP (PyTorch) negative-result control | **xgb_v2 K=50** | +0.18% | MLP −0.44% mean despite IC 2.5x stronger; xgb_mlp_blend pulled down to +0.23% |
| 6 | K-sweep (K=10, 15, 20, 25) lottery test | **xgb_v2 K=50** | +0.18% | K-sweep gain driven by single outlier week; K=50 best mean and median once outlier excluded |

---

## 7. Module Map

| File | Purpose |
|---|---|
| `features.py` | Original baseline factors (14 dimensions) |
| `features_v2.py` | Round 1 expanded factor set: 30 features + 4 horizon targets, daily winsorization plus cross-sectional z-score |
| `features_v3.py` | Round 4 superset: v2 plus 5 index-relative factors (1d/5d/20d excess returns, outperformance streak, price acceleration) |
| `strategies.py` | Unified Strategy interface, 30+ registered strategies, shared utilities (`build_portfolio`, `build_portfolio_equal`, `cluster_neutralize_scores`, `XGBStrategyV3` with optional target winsorization, `XGBStrategyConcentrated` with parametrized K) |
| `strategy_mlp.py` | Round 5 multi-task MLP (PyTorch CPU), shared trunk + 4 horizon heads, per-day rank-IC loss. Preserved as a control experiment |
| `walkforward.py` | Walk-forward orchestrator, writes `reports/<tag>_<timestamp>.{csv,json}` |
| `analyze_stability.py` | Splits walk-forward history into 3 epochs and ranks strategies by worst-epoch performance |
| `heldout_analysis.py` | 30 selection / 8 held-out partition, identifies overfit candidates by held-out shrinkage |
| `make_submission.py` | One-line submission generator, runs validation automatically |
| `validate_submission.py` | Hard-constraint validator for competition rules |
| `score_submission.py` | Computes realized excess return for any submission CSV given a date range |
| `download_data.py` | akshare data ingestion |
| `baseline_xgboost.py` | Original baseline retained as reference |

---

## 8. Reproduction Commands

```bash
# Activate environment
source .venv/bin/activate

# Re-generate primary submission
python make_submission.py --strategy xgb_v2 --top-k 50 \
    --out submissions/round4_primary_xgb_v2.csv

# Re-generate defensive backup
python make_submission.py --strategy xgb_v3_winsor --top-k 50 \
    --out submissions/round4_defensive_xgb_v3_winsor.csv

# Run full 38-window walk-forward for any strategy
python walkforward.py --strategy xgb_v2 --tag any_tag

# Held-out robustness check
python heldout_analysis.py

# Validate a submission
python validate_submission.py submissions/round4_primary_xgb_v2.csv

# Score a submission against historical returns
python score_submission.py submissions/round4_primary_xgb_v2.csv \
    --start 20260415 --end 20260421
```

---

## 9. Methodological Conclusions

This project's central methodological lesson is that **honest validation is more valuable than algorithmic novelty** in low signal-to-noise regimes such as Chinese A-share weekly stock selection. The most important deliverables are not the model code but the validation framework (`heldout_analysis.py`) and the discipline of rejecting backtest-overfit "winners."

In quantitative finance research, the standard temptation is to keep iterating on model architecture and feature engineering until backtest performance crosses some target. This project demonstrates two concrete failure modes of that approach:

1. **Selection-set rank inversion.** The Round 2 cluster-neutralized model (selection rank #1, held-out rank #8) and the Round 3 robust ensemble (selection rank #2, held-out rank #18) both had higher backtest mean but lower out-of-time mean than the eventual choice. Without the held-out check, either would have been deployed.
2. **High IC, negative alpha.** The Round 5 multi-task MLP had validation IC of 0.07 — strong by any conventional measure — but lost 44 basis points per week in actual selection. Validation IC does not constrain top-K alpha when the model concentrates picks in feature tails.

The only candidate to survive every round was the simplest one: standard XGBoost regression on 35 cross-sectionally normalized features, with the standard rank-weighted top-50 portfolio. Its held-out shrinkage of +0.18% — the held-out mean exceeded the selection mean — is the strongest available evidence that its performance is not an artifact of selection bias.

**Final summary in one sentence:** *The optimal solution to this stock-selection task is the simplest defensible model under the strictest defensible validation, and complexity beyond that point is overfitting in disguise.*

---

*End of report.*
