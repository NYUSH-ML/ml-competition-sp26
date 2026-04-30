# CSI500 Stock Selection Competition (Spring 2026)

Build a machine-learning model that selects a portfolio of CSI500 stocks
expected to outperform the index over the following week.

## Competition Rules

### 1. Overview

This is an **individual** project. Each student builds a model that, given
historical market data, produces a long-only portfolio over the CSI500
universe. The student submits a portfolio twice; the course staff holds each
portfolio over a fixed evaluation window and scores it against the CSI500
index. Final grade combines the submitted report, a self-test section, and
two live out-of-sample evaluations.

### 2. Task

On each deadline, submit a portfolio of weights `{w_i}` over CSI500 stocks.
Weights apply for the *following* evaluation window (defined in §4). The
portfolio is long-only and fully invested.

### 3. Portfolio Constraints

All four constraints are checked automatically. Any violation → rejection.

| # | Constraint | Value |
| --- | --- | --- |
| 3.1 | Universe | Stock must be a current CSI500 constituent (as listed in `data/constituents.csv`) |
| 3.2 | Minimum number of names with weight > 0 | **30** |
| 3.3 | Maximum weight per name | **0.10** (10%) |
| 3.4 | Weights non-negative and sum to 1.0 | tolerance ±1e-4 |

### 4. Timeline

All deadlines are **China Standard Time (UTC+8)**.

| Date | Event |
| --- | --- |
| Tue 2026-04-21 | Competition launches. Data snapshot available. |
| Mon 2026-05-04 23:59 | **Submission 1 deadline** (Gradescope) |
| Wed–Fri 2026-05-06 to 05-08 | **Evaluation window 1** (3 trading days) |
| Sun 2026-05-10 23:59 | **Submission 2 deadline** (Gradescope) |
| Mon 2026-05-11 23:59 | **Report deadline** |
| Mon–Fri 2026-05-11 to 05-15 | **Evaluation window 2** (5 trading days) |


The A-share market is closed for the Labor Day holiday (May 1–5, 2026), which
is why evaluation window 1 starts on Wednesday May 6.

### 5. Evaluation Convention

For each window `[t_start, t_end]` (inclusive, trading days only):

- **Entry price** of stock `i` = forward-adjusted close on the last trading
  day strictly before `t_start`. If unavailable, fall back to the open of
  `t_start`.
- **Exit price** of stock `i` = forward-adjusted close on `t_end`. If `t_end`
  has no print (suspended), use the last available close within the window.
- **Per-stock return** `r_i = exit_i / entry_i − 1`.
- **Portfolio return** `R_p = Σ_i w_i · r_i`.
- **Benchmark return** `R_b` = same calculation on the CSI500 index (`000905`).
- **Excess return** `e = R_p − R_b`.

A stock that is suspended for the entire window is held at entry price — its
contribution is zero. Stocks delisted during the window are handled the same
way.

### 6. Grading

Final grade is weighted across four components:

| Weight | Component | How it is graded |
| --- | --- | --- |
| **40%** | Report | Written report documenting method, experiments, and results. Your report must explain:
Factors: what signals you used and why;
Models: how you trained the prediction model and built the portfolio;
Results: how well it performed on held-out data and relative to the baseline;
Analysis: what worked, what did not, and why. Each counts 25% of the report score.|
| **25%** | Self-test | Student must split the provided data into **training / validation / test** sets and report their model's performance on the test set. Graded on (a) soundness of the ML methodology (proper splitting, no leakage, sensible validation) and (b) whether the reported test performance exceeds the provided baseline. |
| **5%** | Live evaluation 1 (May 6–8) | Ranking-based: score is a function of where the student's portfolio ranks by excess return over this 3-day window. |
| **30%** | Live evaluation 2 (May 11–15) | Ranking-based: score is a function of where the student's portfolio ranks by excess return over this 5-day window. |

The exact rank-to-score mapping for §6 rows 3 and 4 will be announced before
the first live evaluation.

**Ranking metric for the live evaluations** — excess return `e` as defined
in §5, computed independently per window.

Secondary metrics (information ratio, max drawdown, turnover) are reported
for context but do not affect the grade.

### 7. Allowed Methods and Data

- **Allowed**:
  - Any machine-learning or statistical model trained *from scratch* by the
    student (GBDT, linear models, deep networks, etc).
  - Any publicly available data with a timestamp no later than the submission
    deadline (fundamentals, news, alternative data, macro series).
- **Not allowed**:
  - **Pretrained models** of any kind (including but not limited to time-series
    foundation models, financial LLMs, pretrained embeddings).
  - **LLMs used to produce the portfolio directly** (e.g. prompting an LLM for
    stock picks). LLMs may be used as coding or brainstorming assistants, but
    the prediction that determines weights must come from a model the student
    trained themselves.
  - Any data that is not public, or that you do not have a license to use.
  - Any data with a timestamp after the submission deadline (no look-ahead).
  - Sharing code or portfolios with other students.

### 8. Submission Procedure

- Channel: **Gradescope**. Upload before the deadline in §4.
- Submissions are overwritten by later uploads; **only the final version on
  Gradescope before the deadline counts**.
- Run `python validate_submission.py <file>` locally first — Gradescope runs
  the same checks and rejects anything that fails.

### 9. Submission Format

CSV with exactly one header row and two columns:

```
stock_code,weight
000001,0.0333
600000,0.0250
...
```

- `stock_code` — zero-padded 6-digit string of a CSI500 constituent.
- `weight` — float in `[0, 0.10]`, all rows sum to `1.0 ± 1e-4`.
- No additional columns, no trailing whitespace, UTF-8 encoded.

### 10. Reproducibility

**Every student** must submit, alongside the report:

- All code used to train the model and produce the two portfolio submissions.
- A `README` or equivalent that explains how to install dependencies and
  rerun the pipeline end-to-end.
- Any non-default random seeds or config needed to reproduce the submitted
  portfolios.

The course staff must be able to rerun the pipeline and reproduce (within
numerical noise) the portfolios that were uploaded to Gradescope. Submissions
that cannot be reproduced will lose points on the report component.

### 11. Academic Integrity

- No sharing of code, features, or portfolios between students.
- Any use of third-party code, tutorials, or LLM coding assistance must be
  explicitly acknowledged in the report.
- Violations are handled per the standard course policy.

### 12. Changes to These Rules

Instructors reserve the right to clarify ambiguities; any rule changes will
be announced on the course LMS and stamped with a date. No change will be
applied retroactively to already-submitted portfolios.

## Quickstart

```bash
# 1. Install deps (Python >= 3.10 recommended)
pip install -r requirements.txt

# 2. Download the data snapshot (first time is slow — ~10 min)
python download_data.py --start 20250101 --end 20260421

# 3. Run the baseline end-to-end
python baseline_xgboost.py --out submissions/week1.csv

# 4. Check your submission against the rules
python validate_submission.py submissions/week1.csv

# 5. Later, before submitting week 2, refresh data
python download_data.py --update --end 20260510
```

## Files

| File | Purpose |
| --- | --- |
| `download_data.py` | Fetch CSI500 constituents, OHLCV, and index from akshare |
| `features.py` | Feature engineering module (importable) |
| `baseline_xgboost.py` | End-to-end GBDT baseline |
| `validate_submission.py` | Pre-submission constraint check |
| `score_submission.py` | Realized-return scoring (used by TAs; works locally too) |
| `submission_example.csv` | Format reference |

## Data

`download_data.py` writes three files under `./data/`:

- **`constituents.csv`** — current CSI500 members (`stock_code`, `stock_name`, `as_of_date`).
- **`prices.parquet`** — daily bars per constituent. Columns: `date`,
  `stock_code`, `open`, `close`, `high`, `low`, `volume`, `amount`, `turnover`,
  `pct_change`. Close is **forward-adjusted** (qfq) for splits/dividends.
  `turnover` is a **fraction in [0, 1]** (not a percentage); `pct_change` is
  in **percent**.
- **`index.parquet`** — daily bars for the CSI500 index itself (the benchmark).

Note: `constituents.csv` is filtered at the end of `download_data.py` to
include only codes whose price data was fetched successfully. If a new CSI500
constituent doesn't yet have history available through akshare's sina
backend, it is dropped from your universe — you cannot select it.

**Heads up — the effective universe is 499 stocks, not 500.** Downloading
one constituent was unstable, so we omit it. Your universe is whatever ends
up in `constituents.csv`.

### Updating data during the competition

The competition runs for multiple weeks, so new trading days become available
every day the market is open. You are expected to **re-run the download
script before each submission deadline** to pick up the latest bars and
constituent changes.

```bash
# first time (full history — ~10 min for 500 stocks × 1 year)
python download_data.py --start 20250101 --end 20260421

# later (incremental — resumes from max date in prices.parquet)
python download_data.py --update --end 20260503  # before submission 1
python download_data.py --update --end 20260510  # before submission 2
```

The `--update` flag reads `data/prices.parquet`, finds the latest date
already cached, and only fetches rows after that. It also refreshes the
constituent list and the benchmark index. An incremental update of a few
trading days takes a few minutes rather than the full download's hour.

You are free to augment the data with anything public (fundamentals, news,
alternative data).  Document any extra data source in your report.

## Submission format

A CSV with exactly two columns and one header row:

```
stock_code,weight
000001,0.0333
600000,0.0250
...
```

`stock_code` is a zero-padded 6-digit string.  Upload through the course LMS
(filename `<team>_week<N>.csv`).  Run `validate_submission.py` first — we run
the same checks on the server.

## Caveats & tips

- **Look-ahead leakage** is the #1 failure mode. Any target shifted from the
  future must never be joined back to the feature frame without an offset.
- **Suspensions / limit-locked days** happen. If you weight a stock that is
  halted on the entry day, your realized weight effectively moves to cash (we
  hold the stock at its pre-halt price in our scoring), so avoid putting
  meaningful weight on stocks with recent trading halts.
- **Turnover isn't penalized** in this iteration, but rebalancing every week
  in a real portfolio would cost ~10–15 bps per side.
- The baseline's **validation rank IC fluctuates a lot** depending on the
  training window (observed range ~0.03 to ~0.17 in spot checks). A single IC
  number is noisy — use cross-validation across multiple windows in your
  self-test to get a reliable estimate.
