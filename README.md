# Reward Shaping in Deep Q-Network Trading

Code, derived data and results for a study of how reward design affects Deep
Q-Network (DQN) trading agents. The repository has two parts:

| Part | What it is | Status of its headline claim |
|---|---|---|
| `expansion/` | **Main study.** 6 ETFs, 87 out-of-sample asset-years (2010–2025), 7 reward specifications, 10 seeds each — 6,090 agents | Current |
| `DQN_Momentum_Strategy.ipynb` | Original single-window study on 0050.TW (one 122-day test window) | **Superseded** — its performance claim does not generalize (see below) |

## Main findings (`expansion/`)

- **Momentum shaping cures policy collapse.** Under raw-return reward, 22.4% of
  agents never trade in their test year; under momentum shaping, 2.3%; with the
  momentum bonus alone, 0.7%.
- **It is not a reward-scale effect.** A raw-return control rescaled to match the
  shaped reward's per-step dispersion (×4.3) collapses as often as raw return (23.0%).
- **Shaping does not improve out-of-sample performance.** Across 87 asset-years,
  total return and Sharpe ratio are statistically indistinguishable from raw return
  (Wilcoxon signed-rank, Holm-adjusted), and maximum drawdown is deeper.
- **Mechanism: the agent learns the rule written into the reward.** Shaped agents hold
  the same position as a 10-day momentum rule on 77% of days (raw-return agents: 42%),
  and their test-year returns track that rule's returns across asset-years
  (Spearman ρ = 0.87; raw-return agents ρ = 0.22).
- **A single window can mislead.** On 0050.TW over 2019–2025 alone, shaped agents beat
  raw-return agents by +4.9 percentage points a year — the kind of result the original
  notebook reported. Across the other markets and periods the difference is zero or
  negative.

Full tables: `expansion/results/analysis/analysis_report.md`.

## Reproducing the main study

Requirements: see `requirements.txt` (Python 3.11, PyTorch with CUDA). Run from
`expansion/`:

```bash
python data_fetch.py          # download prices for the fixed window 2004-01-02 .. 2026-06-22
python data_fingerprint.py    # verify the download against data/price_data_fingerprint.csv
python folds.py               # build the 87 walk-forward asset-years
python validate_env.py        # step-by-step check of the GPU environment against the notebook's
python validate_env_numpy.py  # same check for the CPU environment used in training
python calibrate_scale.py     # reward-scale calibration for the scale-matched control (k = 4.3)
python run_grid.py            # train 6,090 agents; checkpoints after each batch (~2.5 h on a GTX 1060)
python analyze.py             # statistics (fold-level unit, Wilcoxon + Holm, cluster bootstrap)
python mechanism.py           # rule-imitation and collapse-vs-market analyses
python make_figures_v2.py     # figures
```

`validate_replication.py` re-runs the original single-window setting with the new
engine; it needs the checkpoint file produced by the notebook's Section 8.

### Implementation

`engine.py` trains thousands of independent DQN agents at once. Each agent has its own
network, Adam state, replay buffer and gradient-norm clipping; parameters are stacked
along an agent dimension so that one batched matrix multiplication on the GPU serves
all agents, while the trading environment is vectorized across agents on the CPU. The
environment reproduces the notebook's `MomentumEnvironment` exactly (states, portfolio
values and termination flags identical; rewards within 1e-12), and the learning
algorithm reproduces `MomentumDQNAgent`, including its quirk of leaving the target
network in training mode.

## Data

Prices are daily adjusted closes from Yahoo Finance for 0050.TW, SPY, EWA, EZA, EWJ and
GLD. **Raw price files are not included**, in keeping with the data provider's terms.
`data_fetch.py` downloads them for the fixed date range; `data_fingerprint.py` compares
the download with `data/price_data_fingerprint.csv` (SHA-256 of each file, plus row
counts, dates, first/last close and the sum of closes, which do not depend on CSV
formatting). Yahoo Finance may revise back-adjusted prices after new dividends, so a
later download can differ slightly; the fingerprint shows whether it does.

Data cleaning: in the source series for 0050.TW, the adjustment for the fund's 2025
four-for-one split is applied only from 2014-01-02, and 2009 contains moves beyond
Taiwan's ±7% price limit. 0050.TW is therefore used only from 2014-01-02
(`data/data_quality_report.csv`).

Included derived data and results:

| Path | Contents |
|---|---|
| `expansion/data/fold_definitions.csv` | Train/test dates of the 87 asset-years |
| `expansion/data/data_quality_report.csv` | Data audit |
| `expansion/results/analysis/agent_level_metrics.csv` | Metrics for every one of the 6,090 agents |
| `expansion/results/analysis/fold_level_means.csv` | Cell means used for inference |
| `expansion/results/analysis/benchmark_metrics.csv` | Buy-and-hold and momentum-rule benchmarks (50% and 100% exposure) |
| `expansion/results/analysis/table1_summary.csv`, `table2_paired_tests.csv` | Summary and paired tests |
| `expansion/results/analysis/training_curves_every10.csv` | Episode reward every 10 episodes, every agent |

## The original notebook

`DQN_Momentum_Strategy.ipynb` is the first, single-window version of this study
(0050.TW, 2021-06-22 to 2026-06-22, 90/10 split, 10 seeds). Its finding that shaping
significantly raised total return and Sharpe ratio held in that window but does not
hold across the 87 asset-years of the main study, for the reason given above. It is
kept as a teaching companion and as the replication target for
`expansion/validate_replication.py`.

Two corrections to descriptions that accompanied the original version: the drawdown
penalty in the code is `20 × DD × 1{DD > 0.1}` (not `20 × max(0, DD − 0.1)`), and its
buy-and-hold benchmark invests 50% of capital, not 100%.
