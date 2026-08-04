# DQN Momentum Trading: Reward Shaping Experiments

A Deep Q-Network (DQN) trading agent for the Yuanta Taiwan Top 50 ETF (0050.TW),
used to study how reward-function design affects training performance and
stability.

## What this notebook does

- Implements a custom `MomentumEnvironment` (state space of price momentum, RSI,
  MACD, position/cash ratios, return, and volatility; discrete
  sell/hold/buy actions) and a `MomentumDQNAgent` (2-layer MLP, target network,
  experience replay, GPU-aware).
- Compares two reward specifications across 10 random seeds each (500 episodes,
  90/10 non-overlapping train/test split on 5 years of daily data,
  22 June 2021 to 22 June 2026, pinned to an exact date range for
  reproducibility):
  - **`raw_return`** — unshaped portfolio return (ablation baseline)
  - **`momentum_shaped`** — return plus momentum-contingent bonuses/penalties
    and a drawdown penalty
- Benchmarks both against a buy-and-hold strategy and a classical (non-learned)
  time-series momentum rule.
- Reports total return, Sharpe ratio, max drawdown, and drawdown-duration
  statistics, with Mann-Whitney U significance testing across seeds.

Key finding: the shaped reward significantly improves both average performance
and training stability — the unshaped reward produces a degenerate,
non-trading policy in a meaningful fraction of training runs, a failure mode
that is not visible from training-time reward/loss curves alone (see the
"Training Convergence" section of the notebook).

## Requirements

See `requirements.txt`. Tested with Python 3.11.

```bash
pip install -r requirements.txt
```

## Running

Open `DQN_Momentum_Strategy.ipynb` in Jupyter Lab/Notebook or VS Code and run
all cells. Sections are numbered to mirror the accompanying manuscript: 1-6
define the environment/model/data, 7 computes the buy-and-hold benchmark, 8
runs the 20-run reward-comparison experiment, and 9 runs a supplementary
training-convergence analysis. The 20-run experiment in Section 8 is by far
the most expensive part (roughly 15-20 minutes per run, ~5-6 hours total on
CPU); it uses a local checkpoint file so it can be interrupted and resumed
without losing completed runs.

Note: the notebook auto-detects a CUDA GPU if available, but for this network
size (two 128-unit hidden layers, batch size 32) CPU was empirically ~1.7x
*faster* than GPU due to per-step host-device transfer overhead — GPU is not
recommended for reproducing these results.

## Data

Price data is downloaded on demand from Yahoo Finance (`yfinance`); no data
files are included in this repository.
