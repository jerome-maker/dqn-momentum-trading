# -*- coding: utf-8 -*-
"""機制分析：(1) 塑形組的超額報酬是否追隨動能法則的表現；(2) 原始報酬組的崩潰是否集中在下跌年度。
以 fold 為單位；斜率的信賴區間用 fold 群集拔靴法。"""
import os
import numpy as np, pandas as pd
from scipy.stats import spearmanr
A = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "analysis")
fm = pd.read_csv(os.path.join(A, "fold_level_means.csv"))
bm = pd.read_csv(os.path.join(A, "benchmark_metrics.csv")).pivot(index="fold", columns="bench", values="total_return")
w = fm.pivot(index="fold", columns="mode", values="total_return")
col = fm.pivot(index="fold", columns="mode", values="collapsed")
rng = np.random.default_rng(1)

def ols_boot(x, y, n=5000):
    X = np.c_[np.ones(len(x)), x]
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    bs = []
    for _ in range(n):
        i = rng.integers(0, len(x), len(x))
        bs.append(np.linalg.lstsq(X[i], y[i], rcond=None)[0][1])
    return b[1], np.percentile(bs, 2.5), np.percentile(bs, 97.5)

out = []
x = (bm["mom50"] - bm["bh50"]).values           # 動能擇時相對同曝險買入持有的價值
y = (w["momentum_shaped"] - w["raw_return"]).values
s, lo, hi = ols_boot(x, y); r, p = spearmanr(x, y)
out.append(f"(1) shaped−raw 報酬 對 (50%動能法則−50%買入持有) 報酬：斜率 {s:.2f} (95% CI {lo:.2f}, {hi:.2f})；Spearman ρ={r:.2f}, p={p:.2g}")
for mo in ["raw_return", "raw_scaled", "momentum_only", "potential_based"]:
    yy = (w["momentum_shaped"] - w[mo]).values if mo != "raw_return" else y
r2, p2 = spearmanr(bm["mom50"].values, w["momentum_shaped"].values)
out.append(f"    shaped 報酬 與 50%動能法則報酬 的 Spearman ρ={r2:.2f}, p={p2:.2g}；raw 報酬與之 ρ={spearmanr(bm['mom50'].values, w['raw_return'].values)[0]:.2f}")

mkt = bm["bh100"].values
for mo in ["raw_return", "raw_scaled", "drawdown_only", "differential_sharpe", "momentum_shaped"]:
    rr, pp = spearmanr(mkt, col[mo].values)
    down = col[mo].values[mkt < 0].mean(); up = col[mo].values[mkt >= 0].mean()
    out.append(f"(2) {mo:<20} 崩潰率：下跌年度 {down:.1%} vs 上漲年度 {up:.1%}；與市場報酬 Spearman ρ={rr:.2f}, p={pp:.2g}")
out.append(f"    下跌年度數 {(mkt<0).sum()} / {len(mkt)}")
# 崩潰的機會成本：raw 組零交易 agent 若改持 50% 買入持有
txt = "\n".join(out); print(txt)
open(os.path.join(A, "mechanism.txt"), "w", encoding="utf-8").write(txt)
