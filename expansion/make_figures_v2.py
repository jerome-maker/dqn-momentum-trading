# -*- coding: utf-8 -*-
"""新稿（Computational Economics）圖表。單一色相為主；兩組資料時另以點形狀區分，黑白列印可讀。
圖內不放標題（期刊規定），無襯線字體 8–9 pt，寬 174 mm。"""
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import folds as Fd

HERE = os.path.dirname(os.path.abspath(__file__))
A = os.path.join(HERE, "results", "analysis")
OUT = os.environ.get("FIG_DIR", os.path.join(HERE, "..", "manuscript_v2_CompEcon"))
os.makedirs(OUT, exist_ok=True)
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1f1f1f", "#6b6b6b", "#e4e4e1"
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                     "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                     "legend.fontsize": 7.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.linewidth": 0.6, "lines.linewidth": 1.2,
                     "pdf.fonttype": 42, "ps.fonttype": 42})
MM = 1 / 25.4
W = 174 * MM
LABEL = {"raw_return": "Raw return", "momentum_shaped": "Momentum-shaped", "raw_scaled": "Raw ×4.3 (scale control)",
         "momentum_only": "Momentum bonus only", "drawdown_only": "Drawdown penalty only",
         "potential_based": "Potential-based", "differential_sharpe": "Differential Sharpe"}
ORDER = ["raw_return", "raw_scaled", "drawdown_only", "differential_sharpe", "potential_based",
         "momentum_shaped", "momentum_only"]


def save(fig, n):
    for ext in ("pdf", "eps"):
        fig.savefig(os.path.join(OUT, f"Fig{n}.{ext}"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, f"Fig{n}.png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def grid(ax, axis="x"):
    ax.grid(axis=axis, color=GRID, linewidth=0.6); ax.set_axisbelow(True)


# ---------------- Fig 1：六個資產的價格與測試年度 ----------------
panel = Fd.load_panel()
names = {"0050.TW": "Taiwan (0050.TW)", "SPY": "United States (SPY)", "EWA": "Australia (EWA)",
         "EZA": "South Africa (EZA)", "EWJ": "Japan (EWJ)", "GLD": "Gold (GLD)"}
fig, axes = plt.subplots(2, 3, figsize=(W, 82 * MM), sharex=True)
for ax, tk in zip(axes.ravel(), names):
    c = panel[tk]
    y0 = Fd.TEST_YEARS[tk][0]
    ax.axvspan(pd.Timestamp(f"{y0}-01-01"), pd.Timestamp("2025-12-31"), color="#ececea", lw=0)
    ax.plot(c.index, c.values, color=BLUE, lw=0.8)
    ax.set_yscale("log")
    from matplotlib.ticker import LogLocator, FuncFormatter, NullFormatter
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.text(0.03, 0.92, names[tk], transform=ax.transAxes, fontsize=7.5, color=INK, va="top")
    ax.tick_params(axis="x", labelrotation=0)
    grid(ax, "y")
for ax in axes[:, 0]:
    ax.set_ylabel("Adjusted close (log scale)")
fig.tight_layout(w_pad=1.0, h_pad=0.8)
save(fig, 1)

# ---------------- Fig 2：崩潰率 ----------------
s = pd.read_csv(os.path.join(A, "table1_summary.csv"))
c = s[s.metric == "collapsed"].set_index("mode").loc[ORDER]
fig, ax = plt.subplots(figsize=(119 * MM, 60 * MM))
y = np.arange(len(ORDER))[::-1]
ax.barh(y, c["mean"] * 100, color=BLUE, height=0.6)
ax.errorbar(c["mean"] * 100, y, xerr=[(c["mean"] - c["lo"]) * 100, (c["hi"] - c["mean"]) * 100],
            fmt="none", ecolor=INK, elinewidth=0.8, capsize=2)
for yy, v, hi in zip(y, c["mean"] * 100, c["hi"] * 100):
    ax.text(hi + 1.2, yy, f"{v:.1f}%", va="center", fontsize=7.5, color=INK)
ax.set_yticks(y); ax.set_yticklabels([LABEL[m] for m in ORDER])
ax.set_xlabel("Share of agents that never trade in the test year (%)")
ax.set_xlim(0, 55); grid(ax)
save(fig, 2)

# ---------------- Fig 3：相對原始報酬的配對差異（森林圖） ----------------
t = pd.read_csv(os.path.join(A, "table2_paired_tests.csv"))
t = t[t.ref == "raw_return"]
others = [m for m in ORDER if m != "raw_return"]
fig, axes = plt.subplots(1, 3, figsize=(W, 62 * MM), sharey=True)
for ax, (met, xl, scale) in zip(axes, [("total_return", "(a) Total return (pp)", 100),
                                       ("sharpe_ratio", "(b) Sharpe ratio", 1),
                                       ("max_drawdown", "(c) Maximum drawdown (pp)", 100)]):
    d = t[t.metric == met].set_index("mode").loc[others]
    yy = np.arange(len(others))[::-1]
    ax.axvline(0, color=MUTED, lw=0.7, ls="--")
    sig = d.p_holm < 0.05
    ax.errorbar(d.mean_diff * scale, yy, xerr=[(d.mean_diff - d.lo) * scale, (d.hi - d.mean_diff) * scale],
                fmt="none", ecolor=BLUE, elinewidth=1.0, capsize=0)
    ax.scatter(d.mean_diff[sig] * scale, yy[sig.values], s=22, color=BLUE, zorder=3, label="Holm p < 0.05")
    ax.scatter(d.mean_diff[~sig] * scale, yy[~sig.values], s=22, facecolor="white", edgecolor=BLUE,
               zorder=3, label="Holm p ≥ 0.05")
    ax.set_xlabel(xl); grid(ax)
axes[0].set_yticks(np.arange(len(others))[::-1]); axes[0].set_yticklabels([LABEL[m] for m in others])
axes[2].legend(loc="lower right", frameon=False)
fig.supxlabel("Mean paired difference relative to raw return (87 asset-years)", fontsize=8, color=INK)
fig.tight_layout(w_pad=1.2)
save(fig, 3)

# ---------------- Fig 4：塑形 agent 報酬 vs 50% 動能法則 ----------------
fm = pd.read_csv(os.path.join(A, "fold_level_means.csv"))
bm = pd.read_csv(os.path.join(A, "benchmark_metrics.csv")).pivot(index="fold", columns="bench", values="total_return")
w = fm.pivot(index="fold", columns="mode", values="total_return")
fig, ax = plt.subplots(figsize=(119 * MM, 88 * MM))
lim = [min(bm.mom50.min(), w.min().min()) * 100 - 2, max(bm.mom50.max(), w.max().max()) * 100 + 2]
ax.plot(lim, lim, color=MUTED, lw=0.7, ls="--")
ax.scatter(bm.mom50 * 100, w.raw_return * 100, s=16, marker="^", facecolor="white", edgecolor=ORANGE,
           linewidth=0.9, label="Raw return (ρ = 0.22)")
ax.scatter(bm.mom50 * 100, w.momentum_shaped * 100, s=14, color=BLUE, label="Momentum-shaped (ρ = 0.87)")
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("50%-exposure momentum rule: test-year return (%)")
ax.set_ylabel("DQN agent: test-year return, mean of 10 seeds (%)")
ax.text(lim[1] - 1, lim[1] - 3, "45° line", color=MUTED, fontsize=7, ha="right")
ax.legend(loc="upper left", frameon=False); grid(ax, "both")
save(fig, 4)

# ---------------- Fig 5：子樣本 ----------------
fmw = {k: fm.pivot(index="fold", columns="mode", values=k) for k in ("total_return", "collapsed")}
meta = fm.drop_duplicates("fold").set_index("fold")[["ticker", "year"]]
rng = np.random.default_rng(7)
groups = [("All (87)", meta.index)] + [(f"{tk} ({(meta.ticker == tk).sum()})", meta.index[meta.ticker == tk])
          for tk in ["SPY", "EWA", "EZA", "EWJ", "GLD", "0050.TW"]] + \
         [("2010–2018 (45)", meta.index[meta.year <= 2018]), ("2019–2025 (42)", meta.index[meta.year >= 2019])]
fig, axes = plt.subplots(1, 2, figsize=(W, 66 * MM), sharey=True)
for ax, (k, xl) in zip(axes, [("total_return", "Momentum-shaped − raw: total return (pp)"),
                             ("collapsed", "Momentum-shaped − raw: collapse rate (pp)")]):
    yy = np.arange(len(groups))[::-1]
    ax.axvline(0, color=MUTED, lw=0.7, ls="--")
    for y_, (g, ix) in zip(yy, groups):
        d = (fmw[k].loc[ix, "momentum_shaped"] - fmw[k].loc[ix, "raw_return"]).values * 100
        b = d[rng.integers(0, len(d), (4000, len(d)))].mean(1)
        lo, hi = np.percentile(b, [2.5, 97.5])
        ax.plot([lo, hi], [y_, y_], color=BLUE, lw=1.0)
        ax.scatter(d.mean(), y_, s=20, color=BLUE if (lo > 0 or hi < 0) else "white", edgecolor=BLUE, zorder=3)
    ax.set_xlabel(xl); grid(ax)
axes[0].set_yticks(np.arange(len(groups))[::-1]); axes[0].set_yticklabels([g for g, _ in groups])
fig.tight_layout(w_pad=1.2)
save(fig, 5)
print("figures written to", os.path.abspath(OUT))
