# -*- coding: utf-8 -*-
"""主實驗統計分析。

分析單位：資產-年度（fold）。同一 fold 的 10 個種子共用同一段市場資料，不是獨立樣本，
所以先在 fold 內取平均，再對 87 個 fold 做配對檢定；信賴區間用以 fold 為群集的拔靴法。
"""
import os, sys, json, glob, pickle
import numpy as np, pandas as pd
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import engine as E, folds as Fd

RES = os.environ.get("RESULTS_DIR", os.path.join(HERE, "results"))
OUT = os.path.join(RES, "analysis")
os.makedirs(OUT, exist_ok=True)
RNG = np.random.default_rng(20261015)
N_BOOT = 5000

cells = [''.join(c['source']) for c in json.load(open(os.path.join(HERE, '..', 'DQN_Momentum_Strategy.ipynb'), encoding='utf-8'))['cells'] if c['cell_type'] == 'code']
_g = {'np': np, 'pd': pd}
exec(next(c for c in cells if 'def calculate_metrics' in c), _g)
calc_metrics = _g['calculate_metrics']
classic_momentum_strategy = _g['classic_momentum_strategy']

MODE_ORDER = E.MODE_NAMES
LABEL = {"raw_return": "Raw return", "momentum_shaped": "Momentum-shaped", "raw_scaled": "Raw ×4.3 (scale control)",
         "momentum_only": "Momentum bonus only", "drawdown_only": "Drawdown penalty only",
         "potential_based": "Potential-based shaping", "differential_sharpe": "Differential Sharpe"}
METRICS = ["total_return", "sharpe_ratio", "max_drawdown", "max_drawdown_duration", "avg_drawdown_duration"]


def holding_from_actions(actions):
    """由動作序列重建每步結束時是否持有部位（買入只在空手時成交、賣出只在持有時成交）。"""
    h = np.zeros(len(actions), bool)
    pos = False
    for t, a in enumerate(actions):
        if a == 2 and not pos:
            pos = True
        elif a == 0 and pos:
            pos = False
        h[t] = pos
    return h


def benchmarks(fold, feats_te):
    """每個 fold 的固定規則基準（曝險 50% 與 100% 兩種版本）。"""
    test = fold["test"]
    P, F = test[None].astype(np.float64), feats_te[None]
    out = {}
    # 50% 買入持有：在 agent 的動作空間中一直按「買入」
    env = E.NumpyEnv(P, F, np.zeros(1, np.int64)); env.reset(); h = [E.INIT]
    while True:
        _, _, d, v = env.step(np.array([2])); h.append(v[0])
        if d[0]: break
    out["bh50"] = np.array(h)
    # 100% 買入持有
    pos = E.INIT * (1 - E.KAPPA) / test[E.START]
    out["bh100"] = np.r_[E.INIT, pos * test[E.START + 1:]]
    # 50% 動能法則：在 agent 的動作空間中，M10>0 按買入、否則按賣出
    env = E.NumpyEnv(P, F, np.zeros(1, np.int64)); env.reset(); h = [E.INIT]; acts = []
    while True:
        a = 2 if F[0, env.cs, 2] > 0 else 0
        acts.append(a)
        _, _, d, v = env.step(np.array([a])); h.append(v[0])
        if d[0]: break
    out["mom50"] = np.array(h); out["mom50_hold"] = holding_from_actions(np.array(acts))
    # 100% 動能法則（notebook 原始函式）
    out["mom100"] = classic_momentum_strategy(pd.Series(test))
    return out


def cluster_boot_mean(x, n=N_BOOT):
    """x: 每個 fold 一個值 → 平均數與 fold 群集拔靴 95% CI。"""
    x = np.asarray(x, float)
    idx = RNG.integers(0, len(x), (n, len(x)))
    b = x[idx].mean(1)
    return x.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def main():
    files = sorted(glob.glob(os.path.join(RES, "batch_*.pkl")))
    assert files, "沒有結果檔"
    folds = Fd.make_folds()
    feats_tr, feats_te = pickle.load(open(os.path.join(HERE, "data", "features_cache.pkl"), "rb"))

    # ---------- 每個 agent 的指標 ----------
    rows, curves = [], []
    secs, peaks = [], []
    for fn in files:
        d = pickle.load(open(fn, "rb"))
        secs.append(d["secs"]); peaks.append(d["peak_mib"])
        for r in d["rows"]:
            m = calc_metrics(r["history"])
            m = {k: float(v) for k, v in m.items()}
            hold = holding_from_actions(r["actions"])
            m.update(batch=r["batch"], fold=r["fold"], ticker=r["ticker"], year=r["year"], mode=r["mode"],
                     trades=r["trades"], collapsed=int(r["trades"] == 0), exposure=float(hold.mean()))
            m["_hold"] = hold
            rows.append(m)
        curves.append(pd.DataFrame({"mode": [E.MODE_NAMES[i] for i in d["mode"]], "fold": d["fi"]})
                      .assign(**{f"ep{e}": d["ep_reward"][e] for e in range(0, d["ep_reward"].shape[0], 10)}))
    df = pd.DataFrame(rows)

    # ---------- 基準 ----------
    bench_rows, rule_hold = [], {}
    for fi, f in enumerate(folds):
        b = benchmarks(f, feats_te[fi])
        rule_hold[fi] = b["mom50_hold"]
        for k in ("bh50", "bh100", "mom50", "mom100"):
            m = {kk: float(v) for kk, v in calc_metrics(b[k]).items()}
            m.update(fold=fi, ticker=f["ticker"], year=f["year"], bench=k)
            bench_rows.append(m)
    bench = pd.DataFrame(bench_rows)

    # 與 50% 動能法則的持倉一致率
    df["agree_mom50"] = [float((r["_hold"][:len(rule_hold[r["fold"]])] ==
                                rule_hold[r["fold"]][:len(r["_hold"])]).mean()) for _, r in df.iterrows()]
    df_out = df.drop(columns="_hold")
    df_out.to_csv(os.path.join(OUT, "agent_level_metrics.csv"), index=False)
    bench.to_csv(os.path.join(OUT, "benchmark_metrics.csv"), index=False)

    # ---------- fold 層級平均 ----------
    cols = METRICS + ["collapsed", "trades", "exposure", "agree_mom50"]
    fm = df_out.groupby(["fold", "ticker", "year", "mode"])[cols].mean().reset_index()
    n_seeds = df_out.groupby(["fold", "mode"]).size()
    fm.to_csv(os.path.join(OUT, "fold_level_means.csv"), index=False)
    wide = {c: fm.pivot(index="fold", columns="mode", values=c)[MODE_ORDER] for c in cols}
    bw = bench.pivot(index="fold", columns="bench", values="total_return")

    lines = []
    P = lines.append
    P(f"# 主實驗統計分析\n")
    P(f"- 結果檔 {len(files)} 批；agent 數 {len(df)}；資產-年度 {df.fold.nunique()}；"
      f"每組合種子數 {n_seeds.min()}–{n_seeds.max()}")
    P(f"- 訓練耗時 {sum(secs)/60:.0f} 分（GPU + CPU 環境），峰值 VRAM {max(peaks):.0f} MiB\n")

    # 表 1：各獎勵函數摘要
    P("## 表 1　各獎勵函數（fold 平均後再跨 fold 平均；括號為 fold 群集拔靴 95% CI）\n")
    P("| 獎勵函數 | 總報酬 | 夏普比 | 最大回撤 | 崩潰率（零交易） | 平均曝險 | 與 50% 動能法則一致率 |")
    P("|---|---|---|---|---|---|---|")
    summary = []
    for mo in MODE_ORDER:
        cells_ = []
        for c in ["total_return", "sharpe_ratio", "max_drawdown", "collapsed", "exposure", "agree_mom50"]:
            m, lo, hi = cluster_boot_mean(wide[c][mo].values)
            pct = c in ("total_return", "max_drawdown", "collapsed", "exposure", "agree_mom50")
            cells_.append(f"{m:.2%} ({lo:.2%}, {hi:.2%})" if pct else f"{m:.2f} ({lo:.2f}, {hi:.2f})")
            summary.append(dict(mode=mo, metric=c, mean=m, lo=lo, hi=hi))
        P(f"| {LABEL[mo]} | " + " | ".join(cells_) + " |")
    pd.DataFrame(summary).to_csv(os.path.join(OUT, "table1_summary.csv"), index=False)
    P("")
    P("基準（每個 fold 一個值，跨 fold 平均）：")
    for k, lab in [("bh50", "50% 買入持有"), ("bh100", "100% 買入持有"), ("mom50", "50% 動能法則"), ("mom100", "100% 動能法則")]:
        m, lo, hi = cluster_boot_mean(bw[k].values)
        P(f"- {lab}：總報酬 {m:.2%} ({lo:.2%}, {hi:.2%})")
    P("")

    # 表 2：配對檢定
    P("## 表 2　配對比較（87 個 fold；Wilcoxon 符號等級檢定，同一指標內 Holm 校正）\n")
    tests = []
    for ref in ("raw_return", "momentum_shaped"):
        for c in ["total_return", "sharpe_ratio", "max_drawdown", "collapsed"]:
            others = [m for m in MODE_ORDER if m != ref]
            ps, recs = [], []
            for mo in others:
                diff = (wide[c][mo] - wide[c][ref]).values
                if np.allclose(diff, 0):
                    p = 1.0
                else:
                    p = wilcoxon(diff, zero_method="zsplit").pvalue
                m, lo, hi = cluster_boot_mean(diff)
                recs.append(dict(ref=ref, metric=c, mode=mo, mean_diff=m, lo=lo, hi=hi,
                                 median_diff=float(np.median(diff)), share_better=float((diff > 0).mean()), p=p))
                ps.append(p)
            for r, pa in zip(recs, holm(ps)):
                r["p_holm"] = pa
                tests.append(r)
    T = pd.DataFrame(tests)
    T.to_csv(os.path.join(OUT, "table2_paired_tests.csv"), index=False)
    for ref in ("raw_return", "momentum_shaped"):
        P(f"### 相對於 {LABEL[ref]}\n")
        P("| 指標 | 獎勵函數 | 平均差 (95% CI) | fold 中較高的比例 | Holm p |")
        P("|---|---|---|---|---|")
        for _, r in T[T.ref == ref].iterrows():
            pct = r.metric != "sharpe_ratio"
            f = (lambda v: f"{v:+.2%}") if pct else (lambda v: f"{v:+.2f}")
            P(f"| {r.metric} | {LABEL[r['mode']]} | {f(r.mean_diff)} ({f(r.lo)}, {f(r.hi)}) | {r.share_better:.0%} | {r.p_holm:.4f} |")
        P("")

    # 表 3：子樣本穩健性（總報酬與崩潰率）
    P("## 表 3　子樣本：動能塑形 − 原始報酬、尺度對照 − 原始報酬\n")
    P("| 子樣本 | fold 數 | 塑形−原始 總報酬 | 塑形−原始 崩潰率 | 尺度對照−原始 總報酬 | 尺度對照−原始 崩潰率 |")
    P("|---|---|---|---|---|---|")
    meta = fm.drop_duplicates("fold").set_index("fold")[["ticker", "year"]]
    groups = [("全部", meta.index)] + [(tk, meta.index[meta.ticker == tk]) for tk in Fd.TEST_YEARS] + \
             [("2010–2018", meta.index[meta.year <= 2018]), ("2019–2025（平衡面板）", meta.index[meta.year >= 2019])]
    for name, ix in groups:
        out = [name, str(len(ix))]
        for mo in ("momentum_shaped", "raw_scaled"):
            for c in ("total_return", "collapsed"):
                d_ = (wide[c].loc[ix, mo] - wide[c].loc[ix, "raw_return"]).values
                m, lo, hi = cluster_boot_mean(d_, 2000)
                out.append(f"{m:+.2%} ({lo:+.2%}, {hi:+.2%})")
        P("| " + " | ".join(out) + " |")
    P("")

    # 表 4：只看有交易的 agent 的回撤持續期間
    P("## 表 4　回撤持續期間：全部 agent vs 只看有交易的 agent（天）\n")
    P("| 獎勵函數 | 平均回撤持續（全部） | 平均回撤持續（有交易） | 最大回撤持續（全部） | 最大回撤持續（有交易） |")
    P("|---|---|---|---|---|")
    traded = df_out[df_out.collapsed == 0].groupby(["fold", "mode"])[["avg_drawdown_duration", "max_drawdown_duration"]].mean().reset_index()
    for mo in MODE_ORDER:
        a_all = wide["avg_drawdown_duration"][mo].mean(); m_all = wide["max_drawdown_duration"][mo].mean()
        t = traded[traded["mode"] == mo]
        P(f"| {LABEL[mo]} | {a_all:.1f} | {t.avg_drawdown_duration.mean():.1f} | {m_all:.1f} | {t.max_drawdown_duration.mean():.1f} |")
    P("")

    pd.concat(curves).to_csv(os.path.join(OUT, "training_curves_every10.csv"), index=False)
    open(os.path.join(OUT, "analysis_report.md"), "w", encoding="utf-8").write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
