# -*- coding: utf-8 -*-
"""下載擴充資料集並匯出 CSV（固定起訖日期，可重現）。

六個資產：台灣、美國、澳洲、南非、日本股市 ETF，加上黃金（非股票資產類別）。
期間 2004-01-02 ~ 2026-06-22（yfinance 的 end 不含當日，所以設 2026-06-23）。
"""
import os
import sys
import numpy as np
import pandas as pd
import yfinance as yf

DATA_START = "2004-01-02"
DATA_END_EXCLUSIVE = "2026-06-23"
TICKERS = {
    "0050.TW": "Taiwan (Yuanta Taiwan Top 50 ETF)",
    "SPY": "United States (SPDR S&P 500 ETF)",
    "EWA": "Australia (iShares MSCI Australia ETF)",
    "EZA": "South Africa (iShares MSCI South Africa ETF)",
    "EWJ": "Japan (iShares MSCI Japan ETF)",
    "GLD": "Gold (SPDR Gold Shares)",
}
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

rows = []
for tk, desc in TICKERS.items():
    df = yf.download(tk, start=DATA_START, end=DATA_END_EXCLUSIVE, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])
    df.index.name = "Date"
    fn = os.path.join(OUT, f"{tk.replace('.', '_')}_{DATA_START}_{DATA_END_EXCLUSIVE}.csv")
    df.to_csv(fn)  # 完整精度，確保可逐位元重現

    c = df["Close"]
    r = c.pct_change().dropna()
    gaps = df.index.to_series().diff().dt.days
    rows.append({
        "ticker": tk, "desc": desc, "first": df.index[0].date(), "last": df.index[-1].date(),
        "rows": len(df), "nan_close": int(c.isna().sum()), "nonpos_close": int((c <= 0).sum()),
        "dup_dates": int(df.index.duplicated().sum()), "max_gap_days": int(gaps.max()),
        "max_abs_daily_ret": float(r.abs().max()), "date_of_max": r.abs().idxmax().date(),
        "n_abs_ret_gt_15pct": int((r.abs() > 0.15).sum()),
    })

qc = pd.DataFrame(rows)

# ---- 清理後的實驗用面板 ----
# 0050.TW：yfinance 對 2025 年一拆四的還原只套用到 2014-01-02 以後（當日價格 37.19 -> 9.27、
# 成交量放大約 4 倍），2009 年初另有 11 筆超過台股 ±7% 漲跌幅限制的雜訊。
# 因此 0050.TW 只採用 2014-01-02 以後的乾淨區段；其他資產全期間使用。
CLEAN_START = {"0050.TW": "2014-01-02"}
panel = {}
for tk in TICKERS:
    fn = os.path.join(OUT, f"{tk.replace('.', '_')}_{DATA_START}_{DATA_END_EXCLUSIVE}.csv")
    c = pd.read_csv(fn, parse_dates=["Date"], index_col="Date")["Close"]
    if tk in CLEAN_START:
        c = c[c.index >= CLEAN_START[tk]]
    panel[tk] = c
    r = c.pct_change().dropna()
    lim = 0.10 if tk != "0050.TW" else 0.07
    print(f"clean {tk}: {c.index[0].date()} ~ {c.index[-1].date()}  n={len(c)}  max|ret|={r.abs().max():.4f}")
pd.to_pickle(panel, os.path.join(OUT, "clean_close_panel.pkl"))
qc.to_csv(os.path.join(OUT, "data_quality_report.csv"), index=False)
pd.set_option("display.width", 200)
print(qc.to_string(index=False))
