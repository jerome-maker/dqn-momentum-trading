# -*- coding: utf-8 -*-
"""價格資料指紋。原始價格（Yahoo Finance）依資料提供者條款不重新散布；
重現者用 data_fetch.py 依固定日期重新下載後，執行本程式與 data/price_data_fingerprint.csv 比對。
- sha256：逐位元比對（受 pandas 版本的 CSV 寫出格式影響）
- 內容摘要：列數、起訖日、首末收盤價、收盤價總和（不受檔案格式影響）"""
import os, sys, glob, hashlib
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data")
REF = os.path.join(D, "price_data_fingerprint.csv")

rows = []
for fn in sorted(glob.glob(os.path.join(D, "*_2004-01-02_2026-06-23.csv"))):
    c = pd.read_csv(fn, parse_dates=["Date"], index_col="Date")["Close"]
    rows.append(dict(file=os.path.basename(fn), sha256=hashlib.sha256(open(fn, "rb").read()).hexdigest(),
                     rows=len(c), first_date=c.index[0].date().isoformat(), last_date=c.index[-1].date().isoformat(),
                     first_close=f"{c.iloc[0]:.6f}", last_close=f"{c.iloc[-1]:.6f}", sum_close=f"{c.sum():.4f}"))
cur = pd.DataFrame(rows)
if "--write" in sys.argv or not os.path.exists(REF):
    cur.to_csv(REF, index=False)
    print("wrote", REF)
    print(cur.to_string(index=False))
else:
    ref = pd.read_csv(REF, dtype=str)
    m = ref.merge(cur.astype(str), on="file", suffixes=("_ref", "_now"))
    for col in ["sha256", "rows", "first_date", "last_date", "first_close", "last_close", "sum_close"]:
        m[col + "_ok"] = m[col + "_ref"] == m[col + "_now"]
    print(m[["file"] + [c for c in m.columns if c.endswith("_ok")]].to_string(index=False))
