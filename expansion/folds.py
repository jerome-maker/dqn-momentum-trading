# -*- coding: utf-8 -*-
"""滾動前推（walk-forward）切分：每個測試年度 = 該曆年全部交易日；
訓練 = 測試年度第一天之前的 1200 個交易日；測試序列前面接 30 天暖身（只算指標、不交易）。"""
import os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN_LEN = 1200
WARMUP = 30
TEST_YEARS = {tk: list(range(2010, 2026)) for tk in ["SPY", "EWA", "EZA", "EWJ", "GLD"]}
TEST_YEARS["0050.TW"] = list(range(2019, 2026))   # 0050 乾淨資料自 2014 起


def load_panel():
    return pd.read_pickle(os.path.join(HERE, "data", "clean_close_panel.pkl"))


def make_folds(panel=None):
    panel = panel if panel is not None else load_panel()
    folds = []
    for tk, years in TEST_YEARS.items():
        c = panel[tk]
        for y in years:
            idx = c.index
            first = idx.searchsorted(pd.Timestamp(f"{y}-01-01"))
            last = idx.searchsorted(pd.Timestamp(f"{y+1}-01-01"))
            assert first >= TRAIN_LEN, (tk, y)
            train = c.iloc[first - TRAIN_LEN:first]
            test = c.iloc[first - WARMUP:last]
            folds.append(dict(ticker=tk, year=y, train=train.values, test=test.values,
                              train_start=train.index[0].date(), train_end=train.index[-1].date(),
                              test_start=idx[first].date(), test_end=idx[last - 1].date(),
                              n_test=last - first))
    return folds


if __name__ == "__main__":
    f = make_folds()
    print(len(f), "asset-years")
    df = pd.DataFrame([{k: v for k, v in d.items() if k not in ("train", "test")} for d in f])
    df.to_csv(os.path.join(HERE, "data", "fold_definitions.csv"), index=False)
    print(df.groupby("ticker").agg(n=("year", "size"), first=("year", "min"), last=("year", "max"),
                                   min_test_days=("n_test", "min"), max_test_days=("n_test", "max")))
