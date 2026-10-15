# -*- coding: utf-8 -*-
"""主實驗：87 個資產-年度 × 7 種獎勵函數 × 10 個重複（種子），GPU 批次訓練。
每批 = 全部 87×7 組合 × 2 個種子 = 1218 個 agent；共 5 批，每批完成即存檔，可中斷續跑。"""
import os, sys, time, pickle, json
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import engine as E, folds as Fd

OUT = os.environ.get("RESULTS_DIR", os.path.join(HERE, "results"))
os.makedirs(OUT, exist_ok=True)
LOG = os.path.join(OUT, "run_grid.log")
SCALE_K = 4.3            # calibrate_scale.py：隨機策略下 shaped/raw 逐步標準差中位數
MODES = list(range(7))
SEEDS_PER_BATCH = 2
N_BATCHES = int(os.environ.get("N_BATCHES", 5))
EPISODES = int(os.environ.get("EPISODES", 500))


def log(msg):
    line = f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type == "cpu":
        raise RuntimeError("預期要用 GPU 卻取到 CPU")
    folds = Fd.make_folds()
    cache = os.path.join(HERE, "data", "features_cache.pkl")
    if os.path.exists(cache):
        feats_tr, feats_te = pickle.load(open(cache, "rb"))
    else:
        feats_tr = [E.market_features(f["train"]) for f in folds]
        feats_te = [E.market_features(f["test"]) for f in folds]
        pickle.dump((feats_tr, feats_te), open(cache, "wb"))
    combos = [(fi, m, s) for s in range(SEEDS_PER_BATCH) for fi in range(len(folds)) for m in MODES]
    A = len(combos)
    fi_arr = np.array([c[0] for c in combos])
    mode_arr = np.array([c[1] for c in combos])
    log(f"device={torch.cuda.get_device_name(0)} folds={len(folds)} agents/batch={A} episodes={EPISODES}")

    for b in range(N_BATCHES):
        fn = os.path.join(OUT, f"batch_{b}.pkl")
        if os.path.exists(fn):
            log(f"batch {b} 已存在，跳過")
            continue
        t0 = time.time()
        torch.cuda.reset_peak_memory_stats()
        P = np.stack([folds[i]["train"] for i in fi_arr]).astype(np.float64)
        F = np.stack([feats_tr[i] for i in fi_arr])
        env = E.NumpyEnv(P, F, mode_arr, scale_k=SCALE_K)          # 環境在 CPU（見 engine.NumpyEnv 說明）
        agents = E.BatchedAgents(A, dev, seed=1000 + b)              # 網路、replay、Adam 在 GPU
        ep_r, ep_l = E.train_hybrid(agents, env, episodes=EPISODES, log=log, log_every=50)
        del P, F, env

        rows = []
        for fi in range(len(folds)):
            idx = np.where(fi_arr == fi)[0]
            Pt = np.repeat(folds[fi]["test"][None], len(idx), 0).astype(np.float64)
            Ft = np.repeat(feats_te[fi][None], len(idx), 0)
            H, acts, trades, L = E.evaluate_hybrid(agents, Pt, Ft, idx)
            for j, a in enumerate(idx):
                rows.append(dict(batch=b, agent=int(a), fold=fi, ticker=folds[fi]["ticker"], year=folds[fi]["year"],
                                 mode=E.MODE_NAMES[mode_arr[a]], seed_in_batch=int(combos[a][2]),
                                 history=H[j, :L[j]].astype(np.float64), actions=acts[j].astype(np.int8),
                                 trades=int(trades[j])))
        secs = time.time() - t0
        peak = torch.cuda.max_memory_allocated() / 2**20
        pickle.dump(dict(rows=rows, ep_reward=ep_r.astype(np.float32), ep_loss=ep_l.astype(np.float32),
                         fi=fi_arr, mode=mode_arr, secs=secs, peak_mib=peak, scale_k=SCALE_K), open(fn, "wb"))
        log(f"batch {b} 完成：{secs/60:.1f} 分，峰值 VRAM {peak:.0f} MiB")
        del agents
        torch.cuda.empty_cache()
    log("全部完成")


if __name__ == "__main__":
    main()
