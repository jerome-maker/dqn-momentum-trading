# -*- coding: utf-8 -*-
"""以隨機策略在全部 87 個訓練窗口上跑一個 episode，比較塑形獎勵與原始獎勵的逐步標準差，
決定尺度對照組 raw_scaled 的倍數 k（在 CPU 上跑，不佔用訓練中的 GPU）。"""
import numpy as np, torch
import engine as E, folds as Fd
torch.manual_seed(0)
fl = Fd.make_folds()
dev = torch.device('cpu')
P = torch.tensor(np.stack([f['train'] for f in fl]), dtype=torch.float64)
F = torch.tensor(np.stack([E.market_features(f['train']) for f in fl]))
res = {}
for mode in (E.RAW, E.SHAPED, E.PBRS):
    env = E.BatchedEnv(P, F, torch.full((len(fl),), mode))
    env.reset(); rs = []
    while True:
        a = torch.randint(0, 3, (len(fl),))
        alive = env.alive.clone()
        _, r, d, _ = env.step(a)
        rs.append(torch.where(alive, r, torch.full_like(r, float('nan'))))
        env.alive = alive & ~d
        if env.cs >= env.T - 1:
            break
    R = torch.stack(rs, 1).numpy()
    res[mode] = np.nanstd(R, axis=1)
ratio = res[E.SHAPED] / res[E.RAW]
print(f"per-step reward std, random policy, 87 windows")
print(f"  raw    median {np.median(res[E.RAW]):.3f}")
print(f"  shaped median {np.median(res[E.SHAPED]):.3f}")
print(f"  pbrs   median {np.median(res[E.PBRS]):.3f}")
print(f"  shaped/raw ratio: median {np.median(ratio):.2f}  IQR {np.percentile(ratio,25):.2f}-{np.percentile(ratio,75):.2f}")
