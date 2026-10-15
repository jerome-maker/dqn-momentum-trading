# -*- coding: utf-8 -*-
"""驗證 2：在原論文的 0050 窗口（2021-06-22~2026-06-22，90/10 切分）上，
用 GPU 批次引擎重跑 raw / shaped 各 30 個種子，與手稿 Table 2（原程式 10 個種子）比較。"""
import json, os, sys, time, pickle
import numpy as np, pandas as pd, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import engine as E

cells = [''.join(c['source']) for c in json.load(open(os.path.join(HERE, '..', 'DQN_Momentum_Strategy.ipynb'), encoding='utf-8'))['cells'] if c['cell_type'] == 'code']
g = {'np': np, 'pd': pd}
exec(next(c for c in cells if 'def calculate_metrics' in c), g)
calc = g['calculate_metrics']

ser = pd.read_pickle(os.path.join(HERE, 'data', 'clean_close_panel.pkl'))['0050.TW']['2021-06-22':'2026-06-22'].reset_index(drop=True)
print('window rows', len(ser), 'min/max', round(ser.min(), 2), round(ser.max(), 2), '(手稿：1215 rows, 21.75-110.48)')
split = int(len(ser) * 0.90)
tr, te = ser.values[:split], ser.values[split:]

dev = torch.device('cuda')
SEEDS = 30
modes = np.array([E.RAW] * SEEDS + [E.SHAPED] * SEEDS)
A = len(modes)
Ftr = torch.tensor(E.market_features(tr), device=dev).unsqueeze(0).expand(A, -1, -1).contiguous()
Ptr = torch.tensor(tr, device=dev).unsqueeze(0).expand(A, -1).contiguous()
env = E.BatchedEnv(Ptr, Ftr, torch.tensor(modes, device=dev))
agents = E.BatchedAgents(A, dev, seed=20261015)
torch.cuda.reset_peak_memory_stats()
t0 = time.time()
epr, epl = E.train(agents, env, episodes=500, log=print, log_every=100)
torch.cuda.synchronize()
secs = time.time() - t0

Fte = torch.tensor(E.market_features(te), device=dev).unsqueeze(0).expand(A, -1, -1).contiguous()
Pte = torch.tensor(te, device=dev).unsqueeze(0).expand(A, -1).contiguous()
H, acts, trades, L = E.evaluate(agents, Pte, Fte, torch.arange(A, device=dev))
rows = []
for i in range(A):
    m = calc(H[i, :L[i]])
    m.update(mode=E.MODE_NAMES[modes[i]], trades=int(trades[i]))
    rows.append(m)
df = pd.DataFrame(rows)
pickle.dump({'df': df, 'ep_reward': epr, 'secs': secs}, open(os.path.join(HERE, 'results_validate_replication.pkl'), 'wb'))

orig = pd.DataFrame([{k: v for k, v in m.items() if k != 'portfolio_history'}
                     for m in pickle.load(open(os.path.join(HERE, '..', 'reward_comparison_checkpoint_pinned.pkl'), 'rb'))])
orig['trades0'] = orig['total_return'] == 0
df['trades0'] = df['trades'] == 0
from scipy.stats import mannwhitneyu
print(f"\n裝置 GPU({torch.cuda.get_device_name(0)})  agents={A}  耗時 {secs/60:.1f} 分  峰值VRAM {torch.cuda.max_memory_allocated()/2**20:.0f} MiB")
for mode in ['raw_return', 'momentum_shaped']:
    a = orig[orig.reward_mode == mode]; b = df[df['mode'] == mode]
    print(f"\n[{mode}]            原程式(n={len(a)})        新引擎(n={len(b)})     MWU p")
    for col in ['total_return', 'sharpe_ratio', 'max_drawdown']:
        p = mannwhitneyu(a[col], b[col]).pvalue
        print(f"  {col:<14} {a[col].mean():>9.4f}±{a[col].std():.4f}   {b[col].mean():>9.4f}±{b[col].std():.4f}   {p:.3f}")
    print(f"  不交易(崩潰)比例   {a.trades0.mean():.0%}                 {b.trades0.mean():.0%}")
