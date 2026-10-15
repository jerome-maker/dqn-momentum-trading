# -*- coding: utf-8 -*-
"""驗證 1：GPU 批次環境與 notebook 原始環境逐步一致。"""
import json, os, sys
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine as E

NB = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'DQN_Momentum_Strategy.ipynb')
cells = [''.join(c['source']) for c in json.load(open(NB, encoding='utf-8'))['cells'] if c['cell_type'] == 'code']
g = {'np': np, 'pd': pd}
exec(next(c for c in cells if 'class MomentumEnvironment' in c), g)
Orig = g['MomentumEnvironment']

panel = pd.read_pickle('data/clean_close_panel.pkl')
cases = {
    '0050_2021_2026': panel['0050.TW']['2021-06-22':'2026-06-22'],
    'EZA_2006_2010':  panel['EZA']['2006-01-01':'2010-12-31'],   # 含金融海嘯，測停損
    'GLD_2015_2020':  panel['GLD']['2015-01-01':'2020-12-31'],
}
modes = {'raw_return': E.RAW, 'momentum_shaped': E.SHAPED, 'differential_sharpe': E.DSR}
dev = torch.device('cuda')
rng = np.random.default_rng(0)
worst = {}
for cname, ser in cases.items():
    ser = ser.reset_index(drop=True)
    prices = ser.values.astype(np.float64)
    F = E.market_features(prices)
    for policy in ('uniform', 'trader'):
        T = len(prices)
        if policy == 'uniform':
            acts = rng.integers(0, 3, T)
        else:  # 高頻買賣，最大化交易成本以觸發停損
            acts = np.where(np.arange(T) % 2 == 0, 2, 0)
        for mname, mcode in modes.items():
            o = Orig(ser, reward_mode=mname)
            so = o.reset()
            P = torch.tensor(prices[None], device=dev)
            Ft = torch.tensor(F[None], device=dev)
            b = E.BatchedEnv(P, Ft, torch.tensor([mcode], device=dev))
            sb = b.reset()
            ds = np.abs(so - sb.cpu().numpy()[0]).max()
            dr = dv = 0.0
            t = 0
            n = 0
            while True:
                a = int(acts[t]); t += 1
                so, ro, do, info = o.step(a)
                sb, rb, db, vb = b.step(torch.tensor([a], device=dev))
                b.alive = b.alive & ~db
                ds = max(ds, np.abs(so - sb.cpu().numpy()[0]).max())
                dr = max(dr, abs(ro - rb.item()))
                dv = max(dv, abs(info['portfolio_value'] - vb.item()))
                assert bool(do) == bool(db.item()), (cname, mname, t)
                n += 1
                if do:
                    break
            worst[(cname, policy, mname)] = (n, ds, dr, dv, info['portfolio_value'])
print(f"{'case':<16}{'policy':<9}{'mode':<21}{'steps':>6}{'max|Δstate|':>13}{'max|Δreward|':>14}{'max|Δvalue|':>13}{'final value':>14}")
for (c, p, m), (n, ds, dr, dv, fv) in worst.items():
    print(f"{c:<16}{p:<9}{m:<21}{n:>6}{ds:>13.2e}{dr:>14.2e}{dv:>13.2e}{fv:>14.1f}")
