# -*- coding: utf-8 -*-
"""GPU 批次 DQN 引擎：讓上千個「彼此獨立」的 DQN agent 在 GPU 上同步訓練。

與 notebook 中 MomentumEnvironment / MomentumDQNAgent 的對應關係（逐項複製）：
- 環境：狀態向量、交易規則（賣出全部、買入 50% 現金、交易成本 0.1%）、起始步 30、
  50% 停損、各種獎勵函數的運算順序都與原始程式相同（validate_env.py 逐步比對）。
- 網路：10-128-128-3，BatchNorm1d + ReLU + Dropout(0.2)，PyTorch 預設初始化分布。
- 學習：ε-greedy（每次 replay 後 ε×0.998，下限 0.01）、replay buffer 10000、
  minibatch 32（不放回抽樣）、Adam lr=1e-3、γ=0.95、每個 agent 各自梯度範數截斷 1.0、
  每 episode 後 replay 10 次、每 50 episode 硬更新 target network。
- 原始實作的一個特性也照樣複製：target network 從未切到 eval 模式，
  所以計算目標 Q 值時 dropout 與 BatchNorm 批次統計仍在作用。
- 唯一不同：亂數來源（torch.Generator 取代 numpy/random/torch 全域亂數），
  因此結果與原始程式是「統計上等價」而非逐位元相同。

每個 agent 的參數、Adam 狀態、replay buffer、梯度截斷都獨立；
批次化只是讓它們共用同一次 kernel 呼叫。
"""
import math
import numpy as np
import pandas as pd
import torch

KAPPA = 0.001
INIT = 100000.0
START = 30          # lookback_period(20) + 10
GAMMA = 0.95

# 獎勵函數代號
RAW, SHAPED, RAW_SCALED, MOM_ONLY, DD_ONLY, PBRS, DSR = range(7)
MODE_NAMES = ["raw_return", "momentum_shaped", "raw_scaled", "momentum_only",
              "drawdown_only", "potential_based", "differential_sharpe"]


# --------------------------------------------------------------------------- 市場特徵
def market_features(prices):
    """逐日複製 MomentumEnvironment 的指標計算（含 pandas ewm），回傳 [T,7] float64：
    [log(p/100), M5, M10, M20, RSI14, MACD, vol20]"""
    prices = np.asarray(prices, dtype=np.float64)
    T = len(prices)
    F = np.zeros((T, 7))
    for t in range(T):
        p = prices[t]

        def mom(n):
            if t < n:
                return 0
            pp = prices[t - n]
            return (p - pp) / pp if pp > 0 else 0

        if t < 14:
            rsi = 0.5
        else:
            w = prices[t - 14:t]
            if len(w) < 2:
                rsi = 0.5
            else:
                d = np.diff(w)
                ag = np.mean(np.where(d > 0, d, 0))
                al = np.mean(np.where(d < 0, -d, 0))
                if al == 0:
                    rsi = 1.0 if ag > 0 else 0.5
                else:
                    rs = ag / al
                    rsi = np.clip(1 - (1 / (1 + rs)), 0, 1)
        if t < 26:
            macd = 0
        else:
            w = prices[t - 26:t]
            e12 = pd.Series(w).ewm(span=12).mean().values[-1]
            e26 = pd.Series(w).ewm(span=26).mean().values[-1]
            m = (e12 - e26) / e26 if e26 > 0 else 0
            macd = np.clip(m / 0.1, -1, 1)
        if t < 20:
            vol = 0
        else:
            w = prices[t - 20:t]
            vol = np.std(np.diff(w) / w[:-1]) * np.sqrt(252)
        F[t] = [np.log(p / 100), mom(5), mom(10), mom(20), rsi, macd, vol]
    return F


# --------------------------------------------------------------------------- 批次環境
class BatchedEnv:
    """A 個 agent 同步推進的交易環境（GPU, float64）。P/F 可每個 agent 不同，但長度相同。"""

    def __init__(self, P, F, modes, scale_k=1.0, pbrs_coef=50.0, dsr_eta=0.01):
        self.P = P            # [A,T] float64
        self.F = F            # [A,T,7] float64
        self.modes = modes    # [A] long
        self.A, self.T = P.shape
        self.dev = P.device
        self.scale_k = scale_k
        self.pbrs_coef = pbrs_coef
        self.dsr_eta = dsr_eta
        self.ar = torch.arange(self.A, device=self.dev)

    def reset(self):
        z = torch.zeros(self.A, dtype=torch.float64, device=self.dev)
        self.cs = START
        self.cash = z + INIT
        self.pos = z.clone()
        self.maxv = z + INIT
        self.dA = z.clone()
        self.dB = z.clone()
        self.alive = torch.ones(self.A, dtype=torch.bool, device=self.dev)
        self.trades = torch.zeros(self.A, dtype=torch.long, device=self.dev)
        return self.state()

    def _port_feats(self, cs):
        price = self.P[:, cs]
        total = self.cash + self.pos * price
        rr = (total - INIT) / INIT
        posr = torch.where(total > 0, (self.pos * price) / total, torch.zeros_like(total))
        cashr = torch.where(total > 0, self.cash / total, torch.ones_like(total))
        return posr, cashr, rr

    def state(self):
        f = self.F[:, self.cs]
        posr, cashr, rr = self._port_feats(self.cs)
        s = torch.stack([f[:, 0], f[:, 1], f[:, 2], f[:, 3], f[:, 4], f[:, 5],
                         posr, cashr, rr.clamp(-1, 1), f[:, 6]], dim=1)
        return s.float()

    def step(self, action):
        """action: [A] long。已結束(alive=False)的 agent 狀態不變。"""
        al = self.alive
        cs = self.cs
        price = self.P[:, cs]
        prev = self.cash + self.pos * price
        posr_before, _, _ = self._port_feats(cs)
        m10_before = self.F[:, cs, 2]

        sell = al & (action == 0) & (self.pos > 0)
        sv = self.pos * price
        tc_s = sv * KAPPA
        cash = torch.where(sell, self.cash + (sv - tc_s), self.cash)
        pos = torch.where(sell, torch.zeros_like(self.pos), self.pos)

        buy = al & (action == 2) & (pos == 0) & (cash > 0)
        inv = cash * 0.5
        tc_b = inv * KAPPA
        inv2 = inv - tc_b
        buy = buy & (inv2 > 0)
        pos = torch.where(buy, inv2 / price, pos)
        cash = torch.where(buy, cash - (inv2 + tc_b), cash)
        self.trades += (sell | buy).long()
        self.cash, self.pos = cash, pos

        self.cs = cs + 1
        ncs = self.cs
        newp = self.P[:, ncs]
        new = self.cash + self.pos * newp
        ret = torch.where(prev > 0, (new - prev) / prev, torch.zeros_like(prev))
        self.maxv = torch.where(al, torch.maximum(self.maxv, new), self.maxv)

        m = self.F[:, ncs, 2]          # 原始程式在 current_step+1 之後才算動能
        absm = m.abs()
        base = ret * 100
        # 動能塑形項（與原始 if 分支同序）
        bonus = torch.zeros_like(base)
        bonus = torch.where(action == 2, torch.where(m > 0.02, m * 50, torch.full_like(m, -2.0)), bonus)
        bonus = torch.where((action == 0) & (m < -0.02), absm * 50, bonus)
        hold_pos = (action == 1) & (self.pos > 0)
        bonus = torch.where(hold_pos, torch.where(m > 0, absm * 20, -(absm * 10)), bonus)
        dd = (self.maxv - new) / self.maxv
        ddpen = torch.where(dd > 0.1, dd * 20, torch.zeros_like(dd))

        shaped = (base + bonus) - ddpen
        mom_only = base + bonus
        dd_only = base - ddpen
        raw_scaled = base * self.scale_k

        done = (torch.full_like(al, ncs >= self.T - 1)) | (new <= INIT * 0.5)
        posr_after, _, _ = self._port_feats(ncs)
        phi_s = self.pbrs_coef * posr_before * m10_before
        phi_s2 = torch.where(done, torch.zeros_like(phi_s), self.pbrs_coef * posr_after * m)
        pbrs = base + (GAMMA * phi_s2 - phi_s)

        R = ret
        dA = R - self.dA
        dB = R ** 2 - self.dB
        denom = self.dB - self.dA ** 2
        safe = torch.where(denom > 1e-8, denom, torch.ones_like(denom))
        dsr = torch.where(denom > 1e-8, (self.dB * dA - 0.5 * self.dA * dB) / safe ** 1.5,
                          torch.zeros_like(denom))
        self.dA = torch.where(al, self.dA + self.dsr_eta * dA, self.dA)
        self.dB = torch.where(al, self.dB + self.dsr_eta * dB, self.dB)

        table = torch.stack([base, shaped, raw_scaled, mom_only, dd_only, pbrs, dsr * 100], dim=1)
        reward = table[self.ar, self.modes]
        reward = torch.where(al, reward, torch.zeros_like(reward))
        done = done & al
        s2 = self.state()
        value = torch.where(al, new, prev)
        return s2, reward, done, value


# --------------------------------------------------------------------------- 批次網路
class BatchedQNet:
    def __init__(self, A, dev, gen, hidden=128, sd=10, ad=3):
        def lin(fi, fo):
            b = 1.0 / math.sqrt(fi)
            W = ((torch.rand(A, fi, fo, device=dev, generator=gen) * 2 - 1) * b).requires_grad_()
            bias = ((torch.rand(A, fo, device=dev, generator=gen) * 2 - 1) * b).requires_grad_()
            return W, bias
        self.W1, self.b1 = lin(sd, hidden)
        self.W2, self.b2 = lin(hidden, hidden)
        self.W3, self.b3 = lin(hidden, ad)
        one = lambda: torch.ones(A, hidden, device=dev).requires_grad_()
        zero = lambda: torch.zeros(A, hidden, device=dev).requires_grad_()
        self.g1, self.be1, self.g2, self.be2 = one(), zero(), one(), zero()
        self.rm1 = torch.zeros(A, hidden, device=dev)
        self.rv1 = torch.ones(A, hidden, device=dev)
        self.rm2 = torch.zeros(A, hidden, device=dev)
        self.rv2 = torch.ones(A, hidden, device=dev)

    def params(self):
        return [self.W1, self.b1, self.g1, self.be1, self.W2, self.b2, self.g2, self.be2, self.W3, self.b3]

    def buffers(self):
        return [self.rm1, self.rv1, self.rm2, self.rv2]

    def _bn(self, h, g, be, rm, rv, train, update):
        if train:
            mean = h.mean(1, keepdim=True)
            var = h.var(1, unbiased=False, keepdim=True)
            if update:
                with torch.no_grad():
                    rm.mul_(0.9).add_(0.1 * mean.squeeze(1))
                    rv.mul_(0.9).add_(0.1 * h.var(1, unbiased=True))
            hn = (h - mean) / torch.sqrt(var + 1e-5)
        else:
            hn = (h - rm.unsqueeze(1)) / torch.sqrt(rv.unsqueeze(1) + 1e-5)
        return hn * g.unsqueeze(1) + be.unsqueeze(1)

    def forward(self, x, train, gen=None, update=False):
        h = torch.baddbmm(self.b1.unsqueeze(1), x, self.W1)
        h = torch.relu(self._bn(h, self.g1, self.be1, self.rm1, self.rv1, train, update))
        if train:
            h = h * (torch.rand(h.shape, device=h.device, generator=gen) < 0.8) / 0.8
        h = torch.baddbmm(self.b2.unsqueeze(1), h, self.W2)
        h = torch.relu(self._bn(h, self.g2, self.be2, self.rm2, self.rv2, train, update))
        if train:
            h = h * (torch.rand(h.shape, device=h.device, generator=gen) < 0.8) / 0.8
        return torch.baddbmm(self.b3.unsqueeze(1), h, self.W3)

    def copy_from(self, other):
        with torch.no_grad():
            for a, b in zip(self.params() + self.buffers(), other.params() + other.buffers()):
                a.copy_(b)


class BatchedAgents:
    def __init__(self, A, dev, seed, mem=10000, batch=32):
        self.A, self.dev, self.mem, self.batch = A, dev, mem, batch
        self.gen = torch.Generator(device=dev)
        self.gen.manual_seed(seed)
        self.q = BatchedQNet(A, dev, self.gen)
        self.tq = BatchedQNet(A, dev, self.gen)
        self.tq.copy_from(self.q)
        self.opt = torch.optim.Adam(self.q.params(), lr=1e-3)
        self.eps, self.eps_decay, self.eps_min = 1.0, 0.998, 0.01
        self.S = torch.zeros(A, mem, 10, device=dev)
        self.S2 = torch.zeros(A, mem, 10, device=dev)
        self.Ra = torch.zeros(A, mem, device=dev)
        self.Ac = torch.zeros(A, mem, dtype=torch.long, device=dev)
        self.D = torch.zeros(A, mem, dtype=torch.bool, device=dev)
        self.ptr = torch.zeros(A, dtype=torch.long, device=dev)
        self.size = torch.zeros(A, dtype=torch.long, device=dev)
        self.ar = torch.arange(A, device=dev)
        self.loss_hist = []

    @torch.no_grad()
    def act(self, s, explore=True):
        q = self.q.forward(s.unsqueeze(1), train=False).squeeze(1)
        greedy = q.argmax(1)
        if not explore:
            return greedy
        r = torch.rand(self.A, device=self.dev, generator=self.gen)
        ra = torch.randint(0, 3, (self.A,), device=self.dev, generator=self.gen)
        return torch.where(r < self.eps, ra, greedy)

    @torch.no_grad()
    def remember(self, s, a, r, s2, d, mask):
        """不用布林索引（避免每步 GPU 同步）：已結束的 agent 寫回原值且指標不前進。"""
        i = self.ptr
        ar = self.ar
        m1 = mask.unsqueeze(1)
        self.S[ar, i] = torch.where(m1, s, self.S[ar, i])
        self.S2[ar, i] = torch.where(m1, s2, self.S2[ar, i])
        self.Ra[ar, i] = torch.where(mask, r.float(), self.Ra[ar, i])
        self.Ac[ar, i] = torch.where(mask, a, self.Ac[ar, i])
        self.D[ar, i] = torch.where(mask, d, self.D[ar, i])
        self.ptr = torch.where(mask, (i + 1) % self.mem, i)
        self.size = torch.where(mask, torch.clamp(self.size + 1, max=self.mem), self.size)

    def replay(self):
        if int(self.size.min()) < self.batch:
            return
        # 不放回抽樣：先抽 64 個可重複索引，依抽出順序保留前 32 個「第一次出現」者。
        # 依序剔除重複 = 不放回抽樣，分布與 random.sample 相同；buffer ≥ 1169 筆時
        # 64 次抽取少於 32 個相異值的機率可忽略。
        c = (torch.rand(self.A, 64, device=self.dev, generator=self.gen) * self.size.unsqueeze(1)).long()
        srt, perm = c.sort(1)
        dup_sorted = torch.cat([torch.zeros(self.A, 1, dtype=torch.bool, device=self.dev),
                                srt[:, 1:] == srt[:, :-1]], 1)
        dup = torch.zeros_like(dup_sorted).scatter(1, perm, dup_sorted)
        order = torch.arange(64, device=self.dev).unsqueeze(0) + dup.long() * 1000
        keep = order.topk(self.batch, dim=1, largest=False).indices
        idx = c.gather(1, keep)
        g = lambda X: X[self.ar.unsqueeze(1), idx]
        s, a, r, s2, d = g(self.S), g(self.Ac), g(self.Ra), g(self.S2), g(self.D)

        q = self.q.forward(s, train=True, gen=self.gen, update=True)
        cur = q.gather(2, a.unsqueeze(2)).squeeze(2)
        with torch.no_grad():
            nq = self.tq.forward(s2, train=True, gen=self.gen, update=False).max(2)[0]
            tgt = r + GAMMA * nq * (~d)
        loss_per = ((cur - tgt) ** 2).mean(1)                        # 每個 agent 各自 MSE
        self.opt.zero_grad(set_to_none=True)
        loss_per.sum().backward()
        with torch.no_grad():
            sq = torch.zeros(self.A, device=self.dev)
            for p in self.q.params():
                sq += p.grad.pow(2).reshape(self.A, -1).sum(1)
            coef = (1.0 / (sq.sqrt() + 1e-6)).clamp(max=1.0)          # clip_grad_norm_(…, 1.0)
            for p in self.q.params():
                p.grad.mul_(coef.view(-1, *([1] * (p.grad.dim() - 1))))
        self.opt.step()
        self.loss_hist.append(loss_per.detach())
        if self.eps > self.eps_min:
            self.eps *= self.eps_decay


# --------------------------------------------------------------------------- 訓練 / 評估
def train(agents, env, episodes=500, log=None, log_every=50):
    A = agents.A
    ep_reward = torch.zeros(episodes, A, device=env.dev)
    ep_loss = torch.zeros(episodes, A, device=env.dev)
    for ep in range(episodes):
        s = env.reset()
        tot = torch.zeros(A, dtype=torch.float64, device=env.dev)
        while True:
            a = agents.act(s, explore=True)
            alive = env.alive.clone()
            s2, r, d, _ = env.step(a)
            agents.remember(s, a, r, s2, d, alive)
            tot += r
            env.alive = alive & ~d
            s = s2
            if env.cs >= env.T - 1 or (env.cs % 100 == 0 and not bool(env.alive.any())):
                break
        n0 = len(agents.loss_hist)
        for _ in range(10):
            agents.replay()
        new = agents.loss_hist[n0:]
        if new:
            ep_loss[ep] = torch.stack(new).mean(0)
        ep_reward[ep] = tot.float()
        if (ep + 1) % 50 == 0:
            agents.tq.copy_from(agents.q)
        if log and (ep + 1) % log_every == 0:
            log(f"  episode {ep+1}/{episodes}  eps={agents.eps:.3f}  "
                f"mean ep reward={ep_reward[ep].mean().item():.2f}")
    return ep_reward.cpu().numpy(), ep_loss.cpu().numpy()


@torch.no_grad()
def evaluate(agents, P, F, idx):
    """對 agents 的子集合 idx 在共用的測試序列上做貪婪回測（ε=0、BN 用 running stats）。
    回傳：投資組合歷史 H[A,L]、動作 [A,L-1]、交易次數 [A]、每個 agent 的有效長度 [A]
    （觸發 50% 停損的 agent 歷史在停損那步截止，與原始 backtest 相同）。"""
    A = len(idx)
    sub = _SubNet(agents.q, idx)
    env = BatchedEnv(P, F, torch.zeros(A, dtype=torch.long, device=P.device))
    s = env.reset()
    hist = [torch.full((A,), INIT, dtype=torch.float64, device=P.device)]
    acts = []
    length = torch.ones(A, dtype=torch.long, device=P.device)
    while True:
        a = sub.act(s)
        alive = env.alive.clone()
        s, _, d, v = env.step(a)
        hist.append(v)
        acts.append(a)
        length += alive.long()
        env.alive = alive & ~d
        if env.cs >= env.T - 1 or not bool(env.alive.any()):
            break
    H = torch.stack(hist, 1).cpu().numpy()
    return H, torch.stack(acts, 1).cpu().numpy(), env.trades.cpu().numpy(), length.cpu().numpy()


class _SubNet:
    def __init__(self, q, idx):
        self.q, self.idx = q, idx

    def act(self, s):
        q = self.q
        i = self.idx
        x = s.unsqueeze(1)
        h = torch.baddbmm(q.b1[i].unsqueeze(1), x, q.W1[i])
        h = torch.relu((h - q.rm1[i].unsqueeze(1)) / torch.sqrt(q.rv1[i].unsqueeze(1) + 1e-5)
                       * q.g1[i].unsqueeze(1) + q.be1[i].unsqueeze(1))
        h = torch.baddbmm(q.b2[i].unsqueeze(1), h, q.W2[i])
        h = torch.relu((h - q.rm2[i].unsqueeze(1)) / torch.sqrt(q.rv2[i].unsqueeze(1) + 1e-5)
                       * q.g2[i].unsqueeze(1) + q.be2[i].unsqueeze(1))
        return torch.baddbmm(q.b3[i].unsqueeze(1), h, q.W3[i]).squeeze(1).argmax(1)


# =========================================================================== CPU 環境
class NumpyEnv:
    """與 BatchedEnv 逐行對應的 numpy 版本（float64、CPU）。
    環境更新是約 70 個對 A 個元素的小運算，在 GPU 上每個都要付 kernel 啟動成本，
    在 CPU 上反而快一個數量級；神經網路與 replay 仍在 GPU。"""

    def __init__(self, P, F, modes, scale_k=1.0, pbrs_coef=50.0, dsr_eta=0.01):
        self.P, self.F, self.modes = np.asarray(P, np.float64), np.asarray(F, np.float64), np.asarray(modes)
        self.A, self.T = self.P.shape
        self.scale_k, self.pbrs_coef, self.dsr_eta = scale_k, pbrs_coef, dsr_eta
        self.ar = np.arange(self.A)

    def reset(self):
        A = self.A
        self.cs = START
        self.cash = np.full(A, INIT)
        self.pos = np.zeros(A)
        self.maxv = np.full(A, INIT)
        self.dA = np.zeros(A)
        self.dB = np.zeros(A)
        self.alive = np.ones(A, bool)
        self.trades = np.zeros(A, np.int64)
        return self.state()

    def _port_feats(self, cs):
        price = self.P[:, cs]
        total = self.cash + self.pos * price
        rr = (total - INIT) / INIT
        with np.errstate(divide='ignore', invalid='ignore'):
            posr = np.where(total > 0, (self.pos * price) / total, 0.0)
            cashr = np.where(total > 0, self.cash / total, 1.0)
        return posr, cashr, rr

    def state(self):
        f = self.F[:, self.cs]
        posr, cashr, rr = self._port_feats(self.cs)
        s = np.stack([f[:, 0], f[:, 1], f[:, 2], f[:, 3], f[:, 4], f[:, 5],
                      posr, cashr, np.clip(rr, -1, 1), f[:, 6]], axis=1)
        return s.astype(np.float32)

    def step(self, action):
        al = self.alive
        cs = self.cs
        price = self.P[:, cs]
        prev = self.cash + self.pos * price
        posr_before, _, _ = self._port_feats(cs)
        m10_before = self.F[:, cs, 2]

        sell = al & (action == 0) & (self.pos > 0)
        sv = self.pos * price
        tc_s = sv * KAPPA
        cash = np.where(sell, self.cash + (sv - tc_s), self.cash)
        pos = np.where(sell, 0.0, self.pos)

        buy = al & (action == 2) & (pos == 0) & (cash > 0)
        inv = cash * 0.5
        tc_b = inv * KAPPA
        inv2 = inv - tc_b
        buy = buy & (inv2 > 0)
        pos = np.where(buy, inv2 / price, pos)
        cash = np.where(buy, cash - (inv2 + tc_b), cash)
        self.trades += (sell | buy)
        self.cash, self.pos = cash, pos

        self.cs = cs + 1
        ncs = self.cs
        newp = self.P[:, ncs]
        new = self.cash + self.pos * newp
        with np.errstate(divide='ignore', invalid='ignore'):
            ret = np.where(prev > 0, (new - prev) / prev, 0.0)
        self.maxv = np.where(al, np.maximum(self.maxv, new), self.maxv)

        m = self.F[:, ncs, 2]
        absm = np.abs(m)
        base = ret * 100
        bonus = np.zeros(self.A)
        bonus = np.where(action == 2, np.where(m > 0.02, m * 50, -2.0), bonus)
        bonus = np.where((action == 0) & (m < -0.02), absm * 50, bonus)
        hold_pos = (action == 1) & (self.pos > 0)
        bonus = np.where(hold_pos, np.where(m > 0, absm * 20, -(absm * 10)), bonus)
        dd = (self.maxv - new) / self.maxv
        ddpen = np.where(dd > 0.1, dd * 20, 0.0)

        done = (ncs >= self.T - 1) | (new <= INIT * 0.5)
        posr_after, _, _ = self._port_feats(ncs)
        phi_s = self.pbrs_coef * posr_before * m10_before
        phi_s2 = np.where(done, 0.0, self.pbrs_coef * posr_after * m)

        R = ret
        dA = R - self.dA
        dB = R ** 2 - self.dB
        denom = self.dB - self.dA ** 2
        safe = np.where(denom > 1e-8, denom, 1.0)
        dsr = np.where(denom > 1e-8, (self.dB * dA - 0.5 * self.dA * dB) / safe ** 1.5, 0.0)
        self.dA = np.where(al, self.dA + self.dsr_eta * dA, self.dA)
        self.dB = np.where(al, self.dB + self.dsr_eta * dB, self.dB)

        table = np.stack([base, (base + bonus) - ddpen, base * self.scale_k, base + bonus,
                          base - ddpen, base + (GAMMA * phi_s2 - phi_s), dsr * 100], axis=1)
        reward = np.where(al, table[self.ar, self.modes], 0.0)
        done = done & al
        s2 = self.state()
        value = np.where(al, new, prev)
        return s2, reward, done, value


def _bulk_remember(agents, S, Acts, R, S2, D, valid):
    """把一個 episode 的轉移一次寫入 GPU replay buffer（環狀，每個 agent 自己的指標）。
    S,S2:[A,L,10] f32  Acts:[A,L] i64  R:[A,L] f32  D,valid:[A,L] bool（numpy）"""
    dev = agents.dev
    valid_t = torch.from_numpy(valid).to(dev)
    n = valid_t.sum(1)                                   # 每個 agent 本 episode 的轉移數
    k = torch.cumsum(valid_t.long(), 1) - 1              # 有效轉移的序號
    pos = (agents.ptr.unsqueeze(1) + k) % agents.mem
    a_idx = agents.ar.unsqueeze(1).expand_as(pos)[valid_t]
    p_idx = pos[valid_t]
    agents.S[a_idx, p_idx] = torch.from_numpy(S).to(dev)[valid_t]
    agents.S2[a_idx, p_idx] = torch.from_numpy(S2).to(dev)[valid_t]
    agents.Ra[a_idx, p_idx] = torch.from_numpy(R).to(dev)[valid_t]
    agents.Ac[a_idx, p_idx] = torch.from_numpy(Acts).to(dev)[valid_t]
    agents.D[a_idx, p_idx] = torch.from_numpy(D).to(dev)[valid_t]
    agents.ptr = (agents.ptr + n) % agents.mem
    agents.size = torch.clamp(agents.size + n, max=agents.mem)


def train_hybrid(agents, env, episodes=500, log=None, log_every=50):
    """環境在 CPU（NumpyEnv），網路/replay 在 GPU。
    與原始程式等價的關鍵：同一 episode 內網路權重不變（replay 只在 episode 結束後），
    所以把整個 episode 的轉移收集完再一次寫入 buffer，與逐步寫入結果相同。"""
    A, dev = agents.A, agents.dev
    L = env.T - 1 - START
    ep_reward = np.zeros((episodes, A), np.float32)
    ep_loss = np.zeros((episodes, A), np.float32)
    S = np.zeros((A, L, 10), np.float32); S2 = np.zeros_like(S)
    Acts = np.zeros((A, L), np.int64); R = np.zeros((A, L), np.float32)
    D = np.zeros((A, L), bool); V = np.zeros((A, L), bool)
    for ep in range(episodes):
        s = env.reset()
        V[:] = False
        tot = np.zeros(A)
        for t in range(L):
            a = agents.act(torch.from_numpy(s).to(dev), explore=True).cpu().numpy()
            alive = env.alive.copy()
            s2, r, d, _ = env.step(a)
            S[:, t], Acts[:, t], R[:, t], S2[:, t], D[:, t], V[:, t] = s, a, r, s2, d, alive
            tot += r
            env.alive = alive & ~d
            s = s2
        _bulk_remember(agents, S, Acts, R, S2, D, V)
        n0 = len(agents.loss_hist)
        for _ in range(10):
            agents.replay()
        new = agents.loss_hist[n0:]
        if new:
            ep_loss[ep] = torch.stack(new).mean(0).cpu().numpy()
        agents.loss_hist = []
        ep_reward[ep] = tot
        if (ep + 1) % 50 == 0:
            agents.tq.copy_from(agents.q)
        if log and (ep + 1) % log_every == 0:
            log(f"  episode {ep+1}/{episodes}  eps={agents.eps:.3f}  mean ep reward={ep_reward[ep].mean():.2f}")
    return ep_reward, ep_loss


@torch.no_grad()
def evaluate_hybrid(agents, P, F, idx):
    """貪婪回測（CPU 環境 + GPU 網路）。P,F 為 numpy；idx 為 agent 編號 numpy 陣列。"""
    A = len(idx)
    sub = _SubNet(agents.q, torch.as_tensor(idx, device=agents.dev))
    env = NumpyEnv(P, F, np.zeros(A, np.int64))
    s = env.reset()
    hist = [np.full(A, INIT)]
    acts = []
    length = np.ones(A, np.int64)
    while True:
        a = sub.act(torch.from_numpy(s).to(agents.dev)).cpu().numpy()
        alive = env.alive.copy()
        s, _, d, v = env.step(a)
        hist.append(v); acts.append(a)
        length += alive
        env.alive = alive & ~d
        if env.cs >= env.T - 1 or not env.alive.any():
            break
    return np.stack(hist, 1), np.stack(acts, 1), env.trades.copy(), length
