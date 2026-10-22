"""迭代局部搜索 ILS：1-bit 中继的 model-free 训练器（INTEGRATION T；方案 §4.7 的替代 / 对照，RQ6）。

纯算法，只依赖"测一次"回调（与 dspsa.py 同一分层；system 封装见 train.run_ils）：
  measure(x) → (readings (B,K), flags (B,))：一次带噪测量（= 全部训练样本过一遍，计 1 次）；
  score(readings, flags) → J（越小越好，不计次数；train.measurable_objective 的判对优先目标）。

算法（= _scratch/trainers/alt_optimizers.py 的 ao_ils_acc；逐位一致见 tests/test_trainers_t.py）：
  1. 随机初始码本，测 1 次；
  2. 首次改进单比特爬山：固定随机顺序循环翻 1 位、各测 1 次，J 严格变小即接受；连续 p 次未接受 = 局部最优；
  3. 外环：从当前 x* 随机翻 k ∈ [kick] 位 → 测 1 次 → 爬山 → 新局部最优不差于 x*（J ≤ J*）即接受（中性漂移）；
  4. 预算用尽时返回 pocket（测到的单次 J 最小的码本；严格改进才替换）。
extras=True（= ao_ils4_k0_drift）：码本按"读数平均后再算 J"维护（候选池，同一码本多次测量取平均读数）；
  新局部最优严格更差时重测 x* 一次（x* 至多 max_remeasure 次测量，抑制赢家诅咒）；预留最后 race_m·race_k 次测量做终选赛：
  池中平均 J 最好的 race_m 个码本各补测到 race_k+1 次，剩余测量轮流补测当前最好的两个，返回平均 J 最好者。
"""
from __future__ import annotations

import math

import numpy as np


class _Stop(Exception):
    """预算用尽（内部用）。"""


class _Inc:
    """一个码本的重复测量（extras）：读数求和、标记取或，J = score(平均读数, 标记)。"""

    __slots__ = ("x", "ysum", "fl", "n", "f")

    def __init__(self, x, y, fl, f1):
        self.x = np.array(x, dtype=np.int64, copy=True)
        self.ysum = np.array(y, dtype=float, copy=True)
        self.fl = np.array(fl, dtype=bool, copy=True)
        self.n = 1
        self.f = float(f1)


def ils_bits(measure, score, p: int, n_meas: int, rng: np.random.Generator, *, kick=(2, 6), extras: bool = False,
             max_remeasure: int = 8, race=(4, 3), callback=None) -> dict:
    """ILS（1-bit 比特翻转邻域，见模块 docstring）。测量总数严格 ≤ n_meas。

    callback(n_used, x)：pocket（extras 时为池中平均 J 最好者）改变时调用（学习曲线 / bench report）；extras 结束时
    对终选结果再调用一次。返回 {"params", "J", "n_measurements", "n_ils_iter", "n_accept", "n_star_remeasure"}。
    """
    p, n_meas = int(p), int(n_meas)
    k_lo, k_hi = int(kick[0]), int(kick[1])
    st = {"n": 0, "iter": 0, "acc": 0, "rem": 0}
    two = np.full(p, 2, dtype=np.int64)

    def cb(x):
        if callback is not None:
            callback(st["n"], x)

    def kicked(x):
        k = int(rng.integers(k_lo, k_hi + 1))
        z = x.copy()
        z[rng.choice(p, size=min(k, p), replace=False)] ^= 1
        return z

    if not extras:
        best = {"x": None, "f": math.inf}

        def m1(x):
            if st["n"] >= n_meas:
                raise _Stop
            y, fl = measure(x)
            st["n"] += 1
            return score(y, fl)

        def offer(x, f):
            if f < best["f"]:
                best["x"], best["f"] = np.array(x, dtype=np.int64, copy=True), float(f)
                cb(best["x"])

        def local_search(x, f):
            order = rng.permutation(p)
            fails, j = 0, 0
            while fails < p:
                i = order[j % p]
                j += 1
                z = x.copy()
                z[i] ^= 1
                fz = m1(z)
                if fz < f:
                    x, f, fails = z, fz, 0
                    st["acc"] += 1
                    offer(x, f)
                else:
                    fails += 1
            return x, f

        try:
            x = rng.integers(0, two).astype(np.int64)
            f = m1(x)
            offer(x, f)
            xs, fs = local_search(x, f)
            while True:
                z = kicked(xs)
                fz = m1(z)
                offer(z, fz)
                z, fz = local_search(z, fz)
                st["iter"] += 1
                if fz <= fs:
                    xs, fs = z, fz
        except _Stop:
            pass
        if best["x"] is None:
            x = rng.integers(0, two).astype(np.int64)
            cb(x)
            return {"params": x, "J": None, "n_measurements": st["n"], "n_ils_iter": st["iter"],
                    "n_accept": st["acc"], "n_star_remeasure": 0}
        return {"params": best["x"].copy(), "J": best["f"], "n_measurements": st["n"], "n_ils_iter": st["iter"],
                "n_accept": st["acc"], "n_star_remeasure": 0}

    # ---------------------------------------------------------------- extras（ao_ils4_k0_drift）
    race_m, race_k = int(race[0]), int(race[1])
    reserve = race_m * race_k if race_m > 1 else 0
    pool: dict = {}                       # bytes → _Inc（插入顺序决定平局时的先后）
    pk = {"x": None, "f": None}

    def remaining():
        return n_meas - st["n"]

    def left():
        return remaining() - reserve

    def m1(x):
        if left() <= 0:
            raise _Stop
        y, fl = measure(x)
        st["n"] += 1
        return _Inc(x, y, np.asarray(fl, dtype=bool), score(y, fl))

    def remeasure(c):
        if remaining() <= 0:
            raise _Stop
        y, fl = measure(c.x)
        st["n"] += 1
        c.ysum += y
        c.fl |= np.asarray(fl, dtype=bool)
        c.n += 1
        c.f = float(score(c.ysum / c.n, c.fl))

    def offer(c):
        pool[c.x.tobytes()] = c
        b = min(pool.values(), key=lambda q: q.f)
        if pk["x"] is None or b.x.tobytes() != pk["x"].tobytes() or b.f != pk["f"]:
            changed = pk["x"] is None or b.x.tobytes() != pk["x"].tobytes()
            pk["x"], pk["f"] = b.x.copy(), float(b.f)
            if changed:
                cb(pk["x"])

    def start(x):
        c = m1(x)
        offer(c)
        return c

    def local_search(inc):
        order = rng.permutation(p)
        fails, j = 0, 0
        while fails < p:
            i = order[j % p]
            j += 1
            z = inc.x.copy()
            z[i] ^= 1
            c = m1(z)
            if c.f < inc.f:
                fails = 0
                inc = c
                st["acc"] += 1
                offer(inc)
            else:
                fails += 1
        return inc

    try:
        star = local_search(start(rng.integers(0, two).astype(np.int64)))
        while True:
            c = local_search(start(kicked(star.x)))
            st["iter"] += 1
            if c.f <= star.f:                          # 严格更好或不差（中性漂移）都接受
                star = c
            elif left() > 0 and star.n < int(max_remeasure):
                remeasure(star)
                st["rem"] += 1
                offer(star)
    except _Stop:
        pass
    if not pool:
        x = rng.integers(0, two).astype(np.int64)
        cb(x)
        return {"params": x, "J": None, "n_measurements": st["n"], "n_ils_iter": st["iter"], "n_accept": st["acc"],
                "n_star_remeasure": st["rem"]}
    # ---- 终选赛
    cands = sorted(pool.values(), key=lambda q: q.f)[: max(1, race_m)]
    if race_m > 1:
        for _ in range(race_k):
            for c in cands:
                if c.n < race_k + 1 and remaining() > 0:
                    remeasure(c)
        while remaining() > 0:
            for c in sorted(cands, key=lambda q: q.f)[:2]:
                if remaining() > 0:
                    remeasure(c)
    best = min(cands, key=lambda q: q.f)
    cb(best.x)
    return {"params": best.x.copy(), "J": float(best.f), "n_measurements": st["n"], "n_ils_iter": st["iter"],
            "n_accept": st["acc"], "n_star_remeasure": st["rem"]}
