"""离散码本模型 oracle（DESIGN §6.2）：随机重启 + 坐标下降，可选模拟退火预热。

用途：在无噪声的模型损失上回答"这个 1-bit（或多电平）结构最多能做到多好"。
坐标下降：逐分量（随机顺序）尝试该分量的全部电平，取最优；直到一整轮无改进或达到 max_sweeps。
非有限损失（nan/inf，例如数值奇异的样本）一律按 +inf 处理：nan 与任何数比较都为 False，
否则第一次重启出现 nan 就会让之后所有有限结果都被丢弃（审查 TR-5）。
"""
from __future__ import annotations

from typing import Callable

import numpy as np


def _eval(loss_fn, x) -> float:
    """一次损失评估；非有限值映射为 +inf。"""
    f = float(loss_fn(x))
    return f if np.isfinite(f) else np.inf


def _anneal(loss_fn, x, f, L, rng, n_steps):
    """模拟退火预热：单分量随机改电平，Metropolis 接受；温度几何下降 T0 → T0·1e-3。

    T0 取当前点处 p 个随机单分量变动的 |ΔL| 中位数（无变化时取 1）。返回途中最优 (x, f)。
    """
    p = L.size
    probes = []
    for _ in range(p):
        i = rng.integers(p)
        y = x.copy()
        y[i] = (y[i] + rng.integers(1, L[i])) % L[i]
        fy = _eval(loss_fn, y)
        probes.append(abs(fy - f) if np.isfinite(fy) and np.isfinite(f) else np.inf)
    t0 = float(np.median(probes))
    if not (np.isfinite(t0) and t0 > 0):
        t0 = 1.0
    best_x, best_f = x.copy(), f
    ratio = 1e-3 ** (1.0 / max(n_steps - 1, 1))
    temp = t0
    for _ in range(n_steps):
        i = rng.integers(p)
        y = x.copy()
        y[i] = (y[i] + rng.integers(1, L[i])) % L[i]
        fy = _eval(loss_fn, y)
        if not np.isfinite(fy):                     # +inf：永不接受
            temp *= ratio
            continue
        d = fy - f                                  # f = +inf 时 d = −inf → 接受
        if d <= 0 or rng.random() < np.exp(-d / temp):
            x, f = y, fy
            if f < best_f:
                best_x, best_f = x.copy(), f
        temp *= ratio
    return best_x, best_f


def _coordinate_descent(loss_fn, x, f, L, rng, max_sweeps):
    """逐分量穷举所有电平的坐标下降；一整轮无严格改进即停止。"""
    p = L.size
    for _ in range(max_sweeps):
        improved = False
        for i in rng.permutation(p):
            cur = x[i]
            best_l, best_f = cur, f
            for lv in range(L[i]):
                if lv == cur:
                    continue
                x[i] = lv
                fy = _eval(loss_fn, x.copy())   # 传副本，防止回调保存引用后被原地修改
                if fy < best_f:
                    best_l, best_f = lv, fy
            x[i] = best_l
            if best_l != cur:
                f = best_f
                improved = True
        if not improved:
            break
    return x, f


def oracle_bits(loss_fn_exact: Callable[[np.ndarray], float], levels, rng: np.random.Generator,
                n_restarts: int = 8, max_sweeps: int = 30, anneal_steps: int = 0) -> tuple[np.ndarray, float]:
    """随机重启 + 坐标下降（+ 可选退火）求离散码本最优（DESIGN §6.2）。

    loss_fn_exact(x_int) → float：无噪声模型损失；levels：(p,) 每分量电平数。
    返回 (best_params_int (p,) int64, best_loss)；所有评估都非有限时 best_loss = +inf。
    """
    L = np.asarray(levels, dtype=np.int64).ravel()
    if np.any(L < 2):
        raise ValueError("每个分量至少 2 个电平")
    best_x, best_f = None, np.inf
    for _ in range(max(int(n_restarts), 1)):
        x = rng.integers(0, L).astype(np.int64)
        f = _eval(loss_fn_exact, x.copy())
        if anneal_steps > 0:
            x, f = _anneal(loss_fn_exact, x, f, L, rng, int(anneal_steps))
        x, f = _coordinate_descent(loss_fn_exact, x.copy(), f, L, rng, max_sweeps)
        if best_x is None or f < best_f:
            best_x, best_f = x.copy(), f
    return best_x, float(best_f)
