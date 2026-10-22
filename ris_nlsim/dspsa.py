"""离散同时扰动随机逼近 DSPSA（DESIGN §6.1，方案 §4.7）。

两个变体：
- ``faithful``：Wang & Spall (2011) DSPSA，"照搬 Song"。θ ∈ [0, L−1]（逐分量 L = 电平数），
  π(θ) = floor(θ) + ½（floor 截到 ≤ L−2），Δ ~ Rademacher，x± = π(θ) ± Δ/2，
  ĝ = (L(x+) − L(x−))·Δ，θ ← clip(θ − a_k ĝ, 0, L−1−τ)，τ = 1e-10（INTEGRATION G2，同 Song），a_k = a/(1+k+A)^α。
  1-bit（L=2）时 floor(θ) ≡ 0 ⇒ x± = (1±Δ)/2 恒为一对互补码本，与 θ 无关 → 退化（保留以作对照）。
- ``subset``：1-bit 实用变体。每次随机选 m 个分量 S，Δ_S ~ Rademacher，其余分量取当前判决 round(θ)；
  x±_S = π(θ_S) ± Δ_S/2（1-bit 时即 (1±Δ_S)/2），ĝ_S = (L(x+) − L(x−))Δ_S，只更新 S。

每次迭代 2 次测量（先 x+ 后 x−），扰动幅度 ±½。
非有限测量（nan/inf，例如数值奇异的中继样本）不更新 θ：该次迭代跳过并计入 history['n_nonfinite']。

增益 a（审查 S2-4）：a="auto" 在 θ0 处按 mean|ĝ| 归一化（auto_gain），会把"整体变平的损失地形"按比例放大回去——
比较不同条件的收敛速度（方案 RQ6）时请在参考条件上用 auto_gain 标定一次，再把同一个数值 a 传给各条件。

梯度形式（INTEGRATION T，训练器探索后增补）：
- grad="raw"（默认，上述原版）：ĝ = (L+ − L−)·Δ。
- grad="sign"（比较式 / sign-SPSA）：ĝ = sgn(L+ − L−)·Δ，只用两次测量的大小关系，对损失尺度不变；
  配 step0：a = step0·(1+A)^α，首步 |Δθ| 恰为 step0（θ ∈ [0, L−1] 单位），无需标定测量。
- gain_index="update"：a_k 的 k 只在实际更新 θ 的迭代上递增（非有限测量跳过时不推进；sign_acc 变体），
  "iter"（默认）为迭代序号（原版行为）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

_THETA_EPS = 1e-10  # θ 上界 L−1−1e-10（INTEGRATION G2 / Song 的 τ），floor(L−1−τ) = L−2 不越界


@dataclass
class DSPSAResult:
    theta: np.ndarray            # (p,) 连续内部参数
    bits: np.ndarray             # (p,) int，最终判决 round(θ)
    history: dict = field(default_factory=dict)
    n_measurements: int = 0      # 损失函数调用总次数（含 a='auto' 的 20 次）


def _as_levels(levels, p: int | None) -> np.ndarray:
    lv = np.asarray(levels, dtype=np.int64)
    if lv.ndim == 0:
        if p is None:
            raise ValueError("levels 为标量时必须给出 theta0 以确定参数维数 p")
        lv = np.full(p, int(lv), dtype=np.int64)
    lv = lv.ravel()
    if np.any(lv < 2):
        raise ValueError("每个分量至少 2 个电平")
    return lv


def _center(theta: np.ndarray, L: np.ndarray) -> np.ndarray:
    """π(θ) = floor(θ) + ½，floor 截到 [0, L−2]（DESIGN §6.1）。"""
    fl = np.clip(np.floor(theta), 0, L - 2)
    return fl + 0.5


def decide(theta: np.ndarray, L: np.ndarray) -> np.ndarray:
    """判决 round(θ)（四舍五入，非银行家舍入），截到 [0, L−1]。"""
    return np.clip(np.floor(theta + 0.5), 0, L - 1).astype(np.int64)


def _rademacher(rng: np.random.Generator, n: int) -> np.ndarray:
    return 2.0 * rng.integers(0, 2, size=n) - 1.0


def _perturb(theta, L, rng, variant, m):
    """生成一对扰动码本 (x+, x−) 及被扰动分量 idx、Δ。"""
    p = theta.size
    if variant == "faithful":
        idx = np.arange(p)
        x0 = np.zeros(p, dtype=np.int64)
    else:
        idx = np.sort(rng.choice(p, size=m, replace=False))
        x0 = decide(theta, L)
    delta = _rademacher(rng, idx.size)
    c = _center(theta[idx], L[idx])
    x_plus, x_minus = x0.copy(), x0.copy()
    x_plus[idx] = np.rint(c + 0.5 * delta).astype(np.int64)
    x_minus[idx] = np.rint(c - 0.5 * delta).astype(np.int64)
    return idx, delta, x_plus, x_minus


def subset_size_default(p: int, subset_size: int | None = None) -> int:
    """subset 变体每次扰动的分量数 m：默认 max(2, p//8)，截到 [1, p]（DESIGN §6.1）。"""
    m = int(subset_size) if subset_size is not None else max(2, p // 8)
    return min(max(m, 1), p)


def auto_gain(loss_fn: Callable[[np.ndarray, np.random.Generator], float], levels, rng: np.random.Generator,
              theta=None, variant: str = "faithful", subset_size: int | None = None, A: float = 400.0,
              alpha: float = 0.501, n_est: int = 10) -> dict:
    """a="auto" 的标定（DESIGN §6.1）：在 θ 处做 n_est 次独立梯度估计，a = 0.3·(1+A)^α / mean|ĝ|。

    θ=None → 逐分量 U[0, L−1)（levels 须为数组）。mean|ĝ| = 0 或全部非有限（例如 1-bit faithful 遇对称损失）时取
    mean|ĝ| := 1 并记 degenerate=True。返回 {"a", "mean_abs_g", "n_measurements", "n_nonfinite", "degenerate"}。
    公开它是为了让多条件比较共用同一个 a（审查 S2-4）：在参考条件上调用一次，再把 res["a"] 传给 dspsa(a=...)。
    """
    p_hint = None if theta is None else np.asarray(theta).size
    L = _as_levels(levels, p_hint)
    p = L.size
    if theta is None:
        theta = rng.uniform(0.0, 1.0, size=p) * (L - 1)
    theta = np.clip(np.asarray(theta, dtype=float).ravel(), 0.0, L - 1 - _THETA_EPS)
    m = subset_size_default(p, subset_size)
    g_abs, n_bad, n_meas = [], 0, 0
    for _ in range(int(n_est)):
        idx, delta, xp, xm = _perturb(theta, L, rng, variant, m)
        yp = float(loss_fn(xp, rng))
        ym = float(loss_fn(xm, rng))
        n_meas += 2
        if np.isfinite(yp - ym):
            g_abs.append(np.abs((yp - ym) * delta))
        else:
            n_bad += 1
    mean_g = float(np.mean(np.concatenate(g_abs))) if g_abs else float("nan")
    degenerate = not (np.isfinite(mean_g) and mean_g > 0.0)
    a_val = 0.3 * (1.0 + A) ** alpha / (1.0 if degenerate else mean_g)
    return {"a": float(a_val), "mean_abs_g": mean_g, "n_measurements": n_meas, "n_nonfinite": n_bad,
            "degenerate": bool(degenerate)}


def dspsa(loss_fn: Callable[[np.ndarray, np.random.Generator], float], levels: np.ndarray, n_iter: int,
          rng: np.random.Generator, a: float | str = "auto", A: float = 400.0, alpha: float = 0.501,
          variant: str = "faithful", subset_size: int | None = None, theta0=None,
          callback: Callable[[int, np.ndarray], object] | None = None,
          callback_every: int = 50, grad: str = "raw", step0: float | None = None,
          gain_index: str = "iter") -> DSPSAResult:
    """DSPSA 主循环（DESIGN §6.1）。

    loss_fn(x_int, rng) → float：一次（可带噪）测量；x_int 为 (p,) 整数电平。
    levels：(p,) 每分量电平数（或标量，此时须给 theta0）。
    theta0：初始 θ；None → 逐分量 U[0, L−1)（随机初始码本）。
    a="auto"：在 θ0 处做 10 次独立梯度估计，a = 0.3·(1+A)^α / mean|ĝ|（20 次测量计入 n_measurements；
              若 mean|ĝ| = 0（例如 1-bit faithful 遇对称损失）则取 mean|ĝ| := 1，并记 history['a_auto_degenerate']；
              含非有限测量的估计不参与平均）。
    grad："raw"（ĝ = ΔL·Δ）| "sign"（ĝ = sgn(ΔL)·Δ，INTEGRATION T）。
    step0：给定时 a = step0·(1+A)^α（首步 = step0，不做标定，覆盖 a）；grad="sign" 必须给 step0 或数值 a。
    gain_index："iter"（a_k 用迭代序号）| "update"（只在实际更新的迭代上递增，非有限测量不推进）。
    callback(k, θ)：k = 已完成迭代数；在 k = 0、callback_every 的整数倍以及 k = n_iter 时调用，
              返回值（非 None）依次存入 history['callback']，对应 k 存入 history['callback_iter']。
    history：loss_plus / loss_minus / a_k / grad_abs_mean / n_flips（判决翻转数）/ nonfinite（该迭代测量非有限、
             θ 未更新）每迭代一项，以及 a（实际增益）、n_nonfinite（含标定阶段的非有限梯度估计总数）、callback、callback_iter。
    """
    if variant not in ("faithful", "subset"):
        raise ValueError(f"未知 DSPSA 变体 {variant!r}")
    if grad not in ("raw", "sign"):
        raise ValueError(f"grad 只能是 'raw' | 'sign'，得到 {grad!r}")
    if gain_index not in ("iter", "update"):
        raise ValueError(f"gain_index 只能是 'iter' | 'update'，得到 {gain_index!r}")
    p_hint = None if theta0 is None else np.asarray(theta0).size
    L = _as_levels(levels, p_hint)
    p = L.size
    if theta0 is None:
        theta = rng.uniform(0.0, 1.0, size=p) * (L - 1)
    else:
        theta = np.asarray(theta0, dtype=float).ravel().copy()
        if theta.size != p:
            raise ValueError("theta0 与 levels 维数不一致")
    hi = L - 1 - _THETA_EPS
    theta = np.clip(theta, 0.0, hi)
    m = subset_size_default(p, subset_size)

    n_meas = 0
    n_bad = 0                                       # 非有限梯度估计计数
    history: dict = {}

    # ---- 增益 a：step0（首步固定，INTEGRATION T）或自动标定（DESIGN §6.1）
    if step0 is not None:
        a_val = float(step0) * (1.0 + A) ** alpha
    elif grad == "sign" and isinstance(a, str):
        raise ValueError("grad='sign' 需要 step0 或数值 a（mean|ĝ| 标定对符号梯度无意义）")
    elif isinstance(a, str):
        if a != "auto":
            raise ValueError("a 只能是数值或 'auto'")
        cal = auto_gain(loss_fn, L, rng, theta=theta, variant=variant, subset_size=m, A=A, alpha=alpha)
        n_meas += cal["n_measurements"]
        n_bad += cal["n_nonfinite"]
        history["a_auto_degenerate"] = np.array([cal["degenerate"]])
        a_val = cal["a"]
    else:
        a_val = float(a)
    history["a"] = np.array([a_val])

    loss_plus = np.empty(n_iter)
    loss_minus = np.empty(n_iter)
    a_hist = np.empty(n_iter)
    g_hist = np.empty(n_iter)
    flips = np.zeros(n_iter, dtype=np.int64)
    nonfinite = np.zeros(n_iter, dtype=bool)
    cb_vals, cb_iter = [], []

    def _cb(k):
        if callback is None:
            return
        ret = callback(k, theta.copy())
        if ret is not None:
            cb_vals.append(ret)
            cb_iter.append(k)

    _cb(0)
    k_gain = 0                                      # gain_index="update"：已实际更新的迭代数
    for k in range(n_iter):
        idx, delta, xp, xm = _perturb(theta, L, rng, variant, m)
        yp = float(loss_fn(xp, rng))
        ym = float(loss_fn(xm, rng))
        n_meas += 2
        kk = k if gain_index == "iter" else k_gain
        a_k = a_val / (1.0 + kk + A) ** alpha       # a_k = a/(1+k+A)^α
        loss_plus[k], loss_minus[k] = yp, ym
        a_hist[k] = a_k
        if not np.isfinite(yp - ym):
            # 非有限测量：ĝ 无意义，跳过本次更新（否则 nan 会污染全部被扰动分量，判决变成 INT64_MIN）
            nonfinite[k] = True
            n_bad += 1
            g_hist[k] = np.nan
        else:
            if grad == "sign":
                g = float(np.sign(yp - ym)) * delta  # ĝ = sgn(L(x+) − L(x−))·Δ（INTEGRATION T）
            else:
                g = (yp - ym) * delta               # ĝ = (L(x+) − L(x−))·Δ（Δ^{-1} = Δ）
            before = decide(theta, L)
            theta[idx] = np.clip(theta[idx] - a_k * g, 0.0, hi[idx])
            flips[k] = int(np.count_nonzero(decide(theta, L) != before))
            g_hist[k] = float(np.mean(np.abs(g)))
            k_gain += 1
        done = k + 1
        if callback is not None and (done % callback_every == 0 or done == n_iter):
            _cb(done)

    history.update(loss_plus=loss_plus, loss_minus=loss_minus, a_k=a_hist,
                   grad_abs_mean=g_hist, n_flips=flips, nonfinite=nonfinite,
                   n_nonfinite=np.array([n_bad]),
                   callback=cb_vals, callback_iter=np.asarray(cb_iter, dtype=np.int64))
    return DSPSAResult(theta=theta, bits=decide(theta, L), history=history, n_measurements=n_meas)
