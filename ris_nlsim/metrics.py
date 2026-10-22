"""指标：Walsh（布尔 Fourier）高阶权重、奇偶标签、裕量、准确率、所需 SNR、零陷深度（方案 §7；DESIGN §5）。

编码约定：样本索引 i ↔ x_j = (i >> j) & 1；Walsh 系数索引 S 用同样的位编码，|S| = popcount(S)。
"""
from __future__ import annotations

import numpy as np

_DENOM_REL = 1e-30   # 非常数部分能量 ≤ 1e-30 × 总能量 视为"分母为 0"（浮点噪声级）


def popcount(idx) -> np.ndarray:
    """逐元素统计二进制 1 的个数（仅非负整数；负数的算术右移永不归零，故直接报错）。"""
    idx = np.asarray(idx, dtype=np.int64)
    if np.any(idx < 0):
        raise ValueError("popcount 只接受非负整数")
    out = np.zeros_like(idx)
    while np.any(idx):
        out += idx & 1
        idx = idx >> 1
    return out


def _check_n(values: np.ndarray, n: int) -> None:
    if values.shape[0] != 2 ** n:
        raise ValueError(f"values 第 0 维应为 2**n = {2 ** n}，得到 {values.shape[0]}")


def walsh_coeffs(values, n: int) -> np.ndarray:
    """f̂(S) = 2^{-n} Σ_x f(x)(−1)^{|S∧x|}（方案 §7），沿第 0 维做快速 Walsh–Hadamard 变换。

    values: (2**n, ...) 实数或复数；返回同形状。逆变换：f = 2^n · walsh_coeffs(f̂, n)。
    """
    v = np.asarray(values)
    _check_n(v, n)
    dtype = np.complex128 if np.iscomplexobj(v) else np.float64
    N = 2 ** n
    tail = v.shape[1:]
    a = v.astype(dtype, copy=True).reshape(N, -1)
    h = 1
    while h < N:
        # 索引 i = blk·2h + bit·h + low，bit 即第 log2(h) 位
        a = a.reshape(N // (2 * h), 2, h, -1)
        x, y = a[:, 0], a[:, 1]
        a = np.stack([x + y, x - y], axis=1)
        h *= 2
    return (a.reshape((N,) + tail) / N)


def _power_by_degree(values, n: int) -> tuple[np.ndarray, float]:
    """各阶（0..n）Walsh 能量 Σ_{|S|=j}|f̂(S)|²（向量值对分量求和）与总能量。"""
    c = walsh_coeffs(values, n).reshape(2 ** n, -1)
    pw = np.sum(np.abs(c) ** 2, axis=1)
    deg = popcount(np.arange(2 ** n))
    by_deg = np.bincount(deg, weights=pw, minlength=n + 1)
    return by_deg, float(pw.sum())


def walsh_weight_ge(values, n: int, d: int) -> float:
    """W≥d = Σ_{|S|≥d}|f̂(S)|² / Σ_{S≠∅}|f̂(S)|²（方案 §7）；分母为 0（含浮点噪声级）返回 0。"""
    by_deg, total = _power_by_degree(values, n)
    denom = by_deg[1:].sum()
    if denom <= 0.0 or denom <= _DENOM_REL * total:
        return 0.0
    return float(by_deg[max(d, 1):].sum() / denom)


def walsh_degree_profile(values, n: int) -> np.ndarray:
    """长度 n：第 j 项 = 阶数恰为 j+1 的归一化 Walsh 权重（和为 1；分母为 0 时全 0）。"""
    by_deg, total = _power_by_degree(values, n)
    denom = by_deg[1:].sum()
    if denom <= 0.0 or denom <= _DENOM_REL * total:
        return np.zeros(n)
    return by_deg[1:] / denom


def parity_labels(n: int) -> np.ndarray:
    """奇偶标签 (2**n,) int：x 中 1 的个数为奇 → 1。"""
    return (popcount(np.arange(2 ** n)) % 2).astype(int)


def binary_margin_db(readings_dbm, labels) -> float:
    """判决裕量 min_x sgn(x)·(y1−y2)（dB），sgn = +1（奇/label 1）/ −1（偶/label 0）。

    > 0 ⇔ 全部判对（y1 > y2 判为奇，DESIGN §5）。注：方案 §7 写作 min|y1−y2|，这里按 DESIGN 取带符号版本。
    """
    y = np.asarray(readings_dbm, dtype=float)
    sgn = np.where(np.asarray(labels) == 1, 1.0, -1.0)
    return float(np.min(sgn * (y[..., 0] - y[..., 1])))


def accuracy(pred, labels) -> float:
    """判对比例。"""
    pred = np.asarray(pred)
    labels = np.asarray(labels)
    return float(np.mean(pred == labels))


def required_snr_db(snr_db, acc, target: float = 0.9) -> float:
    """准确率达到 target 的最低 SNR（方案 §7），线性插值。

    取"此后始终 ≥ target"的最低交点（对含噪、非单调曲线更稳健）；
    最低 SNR 处已达标返回该网格点；最高 SNR 仍不达标返回 nan。
    """
    s = np.asarray(snr_db, dtype=float)
    a = np.asarray(acc, dtype=float)
    order = np.argsort(s)
    s, a = s[order], a[order]
    ok = a >= target
    if not ok[-1]:
        return float("nan")
    bad = np.nonzero(~ok)[0]
    if bad.size == 0:
        return float(s[0])
    i = bad[-1]                       # 最后一个不达标点，i+1 达标
    s0, s1, a0, a1 = s[i], s[i + 1], a[i], a[i + 1]
    return float(s0 + (target - a0) * (s1 - s0) / (a1 - a0))


def null_depth_db(u0, u0_ref, reduce: str = "max"):
    """零陷深度 10·log10(|u₀|²/|u₀,ref|²)（DESIGN §3.3，逐元素，u0 与 u0_ref 按 numpy 广播）。

    reduce="max"（默认，返回 float）：取逐元素深度的最大值 = 最浅（最差）的零陷，
      对应 INTEGRATION §C "N_r>1 取各中继最大值"；u0 为每样本 (B,)、参考为标量聚焦值时同理。
    reduce="none"：返回逐元素深度数组（0 维时为 float），由调用方自行归约。
    注：不再对元素求功率和——(B,) 对标量参考求和会带来 +10log10(B) dB 的偏差。
    """
    p = np.abs(np.asarray(u0)) ** 2
    p_ref = np.abs(np.asarray(u0_ref)) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        nd = 10.0 * np.log10(p / p_ref)              # 广播；形状不兼容时 numpy 报 ValueError
    if reduce == "max":
        return float(np.max(nd))
    if reduce == "none":
        return float(nd) if np.ndim(nd) == 0 else nd
    raise ValueError(f"reduce 只能是 'max' 或 'none'，得到 {reduce!r}")
