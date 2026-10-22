"""标量格林函数与节点间耦合（DESIGN §1.2，方案 §4.1）。

约定：e^{jωt}，G(R) = exp(-jkR)/(4πR)；|E|² = 功率密度；源幅度 s、接收量 v 单位 √W。
四个耦合函数：a 为行、b 为列，返回 (Na, Nb) complex128，完全向量化。
记 R = |r_a − r_b|，d̂_{a→b} = (r_b − r_a)/R；每个节点的方向图都在"从该节点指向另一节点"的方向上求值：
  src_to_scat  = sqrt(4π) · F_b(d̂_{b→a}) · G(R) · F_a(d̂_{a→b})              源 b(1 √W) → 散射体 a 的局部场
  scat_to_scat = k² · G(R) · F_b(d̂_{b→a}) · F_a(d̂_{a→b})                   偶极矩 b → 散射体 a 的局部场（同集合对角为 0）
  scat_to_rcv  = (λ/sqrt(4π)) · F_a(d̂_{a→b}) · k² · G(R) · F_b(d̂_{b→a})    偶极矩 b → 接收端 a 的 v
  src_to_rcv   = λ · F_a(d̂_{a→b}) · F_b(d̂_{b→a}) · G(R)                     源 b → 接收端 a 的 v（Friis）
推导：源功率 P、方向性 D 在 R 处 |E|² = P·D/(4πR²) ⇒ E = s·sqrt(4π)·F·G；
接收有效面积 A_e = λ²D/(4π) ⇒ v = E·λF/sqrt(4π)；偶极子辐射 E = k²G·p。
"""
from __future__ import annotations

import numpy as np

from .scene import NodeSet

MIN_DIST = 1e-9                    # 跨集合节点最小距离 (m)，更近直接报错（DESIGN §1.2）
_SQRT4PI = float(np.sqrt(4.0 * np.pi))


def green(R, k: float):
    """标量自由空间格林函数 G(R) = exp(-jkR)/(4πR)（e^{jωt} 约定，外行波）。"""
    R = np.asarray(R, dtype=np.float64)
    return np.exp(-1j * k * R) / (4.0 * np.pi * R)


def _check_k_lam(k: float, lam: float | None) -> float:
    """校验 k，并返回 λ = 2π/k；若同时给了 lam，要求与 k 一致（真空）。"""
    if not (np.isfinite(k) and k > 0):
        raise ValueError(f"k 必须为正有限数，得到 {k}")
    lam_k = 2.0 * np.pi / k
    if lam is not None and abs(lam - lam_k) > 1e-9 * lam_k:
        raise ValueError(f"lam={lam} 与 k={k} 不一致（应为 {lam_k}）")
    return lam_k


def _pair_terms(a: NodeSet, b: NodeSet, k: float, same: bool):
    """公共部分：返回 (G, F_a, F_b)，F 为 (Na,Nb) 数组或标量 1.0（各向同性）。

    same=True 时对角（自身）距离被替换为 1 以避免除零，调用方须将对角置 0。
    """
    diff = b.pos[None, :, :] - a.pos[:, None, :]          # r_b − r_a，即 a→b 方向 (Na,Nb,3)
    R = np.sqrt(np.einsum("abk,abk->ab", diff, diff))
    Rc = R.copy()
    if same:
        np.fill_diagonal(Rc, np.inf)
    if Rc.size and Rc.min() < MIN_DIST:
        i, j = np.unravel_index(np.argmin(Rc), Rc.shape)
        raise ValueError(
            f"节点距离 {Rc[i, j]:.3e} m < {MIN_DIST} m（a[{i}]={a.pos[i]}, b[{j}]={b.pos[j]}）"
        )
    if same:
        np.fill_diagonal(R, 1.0)
    G = np.exp(-1j * k * R) / (4.0 * np.pi * R)
    # cos 在各自节点处：n̂_a·d̂_{a→b}，n̂_b·d̂_{b→a} = −n̂_b·d̂_{a→b}
    if a.pattern.isotropic:
        Fa = 1.0
    else:
        Fa = a.pattern.amp_cos(np.einsum("abk,ak->ab", diff, a.normal) / R)
    if b.pattern.isotropic:
        Fb = 1.0
    else:
        Fb = b.pattern.amp_cos(-np.einsum("abk,bk->ab", diff, b.normal) / R)
    return G, Fa, Fb


def _empty_or_none(a: NodeSet, b: NodeSet):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.complex128)
    return None


def src_to_scat(a: NodeSet, b: NodeSet, k: float, *, lam: float | None = None) -> np.ndarray:
    """源 b（单位幅度 1 √W）在散射体 a 处的方向图加权局部场 (Na,Nb)。

    = sqrt(4π)·F_b(d̂_{b→a})·G(R)·F_a(d̂_{a→b})   （DESIGN §1.2）
    """
    _check_k_lam(k, lam)
    out = _empty_or_none(a, b)
    if out is not None:
        return out
    G, Fa, Fb = _pair_terms(a, b, k, same=False)
    return (_SQRT4PI * Fb * G * Fa).astype(np.complex128, copy=False)


def scat_to_scat(a: NodeSet, b: NodeSet, k: float, same: bool | None = None, *,
                 lam: float | None = None) -> np.ndarray:
    """散射体 b 的单位偶极矩在散射体 a 处的局部场 (Na,Nb)。

    = k²·G(R)·F_b(d̂_{b→a})·F_a(d̂_{a→b})   （DESIGN §1.2）
    same：a、b 是否为同一集合（对角=自作用，置 0）；None 时按 `a is b` 判断。
    same=True 时要求两者节点位置完全相同。同集合内的非对角重合节点同样报错。
    """
    _check_k_lam(k, lam)
    if same is None:
        same = a is b
    if same and (len(a) != len(b) or not np.array_equal(a.pos, b.pos)):
        raise ValueError("same=True 要求 a、b 节点位置完全相同")
    out = _empty_or_none(a, b)
    if out is not None:
        return out
    G, Fa, Fb = _pair_terms(a, b, k, same=same)
    C = (k * k) * G * Fb * Fa
    C = np.asarray(C, dtype=np.complex128)
    if same:
        np.fill_diagonal(C, 0.0)
    return C


def scat_to_rcv(a: NodeSet, b: NodeSet, k: float, *, lam: float | None = None) -> np.ndarray:
    """散射体 b 的单位偶极矩在接收端 a 产生的接收量 v (Na,Nb)。

    = (λ/sqrt(4π))·F_a(d̂_{a→b})·k²·G(R)·F_b(d̂_{b→a})   （DESIGN §1.2）
    """
    lam_k = _check_k_lam(k, lam)
    out = _empty_or_none(a, b)
    if out is not None:
        return out
    G, Fa, Fb = _pair_terms(a, b, k, same=False)
    return ((lam_k / _SQRT4PI) * (k * k) * Fa * G * Fb).astype(np.complex128, copy=False)


def src_to_rcv(a: NodeSet, b: NodeSet, k: float, *, lam: float | None = None) -> np.ndarray:
    """源 b（单位幅度）在接收端 a 的接收量 v (Na,Nb)。

    = λ·F_a(d̂_{a→b})·F_b(d̂_{b→a})·G(R)，满足 Friis：|v|² = D_a D_b (λ/4πR)²   （DESIGN §1.2）
    """
    lam_k = _check_k_lam(k, lam)
    out = _empty_or_none(a, b)
    if out is not None:
        return out
    G, Fa, Fb = _pair_terms(a, b, k, same=False)
    return (lam_k * Fa * Fb * G).astype(np.complex128, copy=False)
