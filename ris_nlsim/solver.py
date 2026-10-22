"""线性场求解：耦合偶极子 + 三种物体模式 + 训练用约化表示（DESIGN §2.2–§2.3，方案 §4.1–§4.3）。

散射体方程（方案 §4.1）：(diag(1/α) − C) p = E0，C 为散射体两两 scat_to_scat，E0 为源列 src_to_scat；
接收（DESIGN §2.2）：E_R = E0_R + C_RS p，E0_R 为直达项 src_to_rcv，C_RS 为 scat_to_rcv。
返回的 H 形状 (N_RCV, S)：行 [rx.., probes]，列 [feed(合成), tx..]。

数值取舍：A = diag(1/α) − C 的作用 A⁻¹·rhs 统一写成 (I − diag(α)C)⁻¹·(α⊙rhs)，LU 求解（不求显式逆）。
两式数学上相同，但后者对 α=0（不散射）的节点不会出现 1/0。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import lu_factor, lu_solve
from scipy.special import eval_legendre, spherical_jn, spherical_yn

from .greens import scat_to_scat
from .scene import (NodeSet, Scene, as_passive_phase, assemble_scat_to_rcv, assemble_scat_to_scat,
                    assemble_src_to_rcv, assemble_src_to_scat, check_scatterer_sets, passive_with_phase,
                    stack_alpha)

MODES = ("full", "born", "intra")


def _check_mode(mode: str) -> None:
    if mode not in MODES:
        raise ValueError(f"未知物体模式 {mode!r}，可选 {MODES}")


def _objects(objects: NodeSet | None) -> NodeSet:
    """None → 空集合；非空物体必须带 alpha。"""
    if objects is None:
        return NodeSet.empty()
    if not isinstance(objects, NodeSet):
        raise TypeError("objects 必须是 NodeSet 或 None")
    if len(objects) and objects.alpha is None:
        raise ValueError("objects 缺少 alpha")
    return objects


class _DressedInverse:
    """A = diag(1/α) − C 的 LU 分解；solve(rhs) = A⁻¹·rhs = (I − diag(α)C)⁻¹(α⊙rhs)。"""

    def __init__(self, alpha: np.ndarray, C: np.ndarray):
        self.alpha = np.asarray(alpha, dtype=np.complex128)
        self.n = len(self.alpha)
        self._lu = None
        if self.n:
            M = np.eye(self.n, dtype=np.complex128) - self.alpha[:, None] * C
            self._lu = lu_factor(M, check_finite=False)

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        rhs = np.asarray(rhs, dtype=np.complex128)
        if self.n == 0 or rhs.size == 0:
            return np.zeros(rhs.shape, dtype=np.complex128)
        a = self.alpha if rhs.ndim == 1 else self.alpha[:, None]
        return lu_solve(self._lu, a * rhs, check_finite=False)


# ---------------------------------------------------------------------------
# 物体响应（born / intra）
# ---------------------------------------------------------------------------

def object_groups(objects: NodeSet) -> list[np.ndarray]:
    """intra 模式的分块（每块 = 一个物体）。

    取舍：group=None → 全部节点视为同一个物体；group<0（"非物体"标记）→ 每个节点单独成块。
    """
    n = len(objects)
    if n == 0:
        return []
    if objects.group is None:
        return [np.arange(n)]
    g = objects.group
    blocks = [np.flatnonzero(g == v) for v in np.unique(g[g >= 0])]
    blocks += [np.array([i]) for i in np.flatnonzero(g < 0)]
    return blocks


def _intra_inverses(objects: NodeSet, k: float):
    """逐物体在自由空间的精确多次散射：(diag(1/α_g) − C_gg) 的分解（DESIGN §2.2 intra）。"""
    for idx in object_groups(objects):
        sub = objects.take(idx)
        yield idx, _DressedInverse(sub.alpha, scat_to_scat(sub, sub, k, same=True))


def _object_response(objects: NodeSet, k: float, mode: str, E: np.ndarray) -> np.ndarray:
    """p_O = Lam·E（方案 §4.1 第 3 步）：born 为 α⊙E；intra 为分块求解。E 为 (N_O,) 或 (N_O,M)。"""
    E = np.asarray(E, dtype=np.complex128)
    if len(objects) == 0:
        return np.zeros(E.shape, dtype=np.complex128)
    if mode == "born":
        a = objects.alpha if E.ndim == 1 else objects.alpha[:, None]
        return a * E
    if mode == "intra":
        p = np.zeros(E.shape, dtype=np.complex128)
        for idx, inv in _intra_inverses(objects, k):
            p[idx] = inv.solve(E[idx])
        return p
    raise ValueError(f"_object_response 不支持模式 {mode!r}")


def object_operator(objects: NodeSet, k: float, mode: str) -> np.ndarray:
    """物体响应算子 Lam (N_O,N_O)，p_O = Lam·E_O^bg（DESIGN §2.3）。

    born：diag(α_O)；intra：blockdiag_g[(diag(1/α_g) − C_gg)⁻¹]（物体间不耦合）。full 模式无此算子。
    """
    if mode not in ("born", "intra"):
        raise ValueError("object_operator 只用于 born / intra")
    objects = _objects(objects)
    n = len(objects)
    if mode == "born":
        return np.diag(objects.alpha) if n else np.zeros((0, 0), dtype=np.complex128)
    Lam = np.zeros((n, n), dtype=np.complex128)
    for idx, inv in _intra_inverses(objects, k):
        Lam[np.ix_(idx, idx)] = inv.solve(np.eye(len(idx), dtype=np.complex128))
    return Lam


# ---------------------------------------------------------------------------
# §2.2 直接（参考）求解
# ---------------------------------------------------------------------------

def solve_scene(scene: Scene, objects: NodeSet | None, passive_phase: np.ndarray | None,
                mode: str) -> np.ndarray:
    """直接求解场景响应 H (N_RCV, S)（DESIGN §2.2）。

    背景 B = env ∪ passive（α_P = alpha_base·e^{jφ}）。
    - full：物体、环境、无源面一起解 (diag(1/α) − C)p = E0，E_R = E0_R + C_RS p。
    - born（DWBA）：1) A_BB p_B^bg = E0_B；2) E_O^bg = E0_O + C_OB p_B^bg；3) p_O = α_O⊙E_O^bg；
      4) p_B = p_B^bg + A_BB⁻¹ C_BO p_O；5) E_R = E0_R + C_RB p_B + C_RO p_O。
      ⇒ H 对物体存在向量 x 仿射（物体散射场经背景到接收端，但不再照射任何物体）。
    - intra：同 born，第 3 步换成每个物体在自由空间内的精确多次散射（按 group 分块）。
    """
    _check_mode(mode)
    O = _objects(objects)
    check_scatterer_sets(scene, O)          # 物体与 env/passive 是同一对象时 born/intra 会静默算错
    k = scene.k
    bg = [*scene.env, passive_with_phase(scene, passive_phase)]
    H = assemble_src_to_rcv(scene)                                       # E0_R
    if mode == "full":
        scat = [O, *bg]
        inv = _DressedInverse(stack_alpha(scat), assemble_scat_to_scat(scat, scat, k))
        p = inv.solve(assemble_src_to_scat(scene, scat))
        return H + assemble_scat_to_rcv(scene, scat) @ p
    # DWBA：背景先解，物体只对背景场一阶（或体内精确）响应
    inv_b = _DressedInverse(stack_alpha(bg), assemble_scat_to_scat(bg, bg, k))
    p_bg = inv_b.solve(assemble_src_to_scat(scene, bg))                  # 1
    C_bo = assemble_scat_to_scat(bg, [O], k)
    E_o = assemble_src_to_scat(scene, [O]) + C_bo.T @ p_bg              # 2（互易：C_OB = C_BOᵀ）
    p_o = _object_response(O, k, mode, E_o)                              # 3
    p_b = p_bg + inv_b.solve(C_bo @ p_o)                                 # 4
    return (H + assemble_scat_to_rcv(scene, bg) @ p_b                    # 5
            + assemble_scat_to_rcv(scene, [O]) @ p_o)


# ---------------------------------------------------------------------------
# §2.3 约化表示（训练无源面相位时用）
# ---------------------------------------------------------------------------

@dataclass(eq=False)
class ReducedSample:
    """一个样本与无源面相位无关的部分（DESIGN §2.3）。所有量都已被 env "穿衣"（Schur 补消去环境）。

    full：物体也被 Schur 消去，H、Cpp、Ep、Hrp 含物体精确效应。
    born/intra 且 n_p>0：H、Cpp、Ep、Hrp 只含环境；物体量 Eo、Cop、Cpo、Lam、Hro 另存。
    n_p == 0：H 就是完整响应，其余字段为 None。
    """

    mode: str
    n_rx: int
    n_probe: int
    n_tx: int
    n_p: int
    H: np.ndarray                          # (N_RCV, S)
    alpha_p_base: np.ndarray | None = None  # (N_P,)
    Cpp: np.ndarray | None = None           # (N_P, N_P)
    Ep: np.ndarray | None = None            # (N_P, S)
    Hrp: np.ndarray | None = None           # (N_RCV, N_P)
    Eo: np.ndarray | None = None            # (N_O, S)
    Cop: np.ndarray | None = None           # (N_O, N_P)
    Cpo: np.ndarray | None = None           # (N_P, N_O)
    Lam: np.ndarray | None = None           # (N_O, N_O)
    Hro: np.ndarray | None = None           # (N_RCV, N_O)


def reduce_sample(scene: Scene, objects: NodeSet | None, mode: str) -> ReducedSample:
    """预计算与无源面相位无关的量（DESIGN §2.3）。

    记被消去集合 X（full：物体 ∪ 环境；born/intra：环境），A_XX = diag(1/α_X) − C_XX，
    一次 LU 求 Y = A_XX⁻¹ [E0_X | C_XP | C_XO]，则 Schur 补：
      H   = E0_R + C_RX Y_S          Ep  = E0_P + C_PX Y_S        Cpp = C_PP + C_PX Y_P
      Hrp = C_RP + C_RX Y_P          Eo  = E0_O + C_OX Y_S        Cop = C_OP + C_OX Y_P
      Cpo = C_PO + C_PX Y_O          Hro = C_RO + C_RX Y_O        （C_ba = C_abᵀ，互易）
    """
    _check_mode(mode)
    O = _objects(objects)
    check_scatterer_sets(scene, O)
    k = scene.k
    P = scene.passive
    n_p = len(P)
    if n_p and P.alpha is None:
        raise ValueError("无源面 passive 缺少 alpha")
    full = mode == "full"
    X = [O, *scene.env] if full else list(scene.env)
    n_s = scene.n_src
    inv_x = _DressedInverse(stack_alpha(X), assemble_scat_to_scat(X, X, k))
    E0x = assemble_src_to_scat(scene, X)
    Cxp = assemble_scat_to_scat(X, [P], k)
    Cxo = (np.zeros((len(E0x), 0), dtype=np.complex128) if full
           else assemble_scat_to_scat(X, [O], k))
    Y = inv_x.solve(np.hstack([E0x, Cxp, Cxo]))
    Ys, Yp, Yo = Y[:, :n_s], Y[:, n_s:n_s + n_p], Y[:, n_s + n_p:]
    Crx = assemble_scat_to_rcv(scene, X)
    H = assemble_src_to_rcv(scene) + Crx @ Ys
    base = dict(mode=mode, n_rx=scene.n_rx, n_probe=scene.n_probe, n_tx=scene.n_tx, n_p=n_p)

    if not full:
        Eo = assemble_src_to_scat(scene, [O]) + Cxo.T @ Ys
        Hro = assemble_scat_to_rcv(scene, [O]) + Crx @ Yo
        Lam = object_operator(O, k, mode)
        if n_p == 0:
            return ReducedSample(H=H + Hro @ (Lam @ Eo), **base)
    elif n_p == 0:
        return ReducedSample(H=H, **base)

    Ep = assemble_src_to_scat(scene, [P]) + Cxp.T @ Ys
    Cpp = assemble_scat_to_scat([P], [P], k) + Cxp.T @ Yp
    Hrp = assemble_scat_to_rcv(scene, [P]) + Crx @ Yp
    rs = ReducedSample(H=H, alpha_p_base=P.alpha.copy(), Cpp=Cpp, Ep=Ep, Hrp=Hrp, **base)
    if not full:
        Cop0 = assemble_scat_to_scat([O], [P], k)
        rs.Eo, rs.Lam, rs.Hro = Eo, Lam, Hro
        rs.Cop = Cop0 + Cxo.T @ Yp
        rs.Cpo = Cop0.T + Cxp.T @ Yo
    return rs


def eval_reduced(rs: ReducedSample, passive_phase: np.ndarray | None) -> np.ndarray:
    """由约化表示求 H (N_RCV, S)（DESIGN §2.3）。passive_phase=None 视为全 0。

    M = diag(1/(α_base·e^{jφ})) − Cpp；pP_bg = M⁻¹Ep；
    full：H + Hrp·pP_bg；
    born/intra：pO = Lam(Eo + Cop·pP_bg)，pP = pP_bg + M⁻¹(Cpo·pO)，返回 H + Hrp·pP + Hro·pO。
    数值上 M⁻¹·rhs 写成 (I − diag(α)Cpp)⁻¹(α⊙rhs)（与 solve_scene 同），α_base=0 的单元（Γ=0）不出现 1/0。
    passive_phase 必须是实数（复数 → TypeError）。
    """
    phase = as_passive_phase(passive_phase, rs.n_p)
    if phase is None:
        return rs.H.copy()
    inv = _DressedInverse(rs.alpha_p_base * np.exp(1j * phase), rs.Cpp)
    p_bg = inv.solve(rs.Ep)
    if rs.mode == "full":
        return rs.H + rs.Hrp @ p_bg
    p_o = rs.Lam @ (rs.Eo + rs.Cop @ p_bg)
    p_p = p_bg + inv.solve(rs.Cpo @ p_o)
    return rs.H + rs.Hrp @ p_p + rs.Hro @ p_o


# ---------------------------------------------------------------------------
# 自由空间平面波散射（光学定理 / 自检用）
# ---------------------------------------------------------------------------

def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(3)
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise ValueError("kvec_unit 不能为零向量")
    return v / n


def plane_wave(pos: np.ndarray, k: float, kvec_unit) -> np.ndarray:
    """单位幅度平面波 E_inc(r) = exp(−jk k̂·r)（e^{jωt}，沿 k̂ 传播）。"""
    return np.exp(-1j * k * (np.asarray(pos, dtype=np.float64) @ _unit(kvec_unit)))


def solve_objects_planewave(objects: NodeSet, k: float, kvec_unit, mode: str = "full") -> np.ndarray:
    """自由空间中物体在单位平面波照射下的偶极矩 p (N_O,)（方案 §4.1、§4.8）。

    full：(diag(1/α) − C_OO) p = E_inc；born：p = α⊙E_inc；intra：按 group 分块精确求解。
    """
    _check_mode(mode)
    O = _objects(objects)
    E = plane_wave(O.pos, k, kvec_unit)
    if len(O) == 0:
        return E.astype(np.complex128)
    if mode == "full":
        return _DressedInverse(O.alpha, scat_to_scat(O, O, k, same=True)).solve(E)
    return _object_response(O, k, mode, E)


def far_field_amplitude(p: np.ndarray, objects: NodeSet, k: float, dirs) -> np.ndarray:
    """远场散射幅度 f(k̂) = k²/(4π)·Σ_i p_i·exp(+jk k̂·r_i)，E_s ≈ f·e^{−jkr}/r；dirs (M,3) 或 (3,)。"""
    d = np.asarray(dirs, dtype=np.float64)
    single = d.ndim == 1
    d = d.reshape(-1, 3)
    d = d / np.linalg.norm(d, axis=1, keepdims=True)
    f = (k * k / (4.0 * np.pi)) * (np.exp(1j * k * (d @ objects.pos.T)) @ np.asarray(p))
    return f[0] if single else f


def sigma_ext(p: np.ndarray, objects: NodeSet, k: float, kvec_unit) -> float:
    """消光截面（|E0| = 1）：σ_ext = −k·Im Σ_i p_i·conj(E_inc(r_i))（e^{jωt} 下的光学定理，DESIGN §1.3）。"""
    E = plane_wave(objects.pos, k, kvec_unit)
    return float(-k * np.imag(np.sum(np.asarray(p) * np.conj(E))))


def sigma_sca(p: np.ndarray, objects: NodeSet, k: float) -> float:
    """散射截面（|E0| = 1）：σ_sca = ∮|f|²dΩ = k⁴/(4π)·Σ_ij p_i·conj(p_j)·sinc(k r_ij)，sinc(0) = 1。"""
    p = np.asarray(p, dtype=np.complex128)
    diff = objects.pos[:, None, :] - objects.pos[None, :, :]
    r = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))
    S = np.sinc(k * r / np.pi)                    # np.sinc(x) = sin(πx)/(πx)
    return float(k**4 / (4.0 * np.pi) * np.real(p @ S @ np.conj(p)))


# ---------------------------------------------------------------------------
# 标量 Mie 参考解（体素收敛 / 极化率模型验证用，方案 §4.8）
# ---------------------------------------------------------------------------

def _h2(l: int, x, derivative: bool = False):
    """第二类球 Hankel 函数 h_l^(2) = j_l − j·y_l（e^{jωt} 下的外行波）。"""
    return spherical_jn(l, x, derivative) - 1j * spherical_yn(l, x, derivative)


def mie_scalar_coeffs(eps_r, radius: float, k: float, lmax: int | None = None) -> np.ndarray:
    """均匀球的标量 Mie 系数 a_l，l = 0..lmax（∇²ψ + k²εrψ = 0，ψ 与 ∂ψ/∂r 在球面连续）。

    入射 e^{−jkz} = Σ(2l+1)(−j)^l j_l(kr)P_l，散射 Σ(2l+1)(−j)^l a_l h_l^(2)(kr)P_l，k1 = k·sqrt(εr)：
      a_l = [k1 j_l′(k1a) j_l(ka) − k j_l(k1a) j_l′(ka)] / [k j_l(k1a) h_l′(ka) − k1 j_l′(k1a) h_l(ka)]
    eps_r == "metal"：软球（Dirichlet，ψ=0，标量模型 χ→∞ 的极限）a_l = −j_l(ka)/h_l(ka)。
    lmax 默认按 Wiscombe 判据 ka + 4(ka)^{1/3} + 2 再加 4 项。
    """
    from .materials import resolve_eps     # 延迟导入：materials 依赖 scene，避免循环

    eps = resolve_eps(eps_r)
    x = k * float(radius)
    if not x > 0:
        raise ValueError("radius 与 k 必须为正")
    if lmax is None:
        lmax = int(np.ceil(x + 4.0 * x ** (1.0 / 3.0) + 2.0)) + 4
    ls = np.arange(lmax + 1)
    jl, djl = spherical_jn(ls, x), spherical_jn(ls, x, True)
    hl, dhl = _h2(ls, x), _h2(ls, x, True)
    if eps == "metal":
        return -jl / hl
    k1 = k * np.sqrt(complex(eps))
    x1 = k1 * radius
    j1, dj1 = spherical_jn(ls, x1), spherical_jn(ls, x1, True)
    return (k1 * dj1 * jl - k * j1 * djl) / (k * j1 * dhl - k1 * dj1 * hl)


def mie_scalar_far_field(eps_r, radius: float, k: float, cos_theta, lmax: int | None = None) -> np.ndarray:
    """球心在原点、单位平面波沿 +z 入射时的远场散射幅度 f(θ)，E_s ≈ f·e^{−jkr}/r（与 far_field_amplitude 同约定）。

    h_l^(2)(kr) → j^{l+1}e^{−jkr}/(kr) ⇒ f(θ) = (j/k)·Σ(2l+1)·a_l·P_l(cosθ)。
    光学定理（无耗）：σ_sca = (4π/k²)Σ(2l+1)|a_l|² = σ_ext = −(4π/k)·Im f(0)。
    """
    a = mie_scalar_coeffs(eps_r, radius, k, lmax)
    ct = np.asarray(cos_theta, dtype=np.float64)
    f = sum((2 * l + 1) * a[l] * eval_legendre(l, ct) for l in range(len(a)))
    return (1j / k) * np.asarray(f, dtype=np.complex128)
